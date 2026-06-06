import os
import sys
import numpy as np

from rclpy.node import Node
from std_msgs.msg import String, Float32

# Ensure project root is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Configs.config_loader import load_config
from Controller.gravity_compensator import GravityCompensator


class PoseGenerator:
    """
    Time-based joint pose interpolation engine.

    Registered inside the PolicyManager as a policy provider under the name
    "pose".  When selected by the pipeline, it returns smoothly interpolated
    absolute joint targets (radians) — no neural network inference involved.

    Supports:
      - Named poses loaded from config.yaml (stand, lie_flat, sit, …)
      - Configurable interpolation duration
      - Continuous pushup cycling (stand ↔ crouch)
      - ROS 2 command interface (/pose/command, /pose/interp_duration)
      - Status publishing for the Console progress bar (/pose/status)
    """

    # ======================================================================
    # Hardcoded fallback poses (used if config.yaml is missing entries)
    # ======================================================================
    _DEFAULT_POSES = {
        "stand": [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5],
        "lie_flat": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "sit": [0.1, -0.1, 0.1, -0.1, 1.2, 1.2, 2.0, 2.0, -2.0, -2.0, -2.5, -2.5],
    }

    # Pushup alternates between these two poses
    _PUSHUP_HIGH = "stand"
    _PUSHUP_LOW_POSE = np.array([
        0.1, -0.1, 0.1, -0.1,   # hips (same as stand)
        1.4, 1.4, 1.6, 1.6,     # thighs (crouched)
        -2.4, -2.4, -2.4, -2.4  # calves (deep bend)
    ], dtype=np.float32)

    def __init__(self, node: Node):
        self.node = node

        # ------------------------------------------------------------------
        # 1. Load pose library from config
        # ------------------------------------------------------------------
        config = load_config()
        poses_cfg = config.get("poses", {})

        self.interp_duration = float(poses_cfg.get("default_interp_duration", 3.0))

        # Build pose library: config entries override hardcoded defaults
        self.pose_library = {}
        for name, default_vals in self._DEFAULT_POSES.items():
            cfg_vals = poses_cfg.get(name, default_vals)
            self.pose_library[name] = np.array(cfg_vals, dtype=np.float32)

        # ------------------------------------------------------------------
        # 1b. Gravity Compensator (Pinocchio)
        # ------------------------------------------------------------------
        grav_cfg = config.get("gravity_compensation", {})
        urdf_path = grav_cfg.get("urdf_path", "Configs/go2_model/go2.urdf")
        self._gravity_comp = GravityCompensator(urdf_path)

        # ------------------------------------------------------------------
        # 2. Interpolation state
        # ------------------------------------------------------------------
        self._start_qpos = None          # 12-dim: where we started
        self._end_qpos = None            # 12-dim: where we're going
        self._start_time = None          # sim-time at interpolation start
        self._pending_target = None      # set by command_pose, consumed by step()
        self._duration = self.interp_duration
        self._current_targets = None     # last computed targets
        self._pose_name = "none"         # name of active/last pose
        self._initialized = False        # True after first step() call

        # Pushup state
        self._pushup_active = False
        self._pushup_phase = "high"      # "high" or "low"

        # ------------------------------------------------------------------
        # 3. ROS 2 Interface
        # ------------------------------------------------------------------
        self.node.create_subscription(
            String, "/pose/command",
            self._pose_command_cb, 10)
        self.node.create_subscription(
            Float32, "/pose/interp_duration",
            self._interp_duration_cb, 10)

        # Status publisher: "pose_name|progress_fraction" (e.g. "stand|0.75")
        self._status_pub = self.node.create_publisher(
            String, "/pose/status", 10)

        self.node.get_logger().info(
            f"[PoseGenerator] Initialized with {len(self.pose_library)} poses: "
            f"{list(self.pose_library.keys())}")

    # ======================================================================
    # ROS 2 Callbacks
    # ======================================================================
    def _pose_command_cb(self, msg: String):
        """Handle incoming pose commands from the Console."""
        name = msg.data.strip().lower()
        self.command_pose(name)

    def _interp_duration_cb(self, msg: Float32):
        """Update interpolation duration."""
        if msg.data > 0.0:
            self.interp_duration = msg.data
            self.node.get_logger().info(
                f"[PoseGenerator] Interpolation duration set to {self.interp_duration:.1f}s")

    # ======================================================================
    # Public API
    # ======================================================================
    def command_pose(self, name: str):
        """
        Start interpolation to a named pose.

        If 'pushup' is commanded, enters continuous cycling mode.
        Any other pose command cancels an active pushup cycle.
        """
        if name == "pushup":
            self._pushup_active = True
            self._pushup_phase = "low"  # Start by going to crouch
            self._pose_name = "pushup"
            self._pending_target = self._PUSHUP_LOW_POSE.copy()
            self.node.get_logger().info(
                "[PoseGenerator] Pushup mode: continuous cycle started")
            return

        # Any non-pushup command cancels pushup mode
        self._pushup_active = False

        if name not in self.pose_library:
            self.node.get_logger().warn(
                f"[PoseGenerator] Unknown pose '{name}'. "
                f"Available: {list(self.pose_library.keys())}")
            return

        self._pose_name = name
        self._pending_target = self.pose_library[name].copy()
        self.node.get_logger().info(
            f"[PoseGenerator] Interpolating to '{name}' over {self.interp_duration:.1f}s")

    def _step_interpolation(self, state, current_time: float) -> np.ndarray:
        """
        Internal: compute the current interpolated joint targets.

        Args:
            state:        StandardState (used to capture current joint pos on first call)
            current_time: Simulation or wall-clock time in seconds.

        Returns:
            np.ndarray: 12-dim absolute joint position targets.
        """
        # First-ever call: initialize from current joint positions
        if not self._initialized:
            current_qpos = np.array(
                [state.motorState[i].q for i in range(12)], dtype=np.float32)
            self._current_targets = current_qpos.copy()
            self._initialized = True

        # Check for a pending pose command — start interpolation now
        # that we have the correct time base (sim_time, not wall-clock)
        if self._pending_target is not None:
            self._start_qpos = self._current_targets.copy()
            self._end_qpos = self._pending_target
            self._duration = self.interp_duration
            self._start_time = current_time
            self._pending_target = None

        # No active interpolation → hold last targets
        if self._start_time is None:
            self._publish_status(1.0)
            return self._current_targets

        # Compute interpolation progress
        elapsed = current_time - self._start_time
        alpha = np.clip(elapsed / self._duration, 0.0, 1.0)

        self._current_targets = (
            self._start_qpos + alpha * (self._end_qpos - self._start_qpos)
        )

        # Publish status for Console progress bar
        self._publish_status(alpha)

        # Check completion
        if alpha >= 1.0:
            self._start_time = None  # Mark interpolation as complete

            # Pushup cycling: immediately start next phase
            if self._pushup_active:
                if self._pushup_phase == "low":
                    self._pushup_phase = "high"
                    self._pending_target = self.pose_library.get(
                        self._PUSHUP_HIGH,
                        np.array(self._DEFAULT_POSES["stand"], dtype=np.float32)
                    ).copy()
                else:
                    self._pushup_phase = "low"
                    self._pending_target = self._PUSHUP_LOW_POSE.copy()

        return self._current_targets
    def step(self, state, current_time: float) -> dict:
        """
        Compute the current interpolated joint targets and gravity compensation torques.

        Called by PolicyManager.step_single() when the pipeline mode is "pose".

        Args:
            state:        StandardState (used to capture current joint pos on first call)
            current_time: Simulation or wall-clock time in seconds.

        Returns:
            dict: {"q_des": 12-dim position targets, "tau_ff": 12-dim feedforward torques}
        """
        q_des = self._step_interpolation(state, current_time)

        # Compute gravity compensation using ACTUAL joint positions (not targets)
        # and current contact sensor data
        joint_pos_actual = np.array(
            [state.motorState[i].q for i in range(12)], dtype=np.float64)
        contact = np.array(state.feet_contact, dtype=np.float64)

        tau_ff = self._gravity_comp.compute(
            quat_xyzw=state.imu.quaternion,
            joint_pos=joint_pos_actual,
            contact_flags=contact
        )

        return {"q_des": q_des, "tau_ff": tau_ff}


    def sync_to_current(self):
        """
        Force the PoseGenerator to re-read actual joint positions on the
        next step() call.  Called when switching back to pose mode after
        the NN policy has moved the robot — prevents a sudden jump to
        the previously held targets.
        """
        self._initialized = False
        self._start_time = None
        self._pending_target = None
        self._pushup_active = False
        self._pose_name = "holding"

    @property
    def is_active(self) -> bool:
        """True while an interpolation is in progress."""
        return self._start_time is not None

    @property
    def is_complete(self) -> bool:
        """True when settled at a target pose (not interpolating)."""
        return self._initialized and self._start_time is None

    @property
    def current_pose_name(self) -> str:
        """Name of the active or last-commanded pose."""
        return self._pose_name

    # ======================================================================
    # Internal Helpers
    # ======================================================================
    def _begin_interpolation(self, target_qpos: np.ndarray, current_time: float):
        """Start a new interpolation from current targets to the given target."""
        self._start_qpos = (
            self._current_targets.copy()
            if self._current_targets is not None
            else target_qpos.copy()
        )
        self._end_qpos = np.array(target_qpos, dtype=np.float32)
        self._duration = self.interp_duration
        self._start_time = current_time

    def _publish_status(self, progress: float):
        """Publish status string for the Console's progress bar."""
        status = self._pose_name
        if self._pushup_active:
            status = f"pushup ({self._pushup_phase})"

        if progress >= 1.0:
            label = "complete"
        else:
            label = f"{progress * 100:.0f}%"

        msg = String()
        msg.data = f"{status}|{progress:.3f}|{label}"
        self._status_pub.publish(msg)
