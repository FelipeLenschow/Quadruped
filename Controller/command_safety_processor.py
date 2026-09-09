import os
import sys
import time
import threading
import numpy as np

from rclpy.node import Node
from std_msgs.msg import Bool, Float32

# Ensure project root is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Configs.config_loader import load_config
from Telemetry.estimator import projected_gravity_b
from Controller.robot_defaults import DEFAULT_STANCE_QPOS

# ---------------------------------------------------------------------------
# ANSI helpers for coloured terminal output
# ---------------------------------------------------------------------------
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


class CommandSafetyProcessor:
    """
    Centralized safety arbitrator.

    Responsibilities:
      1. Receive safety parameters from the external Supervisor heartbeat
         (separate Float32 topics under /safety/*).
      2. Evaluate robot safety every step (tilt, pitch, joint ROM).
      3. Accept a dictionary of proposed joint targets in radians.
      4. Select between the main policy targets, safety policy targets, or
         stand pose target (at zero torque) depending on safety status.
      5. Enforce safe range-of-motion limits (soft bounds clipping).
      6. Provide an Enter-to-reset mechanism for the operator.

    Agnostic to action scaling, policy dimensions, or raw reinforcement
    learning models. Works purely with proposed absolute joint targets in radians.
    """

    def __init__(self, node: Node, robot_type: str = "go2",
                 joint_names: list = None):
        self.node = node
        self.robot_type = robot_type

        # ------------------------------------------------------------------
        # 1. Load Configuration (physical limits only)
        # ------------------------------------------------------------------
        self.config = load_config()
        if not self.config:
            self.node.get_logger().error(
                "[CommandSafetyProcessor] Failed to load config. Using hardcoded safety defaults.")

        ctrl_cfg = self.config.get("control", {})
        self.saturation = ctrl_cfg.get("saturation_limit", 0.9)

        self.joint_names = joint_names or [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"
        ]

        # ------------------------------------------------------------------
        # 2. Hardware Joint Limits (from config — physical, never change)
        # ------------------------------------------------------------------
        limits = self.config.get("joint_limits", {})
        abd_limits = limits.get("abduction", {"min": -1.047, "max": 1.047})
        hip_limits = limits.get("hip_front", {"min": -1.571, "max": 3.491})
        knee_limits = limits.get("knee", {"min": -2.723, "max": -0.5})

        # Unified limit arrays (12 joints: 4 abd + 4 hip + 4 knee)
        # Used for BOTH soft clipping AND safety ROM checks.
        self.hard_min = np.array(
            [abd_limits["min"]] * 4 + [hip_limits["min"]] * 4 + [knee_limits["min"]] * 4,
            dtype=np.float32)
        self.hard_max = np.array(
            [abd_limits["max"]] * 4 + [hip_limits["max"]] * 4 + [knee_limits["max"]] * 4,
            dtype=np.float32)

        self.center = (self.hard_min + self.hard_max) / 2.0
        self.half_range = (self.hard_max - self.hard_min) / 2.0
        self.soft_min = (self.center - self.half_range * self.saturation).astype(np.float32)
        self.soft_max = (self.center + self.half_range * self.saturation).astype(np.float32)

        # Safety ROM check uses the same limits as soft clipping
        self.joint_limits_min = self.hard_min.astype(np.float64)
        self.joint_limits_max = self.hard_max.astype(np.float64)

        # Nominal Standing Pose (gating fallback)
        self.desired_qpos = DEFAULT_STANCE_QPOS.copy()

        # ------------------------------------------------------------------
        # 3. Motor Physical Parameters
        # ------------------------------------------------------------------
        motor_cfg = self.config.get("motor", {})
        self.motor_max_torque = float(motor_cfg.get("max_torque", 45.0))

        # ------------------------------------------------------------------
        # 4. Safety Parameters (from Supervisor heartbeat — defaults)
        # ------------------------------------------------------------------
        self.max_torque_percent = 0.0       # Updated by heartbeat
        self.base_tilt_limit_deg = 30.0     # Updated by heartbeat
        self.forward_tilt_limit_deg = 30.0  # Updated by heartbeat
        self.rom_safety_margin = 0.15       # Updated by heartbeat

        # Starting value only. The Supervisor owns this at runtime and sends it
        # on /safety/watchdog_timeout with the rest of the safety parameters;
        # until the first heartbeat arrives the robot is blocked anyway
        # (has_received_heartbeat), so this just avoids an undefined timeout.
        safety_cfg = self.config.get("safety", {})
        self.watchdog_timeout = float(safety_cfg.get("watchdog_timeout", 1.0))

        # Derived values (recomputed when params change)
        self._recompute_safety_limits()

        # ------------------------------------------------------------------
        # 5. Safety State Machine
        # ------------------------------------------------------------------
        self._robot_safe = True
        self._shutdown_logged = False
        self._policy_blocked = False
        # Hard e-stop latch, owned end-to-end by /safety/estop. Separate from
        # _policy_blocked on purpose: that one is the tilt/ROM/watchdog state
        # machine, this one is the operator's kill button, and each clears
        # through its own channel.
        self._estop_latched = False
        self._estop_reset_running = False
        self.active_max_torque = 0.0   # Fail-safe start: zero torque
        self.last_heartbeat_time = 0.0
        self.has_received_heartbeat = False

        # ------------------------------------------------------------------
        # 6. Heartbeat Subscriptions (separate Float32 topics)
        # ------------------------------------------------------------------
        self.node.create_subscription(
            Float32, "/safety/heartbeat",
            self._heartbeat_cb, 10)
        self.node.create_subscription(
            Float32, "/safety/max_torque_percent",
            self._max_torque_percent_cb, 10)
        self.node.create_subscription(
            Float32, "/safety/base_tilt_limit_deg",
            self._base_tilt_limit_cb, 10)
        self.node.create_subscription(
            Float32, "/safety/base_forward_tilt_limit_deg",
            self._forward_tilt_limit_cb, 10)
        self.node.create_subscription(
            Float32, "/safety/joint_rom_safety_margin",
            self._rom_margin_cb, 10)
        self.node.create_subscription(
            Float32, "/safety/watchdog_timeout",
            self._watchdog_timeout_cb, 10)
        self.node.create_subscription(
            Bool, "/safety/reset", self._safety_reset_cb, 10)
        # Explicit e-stop. The watchdog is the backstop for a console that
        # died without warning; this is the path for one that meant it, and it
        # lands within a policy step instead of after watchdog_timeout.
        self.node.create_subscription(
            Bool, "/safety/estop", self._estop_cb, 10)
        # Robot-side truth about the latch, so the console can report it
        # honestly instead of assuming its own request took effect.
        self._estop_state_pub = self.node.create_publisher(
            Bool, "/safety/estop_state", 10)

        self.node.get_logger().info(
            f"[CommandSafetyProcessor] Gated safety arbitrator ready. "
            f"Waiting for Supervisor heartbeat...")

    # ======================================================================
    # E-Stop
    # ======================================================================
    @property
    def active_max_torque(self) -> float:
        """Torque ceiling every driver reads (real, MuJoCo, Gazebo, Isaac).

        A property rather than a plain attribute so the e-stop cannot be
        undone by a later write. Both the pose and policy paths assign to this
        on every step; while the latch is set those writes are recorded but
        reads stay at zero, so no code path can raise torque without first
        going through /safety/reset.
        """
        return 0.0 if self._estop_latched else self._active_max_torque

    @active_max_torque.setter
    def active_max_torque(self, value: float):
        self._active_max_torque = float(value)

    @property
    def is_estopped(self) -> bool:
        return self._estop_latched

    def _estop_cb(self, msg: Bool):
        """Latch a hard stop. True latches; False is ignored.

        Releasing is deliberately not possible over this topic, or any other.
        It happens at the robot, by pressing ENTER on the driver's terminal -
        so re-enabling torque takes someone standing next to the machine who
        has looked at it, which is the point of an emergency stop. The console
        can stop the robot from anywhere; it cannot start it again.

        Independent of _policy_blocked and the tilt/ROM state machine. That one
        describes a robot that broke a limit; this one describes an operator
        who pressed a button. Each clears through its own channel.
        """
        if not msg.data or self._estop_latched:
            return

        self._estop_latched = True
        self.active_max_torque = 0.0
        self._estop_state_pub.publish(Bool(data=True))
        self._print_estop_banner()
        self.node.get_logger().error("[SAFETY] E-STOP latched via /safety/estop.")

        if not self._estop_reset_running:
            self._estop_reset_running = True
            threading.Thread(target=self._estop_reset_worker, daemon=True).start()

    def _estop_reset_worker(self):
        """Wait for ENTER on the driver's stdin, then release the latch."""
        try:
            input()
        except (EOFError, OSError):
            self._estop_reset_running = False
            self.node.get_logger().error(
                "[SAFETY] E-STOP latched, but this driver has no interactive "
                "stdin - nothing can release it. Restart the driver on the "
                "robot, or run it in a terminal.")
            return

        self._estop_reset_running = False
        self._estop_latched = False
        self._estop_state_pub.publish(Bool(data=False))
        print(f"{_YELLOW}{_BOLD}>>> E-STOP released. Torque ceiling back to "
              f"{self.global_max_torque:.1f} Nm.{_RESET}")
        self.node.get_logger().warn("[SAFETY] E-STOP released at the robot.")

    # ======================================================================
    # Heartbeat Callbacks
    # ======================================================================
    def _heartbeat_cb(self, msg: Float32):
        self.last_heartbeat_time = time.time()
        self.has_received_heartbeat = True

    def _max_torque_percent_cb(self, msg: Float32):
        self.max_torque_percent = msg.data
        self._recompute_safety_limits()

    def _base_tilt_limit_cb(self, msg: Float32):
        self.base_tilt_limit_deg = msg.data
        self._recompute_safety_limits()

    def _forward_tilt_limit_cb(self, msg: Float32):
        self.forward_tilt_limit_deg = msg.data
        self._recompute_safety_limits()

    def _rom_margin_cb(self, msg: Float32):
        self.rom_safety_margin = msg.data
        self._recompute_safety_limits()

    def _watchdog_timeout_cb(self, msg: Float32):
        # Ignore nonsense rather than disabling the watchdog on a bad message.
        if msg.data > 0.0:
            self.watchdog_timeout = float(msg.data)

    def _safety_reset_cb(self, msg: Bool):
        """Clear a latched stop from the console, which is always off-robot.

        Same effect as pressing Enter on the driver's stdin. If whatever caused
        the stop is still true, evaluate_safety latches it again on the next
        step, so this cannot force an unsafe robot to keep running.
        """
        if not msg.data or not self._policy_blocked:
            return
        self.node.get_logger().warn("[CommandSafetyProcessor] Safety reset received from console.")
        self._robot_safe = True
        self._shutdown_logged = False
        self._policy_blocked = False

    # ======================================================================
    # Derived Limit Computation
    # ======================================================================
    def _recompute_safety_limits(self):
        """Recalculate all derived safety thresholds from current params."""
        self.tilt_thresh = float(np.cos(np.radians(self.base_tilt_limit_deg)))

        # Joint ROM safe boundaries based on hard limits and safety margin fraction:
        # safe_min = hard_min + margin * hard_range
        # safe_max = hard_max - margin * hard_range
        joint_rom = self.joint_limits_max - self.joint_limits_min
        self.joint_safe_min = self.joint_limits_min + self.rom_safety_margin * joint_rom
        self.joint_safe_max = self.joint_limits_max - self.rom_safety_margin * joint_rom

        self.global_max_torque = (self.max_torque_percent / 100.0) * self.motor_max_torque

    # ======================================================================
    # Safety Evaluation
    # ======================================================================
    def evaluate_safety(self, state) -> tuple:
        """Run all safety checks against current state."""
        now = time.time()
        if not self.has_received_heartbeat:
            return False, "No supervisor heartbeat received yet"
        if now - self.last_heartbeat_time > self.watchdog_timeout:
            return False, f"Supervisor heartbeat lost (>{self.watchdog_timeout:.1f}s)"

        # 2. Base orientation tilt
        proj_grav = projected_gravity_b(state.imu.quaternion)
        base_tilt_cos = -proj_grav[2] / 9.81
        if base_tilt_cos < self.tilt_thresh:
            tilt_deg = float(np.degrees(np.arccos(np.clip(base_tilt_cos, -1.0, 1.0))))
            return False, (
                f"Base tilt {tilt_deg:.1f}° exceeds limit {self.base_tilt_limit_deg:.1f}°")

        # 3. Forward pitch detection
        pitch_rad = np.arctan2(proj_grav[0], -proj_grav[2])
        pitch_deg = float(np.degrees(pitch_rad))
        if pitch_deg > self.forward_tilt_limit_deg:
            return False, (
                f"Forward pitch {pitch_deg:.1f}° exceeds limit "
                f"{self.forward_tilt_limit_deg:.1f}°")

        # 4. Joint ROM boundary check
        for i in range(12):
            val = state.motorState[i].q
            if val < self.joint_safe_min[i]:
                return False, (
                    f"{self.joint_names[i]}: {val:+.3f} rad < safe min {self.joint_safe_min[i]:.3f}")
            elif val > self.joint_safe_max[i]:
                return False, (
                    f"{self.joint_names[i]}: {val:+.3f} rad > safe max {self.joint_safe_max[i]:.3f}")

        return True, ""

    # ======================================================================
    # Safety Gating and Soft Clipping
    # ======================================================================
    def process(self, proposed_targets: dict, state) -> tuple:
        """
        Arbitrate between proposed targets based on safety checks and clip output.

        Args:
            proposed_targets: dict[str, np.ndarray] of absolute targets (radians)
                              e.g. {"main": [...], "safety": [...]}
            state:            StandardState object.

        Returns:
            (final_targets: np.ndarray, max_torque: float)
        """
        # ── Hard E-Stop: outranks everything, including the safety policy ──
        # Checked before _policy_blocked because that branch hands control to
        # the recovery policy at full torque, which is not what an operator
        # hitting the kill button is asking for.
        if self._estop_latched:
            self.active_max_torque = 0.0
            return self.desired_qpos.copy(), 0.0

        # ── Latched Shutdown: stay blocked until operator presses Enter ──
        if self._policy_blocked:
            if "safety" in proposed_targets:
                self.active_max_torque = self.global_max_torque
                targets = proposed_targets["safety"]
                final_targets = np.clip(targets, self.soft_min, self.soft_max)
                return final_targets, self.active_max_torque
            else:
                self.active_max_torque = 0.0
                return self.desired_qpos.copy(), 0.0



        is_safe, reason = self.evaluate_safety(state)

        if is_safe:
            # ── Safe Mode: Use Main Policy ────────────────────────────────
            self._robot_safe = True
            self.active_max_torque = self.global_max_torque

            targets = proposed_targets.get("main", self.desired_qpos)
            final_targets = np.clip(targets, self.soft_min, self.soft_max)
            return final_targets, self.active_max_torque

        # ── Unsafe Mode: Emergency Gating ────────────────────────────────
        if not self._shutdown_logged:
            self._print_shutdown_banner(reason)
            self._shutdown_logged = True
            self._policy_blocked = True
            self._robot_safe = False
            # Background thread: wait for Enter to unlatch
            threading.Thread(target=self._reset_worker, daemon=True).start()

        if "safety" in proposed_targets:
            self.active_max_torque = self.global_max_torque
            targets = proposed_targets["safety"]
            final_targets = np.clip(targets, self.soft_min, self.soft_max)
            return final_targets, self.active_max_torque
        else:
            self.active_max_torque = 0.0
            return self.desired_qpos.copy(), 0.0

    # ======================================================================
    # UI / Reset
    # ======================================================================
    def _print_estop_banner(self):
        """Separate from _print_shutdown_banner, which offers an [ENTER] reset
        serviced by _reset_worker. No such thread runs for the e-stop, so
        reusing that banner would promise a key that does nothing."""
        print("\r" + " " * 120 + "\r", end="", flush=True)
        print(f"\n{_BOLD}{_RED}╔══════════════════════════════════════════════════════╗")
        print(f"║  ⛔  E-STOP  —  torque cut to 0 Nm                   ║")
        print(f"╠══════════════════════════════════════════════════════╣")
        print(f"║  ➜  Latched. No policy, no pose, no recovery run.    ║")
        print(f"║  ➜  Press [ENTER] HERE, on the robot, to release.    ║")
        print(f"║     The console cannot clear this.                   ║")
        print(f"╚══════════════════════════════════════════════════════╝{_RESET}\n")

    def _print_shutdown_banner(self, reason: str):
        print("\r" + " " * 120 + "\r", end="", flush=True)
        print(f"\n{_BOLD}{_RED}╔══════════════════════════════════════════════════════╗")
        print(f"║  ⚠  COMMAND SAFETY PROCESSOR — EMERGENCY SHUTDOWN ⚠  ║")
        print(f"╠══════════════════════════════════════════════════════╣")
        print(f"║  Reason: {reason[:43]:<43} ║")
        if len(reason) > 43:
            remaining = reason[43:]
            while remaining:
                chunk = remaining[:49]
                print(f"║    {chunk:<49} ║")
                remaining = remaining[49:]
        print(f"║                                                      ║")
        print(f"║  ➜  Main policy BLOCKED.                             ║")
        print(f"║  ➜  Press [ENTER] here to RESET and re-enable.       ║")
        print(f"╚══════════════════════════════════════════════════════╝{_RESET}\n")

        self.node.get_logger().error(f"[SAFETY] EMERGENCY SHUTDOWN: {reason}")

    def _reset_worker(self):
        try:
            input()
        except (EOFError, OSError):
            self.node.get_logger().warn(
                "[CommandSafetyProcessor] No interactive stdin; clear this stop "
                "from the console with 'safety = reset'.")
            return
        print(f"{_YELLOW}{_BOLD}>>> Safety RESET. Re-enabling main policy...{_RESET}")
        self.node.get_logger().info(
            "[CommandSafetyProcessor] Operator pressed ENTER. Main policy re-enabled.")
        self._robot_safe = True
        self._shutdown_logged = False
        self._policy_blocked = False

    @property
    def is_policy_blocked(self) -> bool:
        return self._policy_blocked
