import os
import numpy as np
from std_msgs.msg import String
from Telemetry.telemetry import TelemetryManager
from Controller.policy_manager import PolicyManager
from Controller.command_safety_processor import CommandSafetyProcessor
from Controller.distributor import Distributor
from Configs.config_loader import load_config


class LocomotionPipeline:
    """
    Centralized pipeline encapsulating Telemetry, Policy Selection/Inference,
    Safety Arbitration, and Command Distribution.

    Ensures identical execution across MuJoCo, Gazebo, Isaac Sim, and Physical Hardware.
    """
    def __init__(self, node, robot_type="go2", checkpoint=None, obs_dim=49,
                 use_estimator=False, joint_names=None, sim_dt=0.001):
        self.node = node
        self.robot_type = robot_type

        self.config = load_config()
        self.ctrl_cfg = self.config.get("control", {})
        safety_cfg = self.config.get("safety", {})

        # 1. Telemetry Manager
        self.telemetry = TelemetryManager(node, joint_names, use_estimator=use_estimator)

        # 2. Policy Manager (Unified registry for policy runners)
        self.policy_manager = PolicyManager(node, robot_type=robot_type, obs_dim=obs_dim)

        # Register Main Policy (policy under test selected by the launcher)
        if checkpoint:
            self.policy_manager.load_policy("main", checkpoint)
        else:
            self.node.get_logger().warn(
                "[LocomotionPipeline] No main policy checkpoint provided. Policy runner disabled.")

        # Register Safety Policy (backup recovery policy configured in config.yaml)
        safety_policy_path = safety_cfg.get("safety_policy_path", "")
        if safety_policy_path:
            loaded = self.policy_manager.load_policy("safety", safety_policy_path)
            if not loaded:
                self.node.get_logger().warn(
                    "[LocomotionPipeline] Safety policy failed to load. "
                    "Robot will DISABLE directly on safety violations.")
        else:
            self.node.get_logger().info(
                "[LocomotionPipeline] No safety policy configured. "
                "Robot will DISABLE directly on safety violations.")

        self.mj_to_isaac = list(range(12))  # Standard mapping
        self.sim_dt = sim_dt
        self.decimation = self.ctrl_cfg.get("decimation", 4)
        # QUADRUPED_POLICY_HZ, exported by launcher.py from the checkpoint's own env.yaml, wins
        # over the static config: it is the rate the policy was actually trained at. The config's
        # decimation assumes a 50 Hz position policy, which silently runs a 200 Hz torque policy
        # at a quarter speed -- and torque mode has no PD loop to paper over the held command.
        # policy_dt follows, which also matters for the paper observation's linear-acceleration
        # term: it is a finite difference divided by dt, so a wrong dt scales the input directly.
        _hz = os.environ.get("QUADRUPED_POLICY_HZ")
        if _hz:
            _dec = max(1, int(round(1.0 / (float(_hz) * self.sim_dt))))
            if _dec != self.decimation:
                print(f"[Pipeline] Policy rate from checkpoint: {float(_hz):.0f} Hz -> "
                      f"decimation {self.decimation} -> {_dec} "
                      f"(physics {1.0/self.sim_dt:.0f} Hz)")
            self.decimation = _dec
        self.policy_dt = self.decimation * self.sim_dt

        # 3. Command Safety Processor (safety checking + arbitration)
        self.safety_processor = CommandSafetyProcessor(
            node, robot_type=robot_type, joint_names=joint_names)

        # 4. Distributor (hardware/ROS command output)
        self.distributor = Distributor(node, joint_names=joint_names)

        # 5. Pose Generator (registered inside PolicyManager)
        self.policy_manager.register_pose_generator(node)

        # 6. Pipeline Mode: "pose" (default — robot starts in pose mode)
        #    or "policy" (NN inference mode)
        self.mode = "pose"
        self.node.create_subscription(
            String, "/pipeline/mode", self._mode_cb, 10)

        # Nominal standing pose (default fallback)
        self.desired_qpos = np.array([
            0.1, -0.1, 0.1, -0.1,  # hips
            0.8, 0.8, 1.0, 1.0,    # thighs
            -1.5, -1.5, -1.5, -1.5  # calves
        ], dtype=np.float32)

        self.latest_targets = self.desired_qpos.copy()
        self.step_counter = 0
        self._pose_heartbeat_was_ok = False  # Track heartbeat lost→alive transitions

        # Mode Transition State
        self.mode_transition_active = False
        self.mode_transition_start_time = None
        self.mode_transition_duration = 3.0  # Smooth transition duration in seconds
        self.mode_transition_start_targets = self.desired_qpos.copy()

    def _mode_cb(self, msg: String):
        """Handle pipeline mode switch commands from the Console."""
        new_mode = msg.data.strip().lower()
        if new_mode in ("pose", "policy"):
            if new_mode != self.mode:
                self.node.get_logger().info(
                    f"[Pipeline] Mode switched: {self.mode} → {new_mode}")
                self.mode = new_mode

                # Start smooth transition from current targets
                self.mode_transition_active = True
                self.mode_transition_start_time = None
                self.mode_transition_start_targets = self.latest_targets.copy()

                # Clear safety latch when entering pose mode — the pose
                # generator IS the recovery mechanism for unsafe states.
                if new_mode == "pose":
                    self.safety_processor._policy_blocked = False
                    self.safety_processor._shutdown_logged = False
                    self.safety_processor._robot_safe = True
                    # Re-sync pose generator to current joint positions
                    # so it doesn't jump to the old cached targets.
                    pose_gen = self.policy_manager.policies.get("pose")
                    if pose_gen:
                        pose_gen.sync_to_current()
        else:
            self.node.get_logger().warn(
                f"[Pipeline] Unknown mode '{new_mode}'. Use 'pose' or 'policy'.")

    def step(self, raw_state_kwargs, cmd_vel, sim_time):
        """
        Executes one step of the pipeline.

        Args:
            raw_state_kwargs: dict containing q, dq, quat, gyro, accel, pos, vel, contact, etc.
            cmd_vel: list/array of velocity commands [vx, vy, wz, unused]
            sim_time: current simulation or physical time

        Returns:
            latest_targets (np.ndarray): The target joint positions to send to the motors.
        """
        # Determine if this is a policy inference step (e.g., 50Hz)
        is_policy_step = (self.step_counter % self.decimation) == 0
        self.step_counter += 1

        # 1. Standardize State
        raw_state_kwargs['update_estimator'] = is_policy_step
        state = self.telemetry.process_state(**raw_state_kwargs)

        # 2. Policy Inference & Command Processing
        if is_policy_step:

            if self.mode == "pose":
                # ── POSE MODE ────────────────────────────────────────
                # Bypass ROM/tilt safety checks — the whole point of the
                # pose generator is to escape unsafe states (e.g., lying
                # on the ground where joints are at their limits).
                #
                # We still require the Console heartbeat to be alive
                # (watchdog) and apply soft joint clipping.
                # ─────────────────────────────────────────────────────
                import time as _time
                sp = self.safety_processor
                heartbeat_ok = (
                    sp.has_received_heartbeat and
                    (_time.time() - sp.last_heartbeat_time) <= sp.watchdog_timeout
                )

                # Detect heartbeat lost→alive transition (Console restart)
                # and re-sync pose generator to actual joint positions.
                if heartbeat_ok and not self._pose_heartbeat_was_ok:
                    pose_gen = self.policy_manager.policies.get("pose")
                    if pose_gen:
                        pose_gen.sync_to_current()
                        self.node.get_logger().info(
                            "[Pipeline] Heartbeat restored in pose mode — "
                            "synced to current joint positions.")
                self._pose_heartbeat_was_ok = heartbeat_ok

                if heartbeat_ok and "pose" in self.policy_manager.policies:
                    targets = self.policy_manager.step_single(
                        "pose", state, cmd_vel, self.mj_to_isaac,
                        current_time=sim_time,
                        dt=self.policy_dt
                    )
                    # Soft-clip to joint limits (still enforced)
                    final_targets = np.clip(targets, sp.soft_min, sp.soft_max)
                    max_torque = sp.global_max_torque
                    # Update the safety processor's active torque so the
                    # driver's PD loop (which reads it directly) applies force.
                    sp.active_max_torque = max_torque
                else:
                    # No heartbeat → zero torque (same fail-safe as policy mode)
                    final_targets = self.desired_qpos.copy()
                    max_torque = 0.0
                    sp.active_max_torque = 0.0

            else:
                # ── POLICY MODE ──────────────────────────────────────
                # Full safety evaluation: ROM, tilt, and heartbeat checks.
                # ─────────────────────────────────────────────────────
                is_safe, _ = self.safety_processor.evaluate_safety(state)

                if not is_safe:
                    active_policy = "safety"
                else:
                    active_policy = "main"

                proposed_targets = {}
                if active_policy in self.policy_manager.policies:
                    targets = self.policy_manager.step_single(
                        active_policy, state, cmd_vel, self.mj_to_isaac,
                        current_time=sim_time,
                        dt=self.policy_dt
                    )
                    key = "safety" if active_policy == "safety" else "main"
                    proposed_targets[key] = targets

                if getattr(self.policy_manager, "last_output_is_torque", False):
                    # The safety processor arbitrates and clamps JOINT ANGLES; running torques
                    # through it would clip them against soft_min/soft_max in radians, which is
                    # meaningless. Torque mode therefore bypasses it and is limited by the
                    # driver's effort limit instead. The tilt/ROM/heartbeat evaluation above
                    # still ran, and still selects the safety policy or disables on violation --
                    # what is skipped is only the joint-angle clamp.
                    final_targets = proposed_targets.get(
                        "safety" if active_policy == "safety" else "main",
                        np.zeros(12, dtype=np.float32),
                    )
                    max_torque = self.safety_processor.global_max_torque
                    self.safety_processor.active_max_torque = max_torque
                else:
                    final_targets, max_torque = self.safety_processor.process(
                        proposed_targets=proposed_targets,
                        state=state
                    )

            # ── MODE TRANSITION INTERPOLATION ────────────────────
            # Only meaningful between two position targets: blending a stance pose (radians)
            # with a torque command (N*m) would produce neither.
            if self.mode_transition_active and not getattr(
                self.policy_manager, "last_output_is_torque", False
            ):
                if self.mode_transition_start_time is None:
                    self.mode_transition_start_time = sim_time
                
                elapsed = sim_time - self.mode_transition_start_time
                alpha = np.clip(elapsed / self.mode_transition_duration, 0.0, 1.0)
                
                final_targets = (1.0 - alpha) * self.mode_transition_start_targets + alpha * final_targets
                
                if alpha >= 1.0:
                    self.mode_transition_active = False
                    self.mode_transition_start_time = None

            # Interpolating between a position target and a torque command is meaningless, so
            # mode transitions are skipped in torque mode (handled above by leaving
            # mode_transition_active untouched -- see the guard in the interpolation block).
            self.latest_targets = final_targets
            self.output_is_torque = getattr(
                self.policy_manager, "last_output_is_torque", False
            )
            self.distributor.send(final_targets, max_torque)

        # 3. Telemetry Publishing
        if is_policy_step:
            self.telemetry.publish(sim_time=sim_time, state=state)

        return self.latest_targets

