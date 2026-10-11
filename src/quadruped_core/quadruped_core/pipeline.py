import json
import os
import time
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, String
from quadruped_core.telemetry.telemetry import TelemetryManager
from quadruped_core.controller.policy_manager import PolicyManager
from quadruped_core.controller.robot_defaults import DEFAULT_STANCE_QPOS, PROTECTIONS
from quadruped_core.controller.command_safety_processor import CommandSafetyProcessor
from quadruped_core.controller.distributor import Distributor
from quadruped_core.controller.command_slew import CommandSlewLimiter
from quadruped_core.controller.policy_runner import WTW_GAITS
from quadruped_core.telemetry.estimator import projected_gravity_b
from quadruped_core.config_loader import load_config


class LocomotionPipeline:
    """
    Centralized pipeline encapsulating Telemetry, Policy Selection/Inference,
    Safety Arbitration, and Command Distribution.

    Ensures identical execution across MuJoCo, Gazebo, Isaac Sim, and Physical Hardware.
    """
    def __init__(self, node, robot_type="go2", checkpoint=None, obs_dim=49,
                 use_estimator=False, joint_names=None, sim_dt=0.001, safety_switch=True):
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

        # Safety policy (policies.json safety_policy): loaded now, since an emergency can't wait
        # for a file to load. It takes over when the active policy's checks trip, until the robot lies
        # still, then the robot goes numb.
        # safety_switch=False (eval sweeps, which clear every trip themselves): no safety policy and no
        # numb, a trip only cuts torque until it is cleared.
        pm = self.policy_manager
        self.safety_switch = safety_switch
        self.safety_name = pm.safety_policy if safety_switch else ""
        if self.safety_name:
            if self.safety_name not in pm.catalog:
                self.node.get_logger().error(
                    f"[LocomotionPipeline] safety_policy '{self.safety_name}' is not in policies.json.")
                self.safety_name = ""
            elif not pm.load_policy(self.safety_name, pm.catalog[self.safety_name]["path"]):
                self.safety_name = ""
        if safety_switch and not self.safety_name:
            self.node.get_logger().warn(
                "[LocomotionPipeline] No safety policy: the robot goes numb as soon as a check trips.")
        self.safety_timeout = float(safety_cfg.get("safety_policy_timeout_s", 10.0))
        self.lying_hold_s = float(safety_cfg.get("lying_hold_s", 1.0))
        self.lying_cos = float(np.cos(np.radians(float(safety_cfg.get("lying_tilt_deg", 20.0)))))
        self.lying_pose_err = float(safety_cfg.get("lying_pose_err", 0.3))
        self.lying_joint_vel = float(safety_cfg.get("lying_joint_vel", 1.0))
        self.lie_pose = np.array(
            (self.config.get("poses") or {}).get("lie_flat", DEFAULT_STANCE_QPOS), dtype=np.float32)
        # Safety policy running time and how long the robot has lain still, counted in policy steps so a
        # jump of the robot's wall clock (dead RTC, NTP) cannot cut the lie-down short or stretch it.
        self._safety_elapsed = None
        self._lying_for = 0.0

        # The policy that runs in policy mode, switched by name on /pipeline/policy.
        self.active_policy = "main" if "main" in pm.policies else None
        self._pending_policy = None
        self.policy_error = None
        self.policy_switch_blend = float(self.config.get("policy_switch_blend_s", 0.5))
        self.node.create_subscription(String, "/pipeline/policy", self._policy_cb, 10)

        self.mj_to_isaac = list(range(12))  # Standard mapping
        self.sim_dt = sim_dt
        self.decimation = self.ctrl_cfg.get("decimation", 4)
        self.policy_dt = self.decimation * self.sim_dt
        self.command_slew = CommandSlewLimiter.from_config(self.config.get("command_slew"))
        self._slew_time = None
        if self.command_slew.enabled:
            self.node.get_logger().info(
                f"[LocomotionPipeline] Command slew limit: accel {self.command_slew.accel.tolist()}, "
                f"decel {self.command_slew.decel.tolist()} (vx, vy m/s^2; wz rad/s^2).")

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
        # Walk These Ways gait commands (quadruped_operator/gait_teleop.py): the 8 or 9 values of
        # PolicyRunner.gait_command. Policies without a gait input ignore it.
        self.node.create_subscription(
            Float32MultiArray, "/gait_command", self._gait_cb, 10)
        self.node.create_subscription(
            Float32MultiArray, "/gait_command/override", self._gait_override_cb, 10)
        self._gait_base = None
        self._gait_override_time = None

        cmd_cfg = self.config.get("cmd_vel") or {}
        self.cmd_timeout = float(cmd_cfg.get("timeout", 0.5))
        self._cmd = [0.0, 0.0, 0.0, 0.0]
        self._cmd_time = None
        self.node.create_subscription(Twist, "/cmd_vel", self._cmd_vel_cb, 10)

        self._state_pub = self.node.create_publisher(String, "/robot_state", 10)
        self._state_period = 1.0 / float(cmd_cfg.get("robot_state_rate", 10.0))
        self._state_time = 0.0

        # Nominal standing pose (default fallback)
        self.desired_qpos = DEFAULT_STANCE_QPOS.copy()

        self.latest_targets = self.desired_qpos.copy()
        self.step_counter = 0
        self._pose_heartbeat_was_ok = False  # Track heartbeat lost→alive transitions
        self._was_estopped = False           # Track e-stop latched→released
        self._was_numb = False

        # Mode Transition State
        self.mode_transition_active = False
        self.mode_transition_start_time = None
        self._last_measured_q = None
        self.mode_transition_duration = 3.0  # Smooth transition duration in seconds
        self.transition_duration = self.mode_transition_duration
        self.mode_transition_start_targets = self.desired_qpos.copy()

    def reset(self):
        """Drop all per-episode state. Call whenever the simulation is reset.

        latest_targets is the important one: step() only recomputes targets on a policy step
        (every `decimation` calls) and returns this cached array on every other call. So once a
        NaN lands in it, every later step returns that NaN -- across mj_resetData, across a new
        eval run, forever -- which is why one unstable test used to abort the whole sweep at
        t=0.00 with no chance to recover.
        """
        self.policy_manager.reset_policy_state()
        self.command_slew.reset()
        self._slew_time = None
        self.latest_targets = self.desired_qpos.copy()
        self.step_counter = 0
        self.mode_transition_active = False
        self.mode_transition_start_time = None
        self.mode_transition_start_targets = self.desired_qpos.copy()
        self._safety_elapsed = None
        self._lying_for = 0.0
        self._pending_policy = None

    def _wtw_runners(self):
        return [r for r in self.policy_manager.policies.values() if getattr(r, "_obs_layout", None) == "wtw"]

    def _gait_cb(self, msg: Float32MultiArray):
        if len(msg.data) not in (8, 9):
            return
        self._gait_base = list(msg.data)
        if self._gait_override_time is None:
            for runner in self._wtw_runners():
                runner.set_gait_vector(msg.data)

    def _gait_override_cb(self, msg: Float32MultiArray):
        if len(msg.data) not in (8, 9):
            return
        runners = self._wtw_runners()
        if self._gait_override_time is None and self._gait_base is None and runners:
            self._gait_base = runners[0].gait_command.tolist()
        self._gait_override_time = time.monotonic()
        for runner in runners:
            runner.set_gait_vector(msg.data)

    def _release_gait_override(self):
        if self._gait_override_time is None or time.monotonic() - self._gait_override_time <= self.cmd_timeout:
            return
        self._gait_override_time = None
        if self._gait_base is not None:
            for runner in self._wtw_runners():
                runner.set_gait_vector(self._gait_base)

    def _cmd_vel_cb(self, msg: Twist):
        self._cmd = [msg.linear.x, msg.linear.y, msg.angular.z, 0.0]
        self._cmd_time = time.monotonic()

    @property
    def cmd_age(self):
        return None if self._cmd_time is None else time.monotonic() - self._cmd_time

    @property
    def cmd_vel(self):
        age = self.cmd_age
        if age is None or age > self.cmd_timeout:
            return [0.0, 0.0, 0.0, 0.0]
        return list(self._cmd)

    def _publish_robot_state(self, state, cmd_vel, sim_time):
        now = time.monotonic()
        if now - self._state_time < self._state_period:
            return
        self._state_time = now
        sp = self.safety_processor
        pose_gen = self.policy_manager.policies.get("pose")
        pg = projected_gravity_b(state.imu.quaternion)
        tilt = float(np.degrees(np.arccos(np.clip(-pg[2] / np.linalg.norm(pg), -1.0, 1.0))))
        vel = [float(v) for v in (state.base_lin_vel or [0.0, 0.0, 0.0])[:2]]
        yaw_rate = float(state.imu.gyroscope[2]) if state.imu.gyroscope else 0.0
        cmd = [round(float(c), 3) for c in cmd_vel[:3]]
        moving = max(abs(c) for c in cmd) > 0.05 or float(np.hypot(*vel)) > 0.15
        pose = pose_gen.current_pose_name if pose_gen else "none"
        pose_done = bool(pose_gen.is_complete) if pose_gen else False

        if tilt > 60.0:
            posture = "fallen"
        elif sp.active_max_torque <= 0.1 or (self.mode != "policy" and pose == "lie_flat"):
            posture = "lying"
        elif self.mode == "policy":
            posture = "walking" if moving else "standing"
        else:
            posture = pose

        gait = None
        runners = self._wtw_runners()
        if runners:
            g = runners[0].gait_command.tolist()
            names = ["height", "frequency", "phase", "offset", "bound", "swing", "pitch", "width", "duty"]
            gait = {n: round(float(v), 3) for n, v in zip(names, g)}
            gait["name"] = min(WTW_GAITS, key=lambda k: float(np.abs(np.array(WTW_GAITS[k]) - g[2:5]).sum()))
            gait["override"] = self._gait_override_time is not None

        msg = String()
        msg.data = json.dumps({
            "t": round(float(sim_time), 3),
            "mode": self.mode,
            "transition": self.mode_transition_active,
            "posture": posture,
            "pose": pose,
            "pose_done": pose_done,
            "moving": moving,
            "estop": sp.is_estopped,
            "safety_blocked": sp._policy_blocked,
            "safety_policy_active": self._safety_elapsed is not None and not sp.is_numb,
            "trip_reason": sp.trip_reason,
            "numb": sp.is_numb,
            "numb_reason": sp.numb_reason,
            "policy": self.active_policy,
            "policies": self.policy_manager.available,
            "policy_loading": self._pending_policy,
            "policy_error": self.policy_error,
            "protections": list(self.policy_manager.spec(self.active_policy)["protections"])
            if self.active_policy else [],
            "torque_on": sp.active_max_torque > 0.1,
            "tilt_deg": round(tilt, 1),
            "velocity": [round(v, 3) for v in vel] + [round(yaw_rate, 3)],
            "cmd": cmd,
            "gait": gait,
            "heartbeat": sp.heartbeat_stats.summary(time.monotonic()),
        })
        self._state_pub.publish(msg)

    def _mode_cb(self, msg: String):
        """Handle pipeline mode switch commands from the Console."""
        new_mode = msg.data.strip().lower()

        # An e-stop is not a pause. Arming the policy while latched would let
        # it start the instant someone pressed ENTER at the robot, on a robot
        # still lying on the floor.
        if new_mode == "policy" and self.safety_processor.is_estopped:
            self.node.get_logger().warn(
                "[Pipeline] Refusing policy mode: E-STOP is latched. "
                "Release it at the robot first.")
            return
        if new_mode == "policy" and self.safety_processor.is_numb:
            self.node.get_logger().warn("[Pipeline] Refusing policy mode: the robot is numb. Release it first.")
            return

        if new_mode in ("pose", "policy"):
            if new_mode != self.mode:
                self.node.get_logger().info(
                    f"[Pipeline] Mode switched: {self.mode} → {new_mode}")
                self.mode = new_mode

                # Start smooth transition from current targets
                self.mode_transition_active = True
                self.mode_transition_start_time = None
                self.transition_duration = self.mode_transition_duration

                was_limp = self.safety_processor.active_max_torque <= 0.1
                if was_limp and self._last_measured_q is not None:
                    self.mode_transition_start_targets = self._last_measured_q.copy()
                    self.node.get_logger().info(
                        "[Pipeline] Torque was off - blending from measured joints.")
                else:
                    self.mode_transition_start_targets = self.latest_targets.copy()

                # Clear safety latch when entering pose mode — the pose
                # generator IS the recovery mechanism for unsafe states.
                if new_mode == "pose":
                    self.safety_processor._policy_blocked = False
                    self.safety_processor._shutdown_logged = False
                    self.safety_processor._robot_safe = True
                    self.safety_processor.trip_reason = ""
                    self._safety_elapsed = None
                    # Re-sync pose generator to current desired joint positions
                    # so it doesn't jump to the old cached targets.
                    pose_gen = self.policy_manager.policies.get("pose")
                    if pose_gen:
                        pose_gen.sync_to_current(
                            self.mode_transition_start_targets)
        else:
            self.node.get_logger().warn(
                f"[Pipeline] Unknown mode '{new_mode}'. Use 'pose' or 'policy'.")

    def _policy_cb(self, msg: String):
        """Pick the policy that runs in policy mode. Loads it in the background if needed; the switch
        happens on the first policy step after it is ready."""
        name = msg.data.strip()
        if "main" in self.policy_manager.specs and self.policy_manager.catalog.get(name) is self.policy_manager.specs["main"]:
            name = "main"
        sp = self.safety_processor
        if sp.is_estopped or sp.is_numb or sp.is_policy_blocked:
            self.node.get_logger().warn(
                f"[Pipeline] Refusing policy '{name}': e-stop, numb or the safety switch is on.")
            return
        if name == self.active_policy:
            self._pending_policy = None
            return
        if not self.policy_manager.request(name):
            self.node.get_logger().warn(
                f"[Pipeline] Unknown policy '{name}'. Available: {self.policy_manager.available}")
            return
        self._pending_policy = name
        self.policy_error = None
        self.node.get_logger().info(f"[Pipeline] Switching to policy '{name}'...")

    def _advance_policy_switch(self):
        name = self._pending_policy
        if name is None:
            return
        pm = self.policy_manager
        # _policy_cb runs on the executor's thread: only clear the request if no newer one replaced it.
        if name in pm.load_errors:
            self.policy_error = f"{name}: {pm.load_errors[name]}"
            self.node.get_logger().error(f"[Pipeline] Policy switch failed: {self.policy_error}")
            if self._pending_policy == name:
                self._pending_policy = None
            return
        if name not in pm.policies:
            return
        old = self.active_policy
        pm.policies[name].reset_history()
        self.active_policy = name
        if self._pending_policy == name:
            self._pending_policy = None
        if self.mode == "policy":
            self.mode_transition_active = True
            self.mode_transition_start_time = None
            self.mode_transition_start_targets = self.latest_targets.copy()
            self.transition_duration = self.policy_switch_blend
        # Free whatever is neither running nor wanted, including a policy asked for and then replaced.
        keep = {name, self._pending_policy, "main", "pose", self.safety_name}
        for stale in [n for n in list(pm.policies) if n not in keep and not pm.is_loading(n)]:
            pm.unload(stale)
        self.node.get_logger().info(
            f"[Pipeline] Policy '{old}' -> '{name}', protections {list(pm.spec(name)['protections']) or 'none'}.")

    def _is_lying(self, state) -> bool:
        pg = projected_gravity_b(state.imu.quaternion)
        level = -pg[2] / max(float(np.linalg.norm(pg)), 1e-6) >= self.lying_cos
        q = np.array([state.motorState[i].q for i in range(12)], dtype=np.float32)
        dq = np.array([state.motorState[i].dq for i in range(12)], dtype=np.float32)
        return bool(level and np.mean(np.abs(q - self.lie_pose)) < self.lying_pose_err
                    and np.max(np.abs(dq)) < self.lying_joint_vel)

    def _watch_safety_policy(self, state):
        """Once per policy step while the safety switch is on: numb once the robot lies still, or after
        the timeout."""
        sp = self.safety_processor
        if sp.is_numb or not self.safety_switch:
            return
        if self.safety_name not in self.policy_manager.policies:
            sp.set_numb(f"safety switch ({sp.trip_reason}), no safety policy")
            return
        self._safety_elapsed += self.policy_dt
        self._lying_for = self._lying_for + self.policy_dt if self._is_lying(state) else 0.0
        if self._lying_for >= self.lying_hold_s:
            sp.set_numb("lying after the safety switch")
        elif self._safety_elapsed >= self.safety_timeout:
            sp.set_numb(f"safety policy ran {self.safety_timeout:.0f} s without lying still")

    def step(self, raw_state_kwargs, cmd_vel, sim_time, send_cb=None):
        """
        Executes one step of the pipeline.

        Args:
            raw_state_kwargs: dict containing q, dq, quat, gyro, accel, pos, vel, contact, etc.
            cmd_vel: list/array of velocity commands [vx, vy, wz, unused]
            sim_time: current simulation or physical time
            send_cb: optional callable(targets), run on a policy step the moment the final
                targets exist -- before /commands/joint_commands and the telemetry are
                published. The real driver writes to the motors here, so ROS publishing
                (~2 ms on the Jetson) no longer sits between the state read and the command.

        Returns:
            latest_targets (np.ndarray): The target joint positions to send to the motors.
        """
        # Determine if this is a policy inference step (e.g., 50Hz)
        is_policy_step = (self.step_counter % self.decimation) == 0
        self.step_counter += 1
        if cmd_vel is None:
            cmd_vel = self.cmd_vel

        # 1. Standardize State
        raw_state_kwargs['update_estimator'] = is_policy_step
        state = self.telemetry.process_state(**raw_state_kwargs)


        self._last_measured_q = np.array(
            [state.motorState[i].q for i in range(12)], dtype=np.float32)

        # 2. Policy Inference & Command Processing
        if is_policy_step:

            # ── E-STOP RELEASE ───────────────────────────────────────
            # The operator pressed ENTER on the driver. Torque is about to
            # come back on a robot that has been lying limp on the floor,
            # and the pose generator is still holding whatever it was told
            # before the collapse — usually a stand. Handing that back would
            # snap every joint from the floor to standing at full torque.
            #
            # Re-seed from the measured joints instead (sync_to_current with
            # no argument), which is the honest reading precisely because the
            # robot was limp: with kp=0 there was no PD error to account for.
            # The robot then holds exactly where it landed and waits for a
            # pose command.
            # ─────────────────────────────────────────────────────────
            if self._was_estopped and not self.safety_processor.is_estopped:
                self._was_estopped = False
                self.mode = "pose"
                self.mode_transition_active = False
                pose_gen = self.policy_manager.policies.get("pose")
                if pose_gen:
                    pose_gen.sync_to_current()
                self.node.get_logger().warn(
                    "[Pipeline] E-STOP released — holding at measured joint "
                    "positions in POSE mode.")

            if self._was_numb and not self.safety_processor.is_numb:
                self._was_numb = False
                self.mode = "pose"
                self.mode_transition_active = False
                self._safety_elapsed = None
                pose_gen = self.policy_manager.policies.get("pose")
                if pose_gen:
                    pose_gen.sync_to_current()
                self.node.get_logger().warn(
                    "[Pipeline] Numb released: holding at measured joint positions in POSE mode.")

            self._advance_policy_switch()
            self._release_gait_override()

            if self.mode != "policy" or self.safety_processor.is_estopped:
                self.command_slew.reset()
                self._slew_time = None

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
                # The active policy's checks (policies.json). When
                # one trips, the safety policy takes over with no checks.
                # ─────────────────────────────────────────────────────
                sp = self.safety_processor
                pm = self.policy_manager
                name = self.active_policy
                protections = pm.spec(name)["protections"] if name else PROTECTIONS
                is_safe, _ = sp.evaluate_safety(state, protections)

                dt = self.policy_dt
                if self._slew_time is not None and 0.0 < sim_time - self._slew_time <= 0.1:
                    dt = sim_time - self._slew_time
                self._slew_time = sim_time
                cmd_vel = self.command_slew(cmd_vel, dt)

                if not sp.is_policy_blocked:
                    self._safety_elapsed = None
                takeover = not is_safe or sp.is_policy_blocked
                proposed_targets = {}
                if takeover and self.safety_name in pm.policies and not sp.is_numb:
                    if self._safety_elapsed is None:
                        self._safety_elapsed = 0.0
                        self._lying_for = 0.0
                        self.mode_transition_active = False
                        pm.policies[self.safety_name].reset_history()
                        self.node.get_logger().error(
                            f"[Pipeline] Safety policy '{self.safety_name}' takes over from '{name}'.")
                    proposed_targets["safety"] = pm.step_single(
                        self.safety_name, state, cmd_vel, self.mj_to_isaac,
                        current_time=sim_time, dt=self.policy_dt)
                elif not takeover and name in pm.policies:
                    proposed_targets["main"] = pm.step_single(
                        name, state, cmd_vel, self.mj_to_isaac,
                        current_time=sim_time, dt=self.policy_dt)

                final_targets, max_torque = sp.process(
                    proposed_targets=proposed_targets,
                    state=state,
                    protections=protections,
                )
                if sp.is_policy_blocked:
                    if self._safety_elapsed is None:
                        self._safety_elapsed = 0.0
                        self._lying_for = 0.0
                    self._watch_safety_policy(state)
                else:
                    self._safety_elapsed = None

            # ── HARD E-STOP OVERRIDE ─────────────────────────────
            # active_max_torque already reads zero through its property, but
            # the pose branch computes its own local max_torque and hands that
            # to the distributor (and on to /commands/joint_commands), so it
            # has to be cut here too. Targets hold at the last commanded value
            # rather than snapping to the nominal stance: with kp=0 they do
            # nothing now, and they avoid a jump if torque is restored.
            if self.safety_processor.is_estopped:
                self._was_estopped = True
                final_targets = self.latest_targets
                max_torque = 0.0
                self.mode_transition_active = False
            elif self.safety_processor.is_numb:
                self._was_numb = True
                final_targets = self.latest_targets
                max_torque = 0.0
                self.mode_transition_active = False

            # ── MODE TRANSITION INTERPOLATION ────────────────────
            if self.mode_transition_active:
                if self.mode_transition_start_time is None:
                    self.mode_transition_start_time = sim_time
                
                elapsed = sim_time - self.mode_transition_start_time
                alpha = np.clip(elapsed / self.transition_duration, 0.0, 1.0)
                
                final_targets = (1.0 - alpha) * self.mode_transition_start_targets + alpha * final_targets
                
                if alpha >= 1.0:
                    self.mode_transition_active = False
                    self.mode_transition_start_time = None

            self.latest_targets = final_targets
            self.distributor.send(final_targets, max_torque, send_to_robot_cb=send_cb)

        # 3. Telemetry Publishing
        if is_policy_step:
            self.telemetry.publish(sim_time=sim_time, state=state)
            self._publish_robot_state(state, cmd_vel, sim_time)

        return self.latest_targets

