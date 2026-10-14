"""LocomotionPipeline's interface with the C++ Controller behind it.

The control decision (state estimate, policies, safety, poses, blends) runs in C++; this keeps the ROS side:
subscriptions, publishing (telemetry, joint commands, /robot_state, /pose/status), the banners and the
ENTER-to-release threads, and policy loading in the background, handed to the controller on the next step.
"""
import json
import os
import queue
import threading
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32, Float32MultiArray, String

from quadruped_core import paths, realtime
from quadruped_core.config_loader import load_config, load_policies
from quadruped_core.controller.command_safety_processor import (
    CommandSafetyProcessor, HeartbeatStats, _BOLD, _RESET, _YELLOW)
from quadruped_core.controller.distributor import Distributor
from quadruped_core.controller.policy_runner import WTW_GAITS
from quadruped_core.telemetry.telemetry import StandardState, TelemetryManager

from . import _core as qc
from .policies import entry_for, load_spec, read_catalog

LOAD_ERROR = "could not load (see the driver's log)"
GAIT_NAMES = ["height", "frequency", "phase", "offset", "bound", "swing", "pitch", "width", "duty"]


class _RunnerView:
    """What the drivers read off a loaded policy runner: the latest step times."""

    def __init__(self):
        self.inf_times = deque(maxlen=1000)
        self.counter = 0


class _PolicyManagerView:
    def __init__(self, pipeline):
        self._pipeline = pipeline
        self._views = {}

    @property
    def policies(self):
        names = list(self._pipeline.core.loaded) + ["pose"]
        return {n: self._views.setdefault(n, _RunnerView()) for n in names}

    @property
    def available(self):
        return list(self._pipeline.core.available)


class _SafetyView:
    """CommandSafetyProcessor's attributes, read from the C++ one, plus the heartbeat statistics kept here."""

    def __init__(self, core):
        object.__setattr__(self, "_safety", core.safety)
        object.__setattr__(self, "heartbeat_stats", HeartbeatStats())

    def __getattr__(self, name):
        return getattr(self._safety, name)

    def __setattr__(self, name, value):
        setattr(self._safety, name, value)


class _SlewView:
    def __init__(self, core):
        self._core = core

    @property
    def enabled(self):
        return self._core.slew_enabled

    @enabled.setter
    def enabled(self, on):
        self._core.slew_enabled = bool(on)


class CppPipeline:
    def __init__(self, node, robot_type="go2", checkpoint=None, obs_dim=49, use_estimator=False, joint_names=None,
                 sim_dt=0.001, safety_switch=True):
        self.node = node
        self.robot_type = robot_type
        self.sim_dt = sim_dt
        self.config = load_config()
        self._lock = threading.RLock()
        log = node.get_logger()

        cfg = qc.ControllerConfig.from_yaml(paths.config("config.yaml"), paths.description("config", "go2_kinematics.yaml"),
                                            sim_dt, bool(use_estimator), bool(safety_switch))
        if joint_names:
            cfg.joint_names = list(joint_names)
        self.core = qc.Controller(cfg)
        self.safety_processor = _SafetyView(self.core)
        self.policy_manager = _PolicyManagerView(self)
        self.command_slew = _SlewView(self.core)

        self.telemetry = TelemetryManager(node, joint_names, use_estimator=use_estimator)
        self.telemetry.estimator = self.core.estimator
        self.distributor = Distributor(node, joint_names=joint_names)

        # Policies: the catalog, "main" (the launcher's checkpoint) and the safety policy, loaded now.
        policies_cfg = load_policies()
        self.catalog = read_catalog(policies_cfg.get("policies") or {}, warn=log.error)
        self.core.set_catalog([(n, entry_for(s, self._resolve(s["path"]))) for n, s in self.catalog.items()])
        self._loading = set()
        self._loaded = queue.SimpleQueue()
        if checkpoint:
            self._load_main(checkpoint)
        else:
            log.warn("[LocomotionPipeline] No main policy checkpoint provided. Policy runner disabled.")
        safety_name = (policies_cfg.get("safety_policy") or "") if safety_switch else ""
        if safety_name and safety_name not in self.catalog:
            log.error(f"[LocomotionPipeline] safety_policy '{safety_name}' is not in policies.json.")
            safety_name = ""
        elif safety_name and not self._load_now(safety_name):
            safety_name = ""
        self.core.set_safety_policy(safety_name)
        if safety_switch and not safety_name:
            log.warn("[LocomotionPipeline] No safety policy: the robot goes numb as soon as a check trips.")
        self.core.set_active_policy("main" if self.core.has_policy("main") else None)

        # ROS side.
        sub = node.create_subscription
        sub(String, "/pipeline/policy", self._policy_cb, 10)
        sub(String, "/pipeline/mode", lambda m: self._call(self.core.set_mode, m.data), 10)
        sub(Float32MultiArray, "/gait_command", lambda m: self._call(self.core.gait, np.asarray(m.data, np.float32)), 10)
        sub(Float32MultiArray, "/gait_command/override",
            lambda m: self._call(self.core.gait_override, np.asarray(m.data, np.float32), time.monotonic()), 10)
        sub(Twist, "/cmd_vel", lambda m: self._call(self.core.cmd_vel, m.linear.x, m.linear.y, m.angular.z, time.monotonic()), 10)
        sub(Float32MultiArray, "/pipeline/trick", lambda m: self._call(self.core.trick, list(m.data), time.monotonic()), 10)
        sub(String, "/pose/command", lambda m: self._call(self.core.pose_command, m.data.strip().lower()), 10)
        sub(Float32, "/pose/interp_duration", lambda m: self._call(self.core.set_pose_interp_duration, m.data), 10)
        sp = self.core.safety
        sub(Float32, "/safety/heartbeat", self._heartbeat_cb, 10)
        sub(Float32, "/safety/max_torque_percent", lambda m: self._call(sp.set_max_torque_percent, m.data), 10)
        sub(Float32, "/safety/base_tilt_limit_deg", lambda m: self._call(sp.set_base_tilt_limit, m.data), 10)
        sub(Float32, "/safety/base_forward_tilt_limit_deg", lambda m: self._call(sp.set_forward_tilt_limit, m.data), 10)
        sub(Float32, "/safety/joint_rom_safety_margin", lambda m: self._call(sp.set_rom_margin, m.data), 10)
        sub(Float32, "/safety/watchdog_timeout", lambda m: self._call(sp.set_watchdog_timeout, m.data), 10)
        sub(Bool, "/safety/reset", lambda m: m.data and self._call(sp.safety_reset), 10)
        sub(Bool, "/safety/numb", lambda m: self._call(sp.numb, bool(m.data)), 10)
        sub(Bool, "/safety/estop", self._estop_cb, 10)
        self._estop_state_pub = node.create_publisher(Bool, "/safety/estop_state", 10)
        self._pose_status_pub = node.create_publisher(String, "/pose/status", 10)
        self._state_pub = node.create_publisher(String, "/robot_state", 10)
        cmd_cfg = self.config.get("cmd_vel") or {}
        self._state_period = 1.0 / float(cmd_cfg.get("robot_state_rate", 10.0))
        self._state_time = 0.0
        self._estop_reset_running = False
        self._flush()
        log.info("[LocomotionPipeline] C++ controller (quadruped_control).")

    # ---- policies ----

    @staticmethod
    def _resolve(path):
        return path if not path or os.path.isabs(path) else paths.repo(path)

    def _load_main(self, checkpoint):
        log = self.node.get_logger()
        checkpoint = self._resolve(checkpoint)
        if not os.path.exists(checkpoint):
            log.warn(f"[PolicyManager] Policy file not found: {checkpoint}")
            return
        real = os.path.realpath(checkpoint)
        alias = next((n for n, s in self.catalog.items() if s["path"] and os.path.realpath(self._resolve(s["path"])) == real), None)
        spec = self.catalog[alias] if alias else {"path": checkpoint, "protections": ("heartbeat", "tilt", "pitch", "rom"),
                                                  "commands": True}
        try:
            log.info(f"[PolicyManager] Loading policy 'main' from: {checkpoint}")
            self.core.add_policy("main", load_spec(checkpoint, "main"), entry_for(spec, checkpoint))
            self.core.set_main_alias(alias)
        except Exception as e:
            log.error(f"[PolicyManager] Failed to load policy 'main': {e}")

    def _load(self, name):
        """(spec, entry) of a catalog policy, or None after logging why not."""
        log = self.node.get_logger()
        path = self._resolve(self.catalog[name]["path"])
        if not path or not os.path.exists(path):
            log.warn(f"[PolicyManager] Policy file not found: {path}")
            return None
        try:
            log.info(f"[PolicyManager] Loading policy '{name}' from: {path}")
            return load_spec(path, name), entry_for(self.catalog[name], path)
        except Exception as e:
            log.error(f"[PolicyManager] Failed to load policy '{name}': {e}")
            return None

    def _load_now(self, name):
        loaded = self._load(name)
        if loaded:
            self.core.add_policy(name, *loaded)
        return bool(loaded)

    def _load_in_background(self, name):
        # Started from the driver's thread, so it would inherit the policy's core at SCHED_FIFO.
        realtime.leave_realtime()
        try:
            self._loaded.put((name, self._load(name)))
        finally:
            with self._lock:
                self._loading.discard(name)

    def _policy_cb(self, msg):
        name = msg.data.strip()
        with self._lock:
            if self.core.request_policy(name) == qc.PolicyRequest.NEEDS_LOAD and name not in self._loading:
                self._loading.add(name)
                threading.Thread(target=self._load_in_background, args=(name,), daemon=True).start()
        self._flush()

    def _take_loaded(self):
        while True:
            try:
                name, loaded = self._loaded.get_nowait()
            except queue.Empty:
                return
            if loaded:
                self.core.add_policy(name, *loaded)
            else:
                self.core.set_load_error(name, LOAD_ERROR)

    # ---- inputs ----

    def _call(self, fn, *args):
        with self._lock:
            result = fn(*args)
        self._flush()
        return result

    def _heartbeat_cb(self, msg):
        now = time.monotonic()
        self._call(self.core.safety.heartbeat, now)
        self.safety_processor.heartbeat_stats.add(now)

    def _estop_cb(self, msg):
        if not msg.data:
            return
        if not self._call(self.core.safety.estop):
            return
        self._estop_state_pub.publish(Bool(data=True))
        CommandSafetyProcessor._print_estop_banner(self)
        if not self._estop_reset_running:
            self._estop_reset_running = True
            threading.Thread(target=self._estop_reset_worker, daemon=True).start()

    def _estop_reset_worker(self):
        try:
            input()
        except (EOFError, OSError):
            self._estop_reset_running = False
            self.node.get_logger().error(
                "[SAFETY] E-STOP latched, but this driver has no interactive stdin - nothing can release it. "
                "Restart the driver on the robot, or run it in a terminal.")
            return
        self._estop_reset_running = False
        self._call(self.core.safety.release_estop)
        self._estop_state_pub.publish(Bool(data=False))
        print(f"{_YELLOW}{_BOLD}>>> E-STOP released. Torque ceiling back to "
              f"{self.core.safety.global_max_torque:.1f} Nm.{_RESET}")

    def _trip_reset_worker(self):
        try:
            input()
        except (EOFError, OSError):
            self.node.get_logger().warn(
                "[CommandSafetyProcessor] No interactive stdin; clear this stop from the console with 'safety = reset'.")
            return
        print(f"{_YELLOW}{_BOLD}>>> Safety RESET. Re-enabling main policy...{_RESET}")
        self.node.get_logger().info("[CommandSafetyProcessor] Operator pressed ENTER. Main policy re-enabled.")
        self._call(self.core.safety.release)

    def _flush(self):
        """Log the controller's events; a safety trip also gets the banner and the ENTER thread."""
        with self._lock:
            events = self.core.drain_events()
        log = self.node.get_logger()
        for e in events:
            if e.kind == qc.EventKind.SAFETY_TRIP:
                CommandSafetyProcessor._print_shutdown_banner(SimpleNamespace(node=_QuietNode()), e.text)
                threading.Thread(target=self._trip_reset_worker, daemon=True).start()
            elif e.level == qc.LogLevel.ERROR:
                log.error(e.text)
            elif e.level == qc.LogLevel.WARN:
                log.warn(e.text)
            else:
                log.info(e.text)

    # ---- LocomotionPipeline's interface ----

    @property
    def mode(self):
        return self.core.mode

    @mode.setter
    def mode(self, value):
        with self._lock:
            self.core.force_mode(value)

    @property
    def mode_transition_active(self):
        return self.core.transition_active

    @mode_transition_active.setter
    def mode_transition_active(self, on):
        with self._lock:
            self.core.transition_active = bool(on)

    @property
    def decimation(self):
        return self.core.decimation

    @decimation.setter
    def decimation(self, value):
        self.core.decimation = int(value)

    @property
    def policy_dt(self):
        return self.core.policy_dt

    @policy_dt.setter
    def policy_dt(self, value):
        self.core.policy_dt = float(value)

    @property
    def active_policy(self):
        return self.core.active_policy

    @property
    def latest_targets(self):
        return self.core.latest_targets

    @property
    def cmd_vel(self):
        return [float(v) for v in self.core.current_cmd_vel(time.monotonic())]

    @property
    def trick_command(self):
        return [float(v) for v in self.core.trick_command(time.monotonic())]

    def set_slew_enabled(self, on: bool):
        self.core.slew_enabled = bool(on)

    def reset(self):
        with self._lock:
            self.core.reset()

    def step(self, raw_state_kwargs, cmd_vel, sim_time, send_cb=None):
        """One control step: see LocomotionPipeline.step. Returns the latest targets."""
        r = raw_state_kwargs
        start = time.perf_counter()
        with self._lock:
            self._take_loaded()
            res = self.core.step(r["q"], r["dq"], r["quat"], r["gyro"], r.get("accel"), r.get("pos"), r.get("vel"),
                                 r.get("contact"), cmd_vel=None if cmd_vel is None else np.asarray(cmd_vel, dtype=np.float64),
                                 sim_time=float(sim_time), now=time.monotonic())
        self._flush()
        if not res.policy_step:
            return res.targets
        active = self.core.active_policy
        if active:
            self.policy_manager.policies[active].inf_times.append(time.perf_counter() - start)
        if res.pose_status is not None:
            self._pose_status_pub.publish(String(data=res.pose_status))
        self.distributor.send(res.targets, res.max_torque, send_to_robot_cb=send_cb)
        state = self._standard_state()
        self.telemetry.publish(sim_time=sim_time, state=state)
        self._publish_robot_state(state, sim_time)
        return res.targets

    def _standard_state(self):
        s = self.core.state
        out = StandardState()
        out.imu.quaternion = s.quat_wxyz.tolist()
        out.imu.gyroscope = s.gyro.tolist()
        out.imu.accelerometer = s.accel.tolist()
        for i in range(12):
            out.motorState[i].q = float(s.q[i])
            out.motorState[i].dq = float(s.dq[i])
        out.base_pos = s.base_pos.tolist()
        out.feet_contact = s.feet_contact.tolist()
        out.base_lin_vel = s.base_lin_vel.tolist()
        out.base_lin_vel_sim = s.base_lin_vel_sim.tolist()
        out.base_lin_vel_est = s.base_lin_vel_est.tolist()
        self.telemetry._has_pos = s.has_pos
        return out

    def _publish_robot_state(self, state, sim_time):
        now = time.monotonic()
        if now - self._state_time < self._state_period:
            return
        self._state_time = now
        c, sp = self.core, self.core.safety
        pose = c.pose
        pg = qc.projected_gravity_b(state.imu.quaternion)
        tilt = float(np.degrees(np.arccos(np.clip(-pg[2] / np.linalg.norm(pg), -1.0, 1.0))))
        vel = [float(v) for v in state.base_lin_vel[:2]]
        yaw_rate = float(state.imu.gyroscope[2])
        cmd = [round(float(v), 3) for v in c.last_command[:3]]
        moving = max(abs(v) for v in cmd) > 0.05 or float(np.hypot(*vel)) > 0.15
        if tilt > 60.0:
            posture = "fallen"
        elif sp.active_max_torque <= 0.1 or (c.mode != "policy" and pose.current_pose_name == "lie_flat"):
            posture = "lying"
        elif c.mode == "policy":
            posture = "walking" if moving else "standing"
        else:
            posture = pose.current_pose_name

        gait = None
        g = c.gait_command
        if g is not None:
            g = g.tolist()
            gait = {n: round(float(v), 3) for n, v in zip(GAIT_NAMES, g)}
            gait["name"] = min(WTW_GAITS, key=lambda k: float(np.abs(np.array(WTW_GAITS[k]) - g[2:5]).sum()))
            gait["override"] = c.gait_override_active

        active = c.active_policy
        entry = c.entry(active) if active else None
        self._state_pub.publish(String(data=json.dumps({
            "t": round(float(sim_time), 3),
            "mode": c.mode,
            "transition": c.transition_active,
            "posture": posture,
            "pose": pose.current_pose_name,
            "pose_done": pose.is_complete,
            "moving": moving,
            "estop": sp.is_estopped,
            "safety_blocked": sp.is_policy_blocked,
            "safety_policy_active": c.safety_policy_active,
            "trip_reason": sp.trip_reason,
            "numb": sp.is_numb,
            "numb_reason": sp.numb_reason,
            "policy": active,
            "policies": list(c.available),
            "policy_loading": c.pending_policy,
            "policy_error": c.policy_error,
            "protections": list(entry.protections) if entry else [],
            "torque_on": sp.active_max_torque > 0.1,
            "tilt_deg": round(tilt, 1),
            "velocity": [round(v, 3) for v in vel] + [round(yaw_rate, 3)],
            "cmd": cmd,
            "trick": self.trick_command,
            "trick_switches": entry.switches if entry else 0,
            "gait": gait,
            "heartbeat": self.safety_processor.heartbeat_stats.summary(now),
        })))


class _QuietNode:
    """The banner helpers log as well; the controller already did."""

    def get_logger(self):
        return SimpleNamespace(error=lambda *_: None)
