"""The simulators' ROS side of the C++ Controller; driver_node.cpp is the real robot's.

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

import numpy as np
import rclpy
from geometry_msgs.msg import Quaternion, TransformStamped, Twist, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Bool, Float32, Float32MultiArray, String

from quadruped_common import paths, realtime
from quadruped_common.config_loader import load_config, load_policies
from quadruped_common.kinematics import Go2Kinematics, _cross, projected_gravity_b, rot_from_quat

from . import _core as qc
from .export import WTW_GAITS
from .policies import entry_for, load_spec, read_catalog

LOAD_ERROR = "could not load (see the driver's log)"

_RED = "\033[91m"
_YELLOW = "\033[93m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


class HeartbeatStats:
    """Gaps between heartbeat arrivals over the last WINDOW_S. A gap of k periods counts k - 1 missed
    heartbeats; the period is the median gap, so it follows whatever rate the console sends at."""

    WINDOW_S = 10.0

    def __init__(self):
        self.lock = threading.Lock()
        self.first = self.last = None
        self.gaps = deque()  # (arrival, gap s, missed)
        self.missed_total = 0
        self.count = 0

    def add(self, now):
        with self.lock:
            self.count += 1
            if self.last is not None:
                gap = now - self.last
                period = float(np.median([g for _, g, _ in self.gaps])) if len(self.gaps) >= 5 else 0.0
                missed = max(0, round(gap / period) - 1) if period > 0 else 0
                self.gaps.append((now, gap, missed))
                self.missed_total += missed
            else:
                self.first = now
            self.last = now
            self._trim(now)

    def _trim(self, now):
        while self.gaps and self.gaps[0][0] < now - self.WINDOW_S:
            self.gaps.popleft()

    def summary(self, now):
        """max_gap_ms includes the time since the last heartbeat, so a stall shows while it lasts."""
        with self.lock:
            if self.last is None:
                return None
            self._trim(now)
            gaps = [g for _, g, _ in self.gaps] + [now - self.last]
            span = min(self.WINDOW_S, now - self.first)
            return {"hz": round(len(self.gaps) / span, 1) if span > 0 else 0.0,
                    "missed_10s": sum(m for _, _, m in self.gaps),
                    "missed": self.missed_total,
                    "count": self.count,
                    "max_gap_ms": round(max(gaps) * 1000)}


def _estop_banner():
    print("\r" + " " * 120 + "\r", end="", flush=True)
    print(f"\n{_BOLD}{_RED}╔══════════════════════════════════════════════════════╗")
    print("║  ⛔  E-STOP  —  torque cut to 0 Nm                   ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  ➜  Latched. No policy, no pose, no recovery run.    ║")
    print("║  ➜  Press [ENTER] HERE, on the robot, to release.    ║")
    print("║     The console cannot clear this.                   ║")
    print(f"╚══════════════════════════════════════════════════════╝{_RESET}\n")


def _shutdown_banner(reason):
    print("\r" + " " * 120 + "\r", end="", flush=True)
    print(f"\n{_BOLD}{_RED}╔══════════════════════════════════════════════════════╗")
    print("║  ⚠  COMMAND SAFETY PROCESSOR — EMERGENCY SHUTDOWN ⚠  ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Reason: {reason[:43]:<43} ║")
    remaining = reason[43:]
    while remaining:
        print(f"║    {remaining[:49]:<49} ║")
        remaining = remaining[49:]
    print("║                                                      ║")
    print("║  ➜  Main policy BLOCKED.                             ║")
    print("║  ➜  Press [ENTER] here to RESET and re-enable.       ║")
    print(f"╚══════════════════════════════════════════════════════╝{_RESET}\n")
# ---------------------------------------------------------------------------
# Standard State
# ---------------------------------------------------------------------------

class _Imu:
    __slots__ = ("quaternion", "gyroscope", "accelerometer")

    def __init__(self):
        self.quaternion = [1.0, 0.0, 0.0, 0.0]
        self.gyroscope = [0.0, 0.0, 0.0]
        self.accelerometer = [0.0, 0.0, 9.81]   # body-frame specific force


class _MotorState:
    __slots__ = ("q", "dq")

    def __init__(self):
        self.q = 0.0
        self.dq = 0.0


class StandardState:
    """The robot state Telemetry publishes, filled from the controller's."""
    def __init__(self):
        self.imu = _Imu()
        self.base_lin_vel  = [0.0, 0.0, 0.0]
        self.base_lin_vel_sim = [0.0, 0.0, 0.0]
        self.base_lin_vel_est = [0.0, 0.0, 0.0]
        self.base_pos      = [0.0, 0.0, 0.5]
        self.feet_contact  = [0.0, 0.0, 0.0, 0.0]  # FL, FR, RL, RR binary
        self.motorState    = [_MotorState() for _ in range(12)]


class TelemetryManager:
    """Sensors, estimated state and odometry to ROS (what driver_node.cpp publishes on the robot).
    The estimator is the C++ controller's; Pipeline sets it."""

    def __init__(self, node, joint_names=None, use_estimator=None):
        self.node = node

        # Load centralized configuration
        self.config = load_config()
        est_cfg = self.config.get("state_estimator", {})

        # Priority: Constructor arg > ENV variable > YAML config > Default (False)
        if use_estimator is not None:
            self.use_estimator = use_estimator
        else:
            env_val = os.environ.get("USE_ESTIMATOR")
            if env_val is not None:
                self.use_estimator = (env_val == "1")
            else:
                self.use_estimator = est_cfg.get("use_estimator", False)

        self.joint_names = joint_names or [
            "FL_hip_joint",  "FR_hip_joint",  "RL_hip_joint",  "RR_hip_joint",
            "FL_thigh_joint","FR_thigh_joint","RL_thigh_joint","RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
        ]

        self.estimator = None
        self._kin = Go2Kinematics()

        # ── ROS 2 Publishers — raw sensors ────────────────────────────────
        self.joint_pub = self.node.create_publisher(JointState, '/sensors/joint_states', 10)
        self.imu_pub   = self.node.create_publisher(Imu,        '/sensors/imu',          10)
        self.odom_sim_pub = self.node.create_publisher(Odometry,   '/odom/state_simulator', 10)
        self.odom_est_pub = self.node.create_publisher(Odometry,   '/odom/state_estimator', 10)

        # ── ROS 2 Publishers — derived / estimated state ──────────────────
        # Computed once here so all downstream nodes (Supervisor, Controller,
        # Digital Twin) get a consistent, geometry-free view of the robot state.
        self.proj_gravity_pub  = self.node.create_publisher(Vector3,          '/estimator/projected_gravity', 10)
        self.base_lin_vel_pub  = self.node.create_publisher(Vector3,          '/estimator/base_lin_vel',      10)
        self.base_ang_vel_pub  = self.node.create_publisher(Vector3,          '/estimator/base_ang_vel',      10)
        self.base_height_pub   = self.node.create_publisher(Float32,          '/estimator/base_height',       10)
        self.feet_contact_pub  = self.node.create_publisher(Float32MultiArray,'/estimator/feet_contact',      10)

        # Debug: per-leg odometry velocity norms [|v_FL|, |v_FR|, |v_RL|, |v_RR|]
        self.leg_odom_pub = self.node.create_publisher(
            Float32MultiArray, '/estimator/leg_odometry', 10
        )

        odom_cfg = self.config.get("odometry") or {}
        self.odom_source = str(odom_cfg.get("source", "auto"))
        self.odom_height = float(self.config.get("control", {}).get("default_height", 0.33))
        self._has_pos = False
        self._odom_pos = None
        self._odom_time = None
        self._odom_start = None
        self.tf_pub = None
        self.odom_pub = self.node.create_publisher(Odometry, '/odom', 10)
        if odom_cfg.get("publish_tf", True):
            try:
                from tf2_ros import TransformBroadcaster
                self.tf_pub = TransformBroadcaster(self.node)
            except ImportError:
                self.node.get_logger().warn("[Telemetry] tf2_ros missing: no odom -> base transform.")

    # ------------------------------------------------------------------
    def publish(self, sim_time, state: StandardState):
        """Publishes raw sensor data and all derived/estimated state to ROS 2 topics."""
        msg_time = rclpy.time.Time(seconds=sim_time).to_msg()

        # ── 1. Raw sensors ────────────────────────────────────────────────

        # 1a. Joint States
        js          = JointState()
        js.header.stamp = msg_time
        js.name     = self.joint_names
        js.position = [float(m.q)  for m in state.motorState]
        js.velocity = [float(m.dq) for m in state.motorState]
        self.joint_pub.publish(js)

        # 1b. IMU
        imu = Imu()
        imu.header.stamp    = msg_time
        imu.header.frame_id = 'imu_link'
        q = state.imu.quaternion
        imu.orientation          = Quaternion(w=float(q[0]), x=float(q[1]),
                                              y=float(q[2]), z=float(q[3]))
        gv = state.imu.gyroscope
        imu.angular_velocity     = Vector3(x=float(gv[0]), y=float(gv[1]), z=float(gv[2]))
        if hasattr(state.imu, 'accelerometer'):
            ac = state.imu.accelerometer
            imu.linear_acceleration = Vector3(x=float(ac[0]), y=float(ac[1]), z=float(ac[2]))
        self.imu_pub.publish(imu)

        lv = state.base_lin_vel

        # 1c. Odometry (simulator)
        odom_sim = Odometry()
        odom_sim.header.stamp       = msg_time
        odom_sim.header.frame_id    = 'odom'
        odom_sim.child_frame_id     = 'base'
        if state.base_pos:
            odom_sim.pose.pose.position.x = float(state.base_pos[0])
            odom_sim.pose.pose.position.y = float(state.base_pos[1])
            odom_sim.pose.pose.position.z = float(state.base_pos[2])
        if state.imu.quaternion:
            odom_sim.pose.pose.orientation = Quaternion(
                w=float(state.imu.quaternion[0]), x=float(state.imu.quaternion[1]),
                y=float(state.imu.quaternion[2]), z=float(state.imu.quaternion[3]))
        lv_sim = getattr(state, 'base_lin_vel_sim', lv)
        odom_sim.twist.twist.linear = Vector3(x=float(lv_sim[0]), y=float(lv_sim[1]), z=float(lv_sim[2]))
        self.odom_sim_pub.publish(odom_sim)

        # 1d. Odometry (estimator)
        odom_est = Odometry()
        odom_est.header.stamp       = msg_time
        odom_est.header.frame_id    = 'odom'
        odom_est.child_frame_id     = 'base'
        if state.base_pos:
            odom_est.pose.pose.position.x = float(state.base_pos[0])
            odom_est.pose.pose.position.y = float(state.base_pos[1])
            odom_est.pose.pose.position.z = float(state.base_pos[2])
        if state.imu.quaternion:
            odom_est.pose.pose.orientation = Quaternion(
                w=float(state.imu.quaternion[0]), x=float(state.imu.quaternion[1]),
                y=float(state.imu.quaternion[2]), z=float(state.imu.quaternion[3]))
        lv_est = getattr(state, 'base_lin_vel_est', lv)
        odom_est.twist.twist.linear = Vector3(x=float(lv_est[0]), y=float(lv_est[1]), z=float(lv_est[2]))
        self.odom_est_pub.publish(odom_est)

        # ── 2. Derived / Estimated state (/estimator/*) ───────────────────

        # 2a. Projected gravity (gravity vector in body frame)
        pg = projected_gravity_b(state.imu.quaternion)
        self.proj_gravity_pub.publish(
            Vector3(x=float(pg[0]), y=float(pg[1]), z=float(pg[2]))
        )

        # 2b. Linear base velocity in body frame (estimated or GT)
        self.base_lin_vel_pub.publish(
            Vector3(x=float(lv[0]), y=float(lv[1]), z=float(lv[2]))
        )

        # 2c. Angular base velocity (from IMU gyroscope)
        self.base_ang_vel_pub.publish(
            Vector3(x=float(gv[0]), y=float(gv[1]), z=float(gv[2]))
        )

        # 2d. Base height (z from odom position, or default if not available)
        height_msg      = Float32()
        height_msg.data = float(state.base_pos[2]) if state.base_pos else 0.35
        self.base_height_pub.publish(height_msg)

        # 2e. Feet contact flags [FL, FR, RL, RR] — binary floats
        fc_msg      = Float32MultiArray()
        fc_msg.data = [float(c) for c in state.feet_contact]
        self.feet_contact_pub.publish(fc_msg)

        # 2f. Leg odometry debug — velocity norm per leg from FK
        _kin = self._kin
        omega = np.asarray(state.imu.gyroscope, dtype=np.float64)
        norms = []
        for leg_idx in range(4):
            idx     = [leg_idx, leg_idx + 4, leg_idx + 8]
            q_leg   = np.array([state.motorState[i].q  for i in idx])
            dq_leg  = np.array([state.motorState[i].dq for i in idx])
            r_foot  = _kin.foot_position_body(leg_idx, q_leg)
            J       = _kin.foot_jacobian_body(leg_idx, q_leg)
            v_leg   = -J @ dq_leg - _cross(omega, r_foot)
            norms.append(float(np.linalg.norm(v_leg)))
        lo_msg      = Float32MultiArray()
        lo_msg.data = norms
        self.leg_odom_pub.publish(lo_msg)

        self._publish_odom(msg_time, sim_time, state)

    def _odom_pose(self, sim_time, state):
        """Leg odometry (the estimated body velocity turned by the IMU orientation, integrated), or the
        simulator's pose for source ground_truth, or for auto when the estimator is off."""
        quat = state.imu.quaternion
        ground_truth = self.odom_source == "ground_truth" or (self.odom_source == "auto" and not self.use_estimator)
        if self._has_pos and ground_truth:
            # Relative to where the robot started, as on the real robot: the world pose would give a
            # randomly spawned robot's position away to the localizer through the odom frame.
            w, x, y, z = quat
            yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            if self._odom_start is None:
                self._odom_start = (np.asarray(state.base_pos, dtype=np.float64)[:2].copy(), yaw)
            p0, yaw0 = self._odom_start
            c, s = np.cos(yaw0), np.sin(yaw0)
            d = np.asarray(state.base_pos, dtype=np.float64)[:2] - p0
            pos = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], state.base_pos[2]])
            hw, hz = np.cos(yaw0 / 2), -np.sin(yaw0 / 2)
            return pos, (hw * w - hz * z, hw * x - hz * y, hw * y + hz * x, hw * z + hz * w)
        if self._odom_pos is None:
            self._odom_pos = np.array([0.0, 0.0, self.odom_height])
        if self._odom_time is not None and 0.0 < sim_time - self._odom_time < 0.5:
            v_world = rot_from_quat(quat) @ np.asarray(state.base_lin_vel_est, dtype=np.float64)
            self._odom_pos[:2] += v_world[:2] * (sim_time - self._odom_time)
        self._odom_time = sim_time
        return self._odom_pos, quat

    def _publish_odom(self, msg_time, sim_time, state):
        pos, (w, x, y, z) = self._odom_pose(sim_time, state)
        base = TransformStamped()
        base.header.stamp = msg_time
        base.header.frame_id, base.child_frame_id = "odom", "base"
        base.transform.translation.x, base.transform.translation.y, base.transform.translation.z = (
            float(pos[0]), float(pos[1]), float(pos[2]))
        base.transform.rotation = Quaternion(w=float(w), x=float(x), y=float(y), z=float(z))
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        footprint = TransformStamped()
        footprint.header.stamp = msg_time
        footprint.header.frame_id, footprint.child_frame_id = "odom", "base_footprint"
        footprint.transform.translation.x, footprint.transform.translation.y = float(pos[0]), float(pos[1])
        footprint.transform.rotation = Quaternion(w=float(np.cos(yaw / 2)), z=float(np.sin(yaw / 2)))
        if self.tf_pub is not None:
            self.tf_pub.sendTransform([base, footprint])

        odom = Odometry()
        odom.header.stamp = msg_time
        odom.header.frame_id, odom.child_frame_id = "odom", "base_footprint"
        odom.pose.pose.position.x, odom.pose.pose.position.y = float(pos[0]), float(pos[1])
        odom.pose.pose.orientation = footprint.transform.rotation
        v, wz = state.base_lin_vel, state.imu.gyroscope[2]
        odom.twist.twist.linear = Vector3(x=float(v[0]), y=float(v[1]), z=0.0)
        odom.twist.twist.angular = Vector3(x=0.0, y=0.0, z=float(wz))
        self.odom_pub.publish(odom)


class Distributor:
    """Joint targets to /commands/joint_commands, and to the simulator through its callback."""

    def __init__(self, node, joint_names=None):
        self.node = node
        self.joint_names = joint_names or [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"
        ]

        self.cmd_pub = self.node.create_publisher(JointState, '/commands/joint_commands', 10)
        self.node.get_logger().info("[Distributor] Initialized.")

    def send(self, targets: np.ndarray, max_torque: float, send_to_robot_cb=None):
        """
        Publish final targets + torque to the ROS 2 topic, and optionally
        invoke a driver-specific callback (SDK write, sim actuator set, etc.).

        Args:
            targets:          12-dim array of joint position targets.
            max_torque:       Maximum torque (Nm) to apply.
            send_to_robot_cb: Optional callable(targets) for driver-specific output.
        """
        if send_to_robot_cb:
            send_to_robot_cb(targets)

        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = targets.tolist()
        msg.effort = [float(max_torque)] * len(self.joint_names)
        self.cmd_pub.publish(msg)


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


class Pipeline:
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
            log.warn("[Pipeline] No main policy checkpoint provided. Policy runner disabled.")
        safety_name = (policies_cfg.get("safety_policy") or "") if safety_switch else ""
        if safety_name and safety_name not in self.catalog:
            log.error(f"[Pipeline] safety_policy '{safety_name}' is not in policies.json.")
            safety_name = ""
        elif safety_name and not self._load_now(safety_name):
            safety_name = ""
        self.core.set_safety_policy(safety_name)
        if safety_switch and not safety_name:
            log.warn("[Pipeline] No safety policy: the robot goes numb as soon as a check trips.")
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
        log.info("[Pipeline] C++ controller (quadruped_control).")

    # ---- policies ----

    @staticmethod
    def _resolve(path):
        return path if not path or os.path.isabs(path) else paths.repo(path)

    def _load_main(self, checkpoint):
        log = self.node.get_logger()
        checkpoint = self._resolve(checkpoint)
        if not os.path.exists(checkpoint):
            log.warn(f"[Policies] Policy file not found: {checkpoint}")
            return
        real = os.path.realpath(checkpoint)
        alias = next((n for n, s in self.catalog.items() if s["path"] and os.path.realpath(self._resolve(s["path"])) == real), None)
        spec = self.catalog[alias] if alias else {"path": checkpoint, "protections": ("heartbeat", "tilt", "pitch", "rom"),
                                                  "commands": True}
        try:
            log.info(f"[Policies] Loading policy 'main' from: {checkpoint}")
            self.core.add_policy("main", load_spec(checkpoint, "main"), entry_for(spec, checkpoint))
            self.core.set_main_alias(alias)
        except Exception as e:
            log.error(f"[Policies] Failed to load policy 'main': {e}")

    def _load(self, name):
        """(spec, entry) of a catalog policy, or None after logging why not."""
        log = self.node.get_logger()
        path = self._resolve(self.catalog[name]["path"])
        if not path or not os.path.exists(path):
            log.warn(f"[Policies] Policy file not found: {path}")
            return None
        try:
            log.info(f"[Policies] Loading policy '{name}' from: {path}")
            return load_spec(path, name), entry_for(self.catalog[name], path)
        except Exception as e:
            log.error(f"[Policies] Failed to load policy '{name}': {e}")
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
        _estop_banner()
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
                "[Safety] No interactive stdin; clear this stop from the console with 'safety = reset'.")
            return
        print(f"{_YELLOW}{_BOLD}>>> Safety RESET. Re-enabling main policy...{_RESET}")
        self.node.get_logger().info("[Safety] Operator pressed ENTER. Main policy re-enabled.")
        self._call(self.core.safety.release)

    def _flush(self):
        """Log the controller's events; a safety trip also gets the banner and the ENTER thread."""
        with self._lock:
            events = self.core.drain_events()
        log = self.node.get_logger()
        for e in events:
            if e.kind == qc.EventKind.SAFETY_TRIP:
                _shutdown_banner(e.text)
                threading.Thread(target=self._trip_reset_worker, daemon=True).start()
            elif e.level == qc.LogLevel.ERROR:
                log.error(e.text)
            elif e.level == qc.LogLevel.WARN:
                log.warn(e.text)
            else:
                log.info(e.text)

    # ---- What the drivers call ----

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
        """One control step. raw_state_kwargs: q, dq, quat, gyro and optionally accel, pos, vel, contact; cmd_vel
        overrides /cmd_vel when given; send_cb(targets) sends to the robot. Returns the latest targets."""
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

