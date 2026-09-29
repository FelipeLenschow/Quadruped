"""
Gazebo Driver for Quadruped Locomotion.

Runs the Gazebo server in-process and steps it in lockstep with the controller, the way the
MuJoCo driver steps MuJoCo: the PD runs on every 1 ms physics step and the pipeline every 5 ms,
so a slow Python loop only slows the simulation down instead of delaying the torques. The GUI,
when wanted, is a separate `gz sim -g` client on the same partition.
"""

import os
import sys
import time
import signal
import argparse
import threading
import subprocess

import numpy as np

from quadruped_core.pipeline import LocomotionPipeline
from quadruped_core.telemetry.estimator import rot_from_quat
from quadruped_core.telemetry.kinematics import Go2Kinematics
from quadruped_core.config_loader import load_config
from quadruped_core.controller.robot_defaults import DEFAULT_STANCE_QPOS
from quadruped_core import paths

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.utilities import remove_ros_args
from rosgraph_msgs.msg import Clock
from rclpy.time import Time
from std_msgs.msg import Bool, Float32

try:
    import gz.math7  # noqa: F401  (registers the Pose3d/Vector3d types the sim bindings return)
    from gz.sim8 import TestFixture, World, Model, Link, Joint, world_entity
except ImportError:
    print("[ERROR] Gazebo (Harmonic) Python bindings not found (python3-gz-sim8).")
    sys.exit(1)


# Isaac order (grouped by joint type): FL, FR, RL, RR
JOINT_NAMES = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]
PHYSICS_DT = 0.001
CONTROL_STEPS = 5        # physics steps per pipeline step (200 Hz)
POLICY_DT = 0.02


class Ros2GazeboDriver(Node):
    def __init__(self, robot_type, world_name="scene", checkpoint=None, obs_dim=49,
                 use_estimator=False, headless=False):
        super().__init__("gazebo_bridge_node")
        self.robot_type = robot_type
        self.world_name = world_name
        self.headless = headless

        self.config = load_config()
        self.ctrl_cfg = self.config.get("control", {})
        self.motor_cfg = self.config.get("motor", {})
        est_cfg = self.config.get("state_estimator", {})

        self.kp = float(self.ctrl_cfg.get("kp", 0.0))
        self.kd = float(self.ctrl_cfg.get("kd", 0.0))
        self.cmd_vel = [0.0, 0.0, 0.0, 0.0]

        self.pipeline = LocomotionPipeline(
            node=self,
            robot_type=robot_type,
            checkpoint=checkpoint,
            obs_dim=obs_dim,
            use_estimator=use_estimator or est_cfg.get("use_estimator", False),
            joint_names=JOINT_NAMES,
            sim_dt=PHYSICS_DT * CONTROL_STEPS,
        )
        self.pipeline.decimation = int(round(POLICY_DT / (PHYSICS_DT * CONTROL_STEPS)))
        self.pipeline.policy_dt = POLICY_DT

        self.create_subscription(Float32, "/control/kp", self._kp_cb, 10)
        self.create_subscription(Float32, "/control/kd", self._kd_cb, 10)
        self.create_subscription(Bool, "/base/freeze", self._freeze_base_cb, 10)
        self.clock_pub = self.create_publisher(Clock, "/clock", 10)

        self.kinematics = Go2Kinematics()
        self.sim_time = 0.0
        self.q = DEFAULT_STANCE_QPOS.astype(np.float64).copy()
        self.dq = np.zeros(12)
        self._steps_left = 0
        self.base_pos = np.zeros(3)
        self.base_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.base_ang_vel = np.zeros(3)
        self.base_lin_vel_b = np.zeros(3)
        self.base_accel = np.array([0.0, 0.0, 9.81])
        self.targets = DEFAULT_STANCE_QPOS.astype(np.float64).copy()
        self.effort_limit = 0.0
        self.torques = np.zeros(12)

        self._joints = None
        self._base = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._done = threading.Event()
        self.gui_proc = None

        with open(paths.description("gazebo", f"{world_name}.sdf"), "r") as f:
            scene_xml = f.read()
        scene_xml = scene_xml.replace("go2_description", f"{self.robot_type}_description")
        scene_xml = scene_xml.replace("<name>go2</name>", f"<name>{self.robot_type}</name>")
        # Unthrottled: the driver paces sim time to the wall clock itself, and Gazebo's own
        # throttle stacked on top of the controller's work held it near 0.8x real time.
        scene_xml = scene_xml.replace("<real_time_factor>1.0</real_time_factor>", "<real_time_factor>0</real_time_factor>")
        self.world_path = f"/tmp/gazebo_scene_{self.robot_type}_{os.getpid()}.sdf"
        with open(self.world_path, "w") as f:
            f.write(scene_xml)

        self.physics_thread = threading.Thread(target=self._physics_loop, daemon=True)
        self.physics_thread.start()
        # The in-process Gazebo server installs its own SIGINT handler, which would swallow
        # Ctrl+C; put Python's back once it exists so rclpy.spin still stops.
        self._ready.wait(timeout=60.0)
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.signal(signal.SIGTERM, signal.default_int_handler)

    # --- ROS callbacks ---
    def _kp_cb(self, msg):
        if float(msg.data) != self.kp:
            self.kp = float(msg.data)
            self.get_logger().info(f"[GazeboDriver] Dynamic Kp updated to: {self.kp:.1f}")

    def _kd_cb(self, msg):
        if float(msg.data) != self.kd:
            self.kd = float(msg.data)
            self.get_logger().info(f"[GazeboDriver] Dynamic Kd updated to: {self.kd:.2f}")

    def _freeze_base_cb(self, msg: Bool):
        state = "on" if msg.data else "off"
        print(f"[GazeboDriver] freeze_base ({state}): Not supported on this simulator.")

    # --- Gazebo system callbacks (run inside server.run, on every physics step) ---
    def _setup(self, ecm):
        model = Model(World(world_entity(ecm)).model_by_name(ecm, self.robot_type))
        self._joints = [Joint(model.joint_by_name(ecm, name)) for name in JOINT_NAMES]
        # Gazebo 8 ignores <initial_position>, so the stance is set here, as MuJoCo's reset does.
        for joint, q0 in zip(self._joints, DEFAULT_STANCE_QPOS):
            joint.enable_position_check(ecm, True)
            joint.enable_velocity_check(ecm, True)
            joint.reset_position(ecm, [float(q0)])
        self._base = Link(model.link_by_name(ecm, "base"))
        self._base.enable_velocity_checks(ecm, True)
        self._base.enable_acceleration_checks(ecm, True)

    def _read_joints(self, ecm):
        for i, joint in enumerate(self._joints):
            p = joint.position(ecm)
            v = joint.velocity(ecm)
            if p:
                self.q[i] = p[0]
            if v:
                self.dq[i] = v[0]

    def _pre_update(self, info, ecm):
        if self._joints is None:
            self._setup(ecm)
            return
        if info.paused:
            return
        self._read_joints(ecm)
        self.torques = self._pd_torques()
        for joint, tau in zip(self._joints, self.torques):
            joint.set_force(ecm, [float(tau)])

    def _post_update(self, info, ecm):
        self.sim_time = info.sim_time.total_seconds()
        self._steps_left -= 1
        if self._base is None or self._steps_left > 0:
            return
        self._read_joints(ecm)
        pose = self._base.world_pose(ecm)
        if pose is None:
            return
        p, r = pose.pos(), pose.rot()
        self.base_pos = np.array([p.x(), p.y(), p.z()])
        self.base_quat = np.array([r.w(), r.x(), r.y(), r.z()])
        R = rot_from_quat(self.base_quat)
        v = self._base.world_linear_velocity(ecm)
        w = self._base.world_angular_velocity(ecm)
        a = self._base.world_linear_acceleration(ecm)
        if v is not None:
            self.base_lin_vel_b = R.T @ np.array([v.x(), v.y(), v.z()])
        if w is not None:
            self.base_ang_vel = R.T @ np.array([w.x(), w.y(), w.z()])
        if a is not None:
            self.base_accel = R.T @ (np.array([a.x(), a.y(), a.z()]) + np.array([0.0, 0.0, 9.81]))

    def _pd_torques(self):
        """DC-motor PD matching the MuJoCo driver: torque limited by the safety processor's
        active limit and the torque-speed curve."""
        effort_limit = self.effort_limit
        if effort_limit <= 0.1:
            return np.zeros(12)
        sat_effort = self.motor_cfg.get("max_torque", 45.0)
        vel_lim = self.motor_cfg.get("max_velocity", 30.0)
        raw = self.kp * (self.targets - self.q) - self.kd * self.dq
        v_max = vel_lim * (1 + effort_limit / sat_effort)
        v = np.clip(self.dq, -v_max, v_max)
        t_top = effort_limit * (1.0 - v / vel_lim)
        t_bot = effort_limit * (-1.0 - v / vel_lim)
        return np.clip(raw, np.maximum(t_bot, -effort_limit), np.minimum(t_top, effort_limit))

    # --- Main loop ---
    def _foot_contacts(self):
        """Low and still, the way a force sensor would see it: height alone also flagged swing feet
        skimming the ground, which pulled the estimator's leg odometry to ~70% of the true speed."""
        contact = [0.0, 0.0, 0.0, 0.0]
        R = rot_from_quat(self.base_quat)
        for leg in range(4):
            idx = [leg, leg + 4, leg + 8]
            r_foot = self.kinematics.foot_position_body(leg, self.q[idx])
            if (self.base_pos + R @ r_foot)[2] >= 0.04:  # foot radius is 2.2 cm
                continue
            v_foot = (self.base_lin_vel_b + np.cross(self.base_ang_vel, r_foot)
                      + self.kinematics.foot_jacobian_body(leg, self.q[idx]) @ self.dq[idx])
            if np.linalg.norm(v_foot) < 0.25:
                contact[leg] = 1.0
        return contact

    def _physics_loop(self):
        try:
            self._run_physics()
        finally:
            self._done.set()

    def _run_physics(self):
        partition = os.environ.get("GZ_PARTITION") or f"quadruped_sim_{os.getpid() % 100000}"
        os.environ["GZ_PARTITION"] = partition
        os.environ["GZ_SIM_RESOURCE_PATH"] = os.pathsep.join(
            [paths.description("gazebo"), paths.description("gazebo", "models")]
            + [paths.description("robots", r, "models") for r in ("Unitree_Go2", "Unitree_Go1", "Unitree_A1")]
            + [os.environ.get("GZ_SIM_RESOURCE_PATH", "")]
        )
        print(f"[GazeboDriver] Partition: {partition}")

        fixture = TestFixture(self.world_path)
        fixture.on_pre_update(self._pre_update)
        fixture.on_post_update(self._post_update)
        fixture.finalize()
        server = fixture.server()
        self._steps_left = 1
        server.run(True, 1, False)  # first step runs _setup
        self._steps_left = 1
        server.run(True, 1, False)  # the stance set by _setup is now in the state
        self._ready.set()

        if not self.headless:
            env = os.environ.copy()
            if os.environ.get("FORCE_SOFTWARE_RENDER", "1") == "1":
                env["LIBGL_ALWAYS_SOFTWARE"] = "1"
                env["QT_X11_NO_MITSHM"] = "1"
            self.gui_proc = subprocess.Popen(["gz", "sim", "-g"], env=env, preexec_fn=os.setsid)

        print(f"[GazeboDriver] Initialized for {self.robot_type.upper()}: physics {1 / PHYSICS_DT:.0f} Hz, "
              f"pipeline {1 / (PHYSICS_DT * CONTROL_STEPS):.0f} Hz, lockstep.")

        wall_ref, sim_ref = time.time(), self.sim_time
        rtf_wall, rtf_sim = wall_ref, self.sim_time
        count = 0
        while not self._stop.is_set():
            raw_data = {
                "q": self.q.copy(),
                "dq": self.dq.copy(),
                "quat": self.base_quat,
                "gyro": self.base_ang_vel,
                "vel": self.base_lin_vel_b,
                "pos": self.base_pos,
                "accel": self.base_accel,
                "contact": self._foot_contacts(),
            }
            self.clock_pub.publish(Clock(clock=Time(seconds=self.sim_time).to_msg()))
            self.cmd_vel = self.pipeline.cmd_vel
            self.targets = np.asarray(self.pipeline.step(
                raw_state_kwargs=raw_data, cmd_vel=self.cmd_vel, sim_time=self.sim_time), dtype=np.float64)
            self.effort_limit = self.pipeline.safety_processor.active_max_torque

            self._steps_left = CONTROL_STEPS
            server.run(True, CONTROL_STEPS, False)

            ahead = (self.sim_time - sim_ref) - (time.time() - wall_ref)
            if ahead > 0:
                time.sleep(ahead)
            elif ahead < -0.1:
                wall_ref, sim_ref = time.time(), self.sim_time

            count += 1
            if count % 40 == 0:
                now = time.time()
                rtf = (self.sim_time - rtf_sim) / max(now - rtf_wall, 1e-6)
                rtf_wall, rtf_sim = now, self.sim_time
                runner = self.pipeline.policy_manager.policies.get("main")
                inf_ms = runner.inf_times[-1] * 1000 if runner and getattr(runner, "inf_times", None) else 0.0
                print(
                    f"\r[Bridge] t={self.sim_time:7.2f} rtf={rtf:4.2f}x h={self.base_pos[2]:.2f} "
                    f"vx={self.base_lin_vel_b[0]:+5.2f} vy={self.base_lin_vel_b[1]:+5.2f} "
                    f"wz={self.base_ang_vel[2]:+5.2f} tq={np.linalg.norm(self.torques):.1f} "
                    f"{self.pipeline.mode} cmd={self.cmd_vel[0]:+.2f} | inf={inf_ms:4.1f}ms   ",
                    end="", flush=True,
                )

    def _cleanup(self):
        # The server must be torn down in its own thread before the interpreter exits, or its
        # threads abort the process ("terminate called without an active exception").
        self._stop.set()
        self._done.wait(timeout=15.0)
        self.physics_thread.join(timeout=5.0)
        if self.gui_proc:
            try:
                os.killpg(os.getpgid(self.gui_proc.pid), 15)
                self.gui_proc.wait(timeout=5)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="go2")
    parser.add_argument("--world", type=str, default="scene", help="world file in quadruped_description/gazebo: scene, nav")
    parser.add_argument("--internal_policy", type=str, default=None, help="Path to policy checkpoint")
    parser.add_argument("--obs_dim", type=int, default=49)
    parser.add_argument("--use_estimator", action="store_true", default=False,
                        help="Replace perfect odometry with contact-aided IMU velocity estimator")
    parser.add_argument("--headless", action="store_true", help="No Gazebo GUI")
    args = parser.parse_args(remove_ros_args()[1:])
    rclpy.init()
    node = Ros2GazeboDriver(
        args.robot, args.world, checkpoint=args.internal_policy, obs_dim=args.obs_dim,
        use_estimator=args.use_estimator,
        headless=args.headless or os.environ.get("GZ_HEADLESS", "0") == "1",
    )
    try:
        # spin_once with a timeout, not spin(): Python's SIGINT handler (see __init__) only
        # runs when this thread gets back to Python.
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node._cleanup()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
