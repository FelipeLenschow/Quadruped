"""
Gazebo Driver for Quadruped Locomotion.
Manages high-frequency physics stepping, ActuatorNet simulation,
and deterministic policy deployment (Turbo Mode).
"""

import os
import sys

# Ensure absolute path of the repository is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
import torch
import argparse
import signal
import threading
import subprocess
from pathlib import Path
from scipy.spatial.transform import Rotation
from Controller.policy_runner import PolicyRunner
from Controller.policy_bridge import CommandProcessor
from Controller.Utils.telemetry import TelemetryManager

# ROS 2 Standard Imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Vector3, Twist

# Gazebo Transport & Msgs
try:
    from gz.transport13 import Node as GzTransportNode
    from gz.msgs10 import (
        double_pb2,
        model_pb2,
        imu_pb2,
        pose_pb2,
        world_stats_pb2,
        odometry_pb2,
    )
except ImportError:
    print("[ERROR] Gazebo (Harmonic) Python bindings not found.")
    sys.exit(1)


# Isaac order: FL_haa, FR_haa, RL_haa, RR_haa, FL_hfe, FR_hfe, RL_hfe, RR_hfe, FL_kfe, FR_kfe, RL_kfe, RR_kfe
JOINT_NAMES = [
    "lf_haa_joint",
    "rf_haa_joint",
    "lh_haa_joint",
    "rh_haa_joint",
    "lf_hfe_joint",
    "rf_hfe_joint",
    "lh_hfe_joint",
    "rh_hfe_joint",
    "lf_kfe_joint",
    "rf_kfe_joint",
    "lh_kfe_joint",
    "rh_kfe_joint",
]


class Ros2GazeboDriver(Node):
    def __init__(
        self, robot_type, world_name="quadruped_world", checkpoint=None, obs_dim=49
    ):
        super().__init__("gazebo_bridge_node")
        self.robot_type = robot_type
        self.cmd_vel = [0.0, 0.0, 0.0, 0.0]

        # Handles internal policy inference, physics stepping,
        # and standardizes telemetry for ROS 2 monitoring.
        self.runner = None
        if checkpoint:
            print(f"[GazeboDriver] Loading internal policy runner: {checkpoint}")
            self.runner = PolicyRunner(
                checkpoint, obs_dim=obs_dim, robot_type=robot_type
            )
            self.runner.decimation = 4  # 200Hz / 4 = 50Hz
            self.mj_to_isaac = list(range(12))  # Identity
        self.world_name = world_name

        # 4. ROS 2 Telemetry Manager
        self.telemetry = TelemetryManager(self, JOINT_NAMES)
        self.command_processor = CommandProcessor(
            self, robot_type=robot_type, joint_names=JOINT_NAMES
        )

        self.create_subscription(Twist, "/cmd_vel", self._teleop_cb, 10)

        # 2. State & Control Buffers
        self.desired_qpos = np.array(
            [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5],
            dtype=np.float32,
        )
        self.latest_torques = np.zeros(12, dtype=np.float32)
        self.latest_targets = self.desired_qpos.copy()
        self.sim_time = 0.0
        self.q = np.zeros(12)
        self.dq = np.zeros(12)
        self.base_pos = np.zeros(3)
        self.base_quat = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
        self.base_ang_vel = np.zeros(3)
        self.base_lin_vel_b = np.zeros(3)

        # 3. Gazebo Transport (Deferred until physics loop for partitioning)
        self.gz_node = None
        self.joint_pubs = []

        self.new_data_event = threading.Event()
        self._stop = threading.Event()
        self.world_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "scene.sdf")
        )

        # 4. Starting Threads
        self.physics_thread = threading.Thread(target=self._physics_loop, daemon=True)
        self.repeater_thread = threading.Thread(target=self._repeater_loop, daemon=True)
        self.physics_thread.start()
        self.repeater_thread.start()

        print(
            f"[GazeboDriver] Initialized for {robot_type.upper()}. Physics at 500Hz (Slave)."
        )

    def _teleop_cb(self, msg):
        self.cmd_vel = [msg.linear.x, msg.linear.y, msg.angular.z, 0.0]

    # --- Gazebo Callbacks ---
    def _stats_cb(self, msg):
        self.sim_time = msg.sim_time.sec + msg.sim_time.nsec * 1e-9

    def _imu_cb(self, msg):
        # msg.orientation is [x, y, z, w] in protobuf
        q = msg.orientation
        self.base_quat = np.array([q.w, q.x, q.y, q.z])
        self.base_ang_vel = np.array(
            [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        )

    def _joint_cb(self, msg):
        if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
            self.sim_time = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
            
        for joint in msg.joint:
            if joint.name in JOINT_NAMES:
                idx = JOINT_NAMES.index(joint.name)
                try:
                    # Mapping joint state (Gazebo -> Isaac order - NO FLIPS)
                    self.q[idx] = joint.axis1.position
                    self.dq[idx] = joint.axis1.velocity
                    # Diagnostic print (commented out by default)
                    # if idx == 0: print(f"[GazeboBridge] LF_HAA q: {self.q[0]:.3f} dq: {self.dq[0]:.3f}")
                except AttributeError:
                    pass
        # Signal the physics loop to step
        self.new_data_event.set()

    def _odom_cb(self, msg):
        self.base_pos = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        )
        # msg.pose.orientation is [x, y, z, w]
        q = msg.pose.orientation
        q_scipy = [q.x, q.y, q.z, q.w]
        R = Rotation.from_quat(q_scipy).as_matrix()

        # Gazebo Sim2Sim Odometry (gz.msgs.Odometry):
        # In gz.msgs.Odometry, twist is typically in the world/odom frame (unlike ROS nav_msgs).
        # We must rotate it into the local body frame using R.T.
        v_world = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])

        try:
            self.base_lin_vel_b[:] = R.T @ v_world
        except Exception:
            self.base_lin_vel_b[:] = v_world  # Fallback

        # Log Suspected Drift
        if np.abs(self.base_lin_vel_b[0]) > 2.0 and self.sim_time < 0.2:
            print(
                f"\r[Bridge] WARNING: Extreme vx={self.base_lin_vel_b[0]:.2f}. Collision likely."
            )

    def _repeater_loop(self):
        """
        Background thread to repeat the latest torques at 1000Hz.
        Gazebo clears forces every step; this ensures the robot stays powered.
        """
        while not self._stop.is_set():
            if hasattr(self, "joint_pubs") and len(self.joint_pubs) == 12:
                torch_torques = self.latest_torques.copy()
                for i, torque in enumerate(torch_torques):
                    msg = double_pb2.Double()
                    msg.data = float(torque)
                    self.joint_pubs[i].publish(msg)
            time.sleep(0.001)

    def _physics_loop(self):
        # 1. Load ActuatorNet
        act_net_path = Path(__file__).parent.parent / "Mujoco" / "unitree_quadruped.pt"
        if not act_net_path.exists():
            print(f"[Ros2GazeboBridge] ERROR: ActuatorNet missing at {act_net_path}")
            return
        self.act_net = torch.jit.load(str(act_net_path), map_location="cpu").eval()

        # 2. Resource & Global Cleanup
        root_dir = os.path.dirname(os.path.abspath(__file__))
        subprocess.run(
            ["pkill", "-9", "-f", "gz-sim-server"], stderr=subprocess.DEVNULL
        )
        time.sleep(1.0)

        # 3. Environment Setup (Crucial for Resource Path and Partition)
        env = os.environ.copy()

        # Ensure a truly unique partition for THIS specific bridge instance
        unique_id = os.getpid() % 100000
        partition = f"quadruped_sim_{unique_id}"
        env["GZ_PARTITION"] = partition

        # Resource paths for Gazebo to find models and meshes
        model_path = os.path.join(root_dir, "models")
        env["GZ_SIM_RESOURCE_PATH"] = (
            root_dir
            + os.pathsep
            + model_path
            + os.pathsep
            + env.get("GZ_SIM_RESOURCE_PATH", "")
        )

        # VDI Rendering Fixes
        if os.environ.get("FORCE_SOFTWARE_RENDER", "1") == "1":
            env["LIBGL_ALWAYS_SOFTWARE"] = "1"
            env["QT_X11_NO_MITSHM"] = "1"

        gz_args = ["gz", "sim", self.world_path]
        if os.environ.get("GZ_HEADLESS", "0") == "1":
            gz_args.append("-s")

        print(f"[GazeboDriver] Partition: {partition}")
        print(f"[GazeboDriver] Launching: {' '.join(gz_args)}")

        # Launch Gazebo in a new process group with the ENRICHED env
        self.gz_proc = subprocess.Popen(gz_args, env=env, preexec_fn=os.setsid)

        # Match our own process partition for topic discovery
        os.environ["GZ_PARTITION"] = partition
        print(f"[GazeboDriver] Initializing GzTransportNode on partition: {partition}")
        self.gz_node = GzTransportNode()

        # Advertise joint topics
        self.joint_pubs = []
        for jname in JOINT_NAMES:
            topic = f"/model/{self.robot_type}/joint/{jname}/cmd_force"
            self.joint_pubs.append(self.gz_node.advertise(topic, double_pb2.Double))

        time.sleep(5.0)

        # 4. Bind Subscriptions
        self.gz_node.subscribe(
            imu_pb2.IMU,
            f"/model/{self.robot_type}/link/base/sensor/imu/imu",
            self._imu_cb,
        )
        self.gz_node.subscribe(
            model_pb2.Model, f"/model/{self.robot_type}/joint_state", self._joint_cb
        )
        self.gz_node.subscribe(
            odometry_pb2.Odometry, f"/model/{self.robot_type}/odometry", self._odom_cb
        )
        self.gz_node.subscribe(
            world_stats_pb2.WorldStatistics,
            f"/world/{self.world_name}/stats",
            self._stats_cb,
        )

        # 5. Wait for First State
        print("[GazeboDriver] Waiting for simulation to start and first joint state...")
        while (self.sim_time == 0) and not self._stop.is_set():
            time.sleep(0.1)

        print(
            f"[GazeboDriver] Simulation started at t={self.sim_time:.2f}. Initializing targets."
        )
        self.latest_targets[:] = (
            self.q if not np.all(self.q == 0) else self.latest_targets
        )

        print(
            "[GazeboDriver] Simulation started. Waiting 5s for drop/stabilization..."
        )
        time.sleep(5.0)

        print(
            "\n=======================================================\n"
            "[GazeboDriver] Activating Control Loop..."
            "\n=======================================================\n"
        )
        actuator_count = 0
        count = 0
        pos_err_hist = np.zeros((6, 12), dtype=np.float32)
        vel_hist = np.zeros((6, 12), dtype=np.float32)

        while not self._stop.is_set():
            self.new_data_event.wait(timeout=0.1)
            self.new_data_event.clear()

            # We process a new PD/NN step every time we get a joint state callback (200 Hz).
            # Do NOT skip steps based on sim_time, as it can cause massive latency
            # if timestamps are coarse or not updated fast enough.
            
            # --- Internal Inference (50 Hz) ---
            if self.runner:
                if self.runner.should_step():
                    # 1. Use centralized parser for Standardization
                    state = self.telemetry.parse_bridge_data(
                        self.q,
                        self.dq,
                        self.base_quat,
                        self.base_ang_vel,
                        self.base_lin_vel_b,
                        self.base_pos,
                    )

                    # 2. Feed Policy (Unified Inference with Timing)
                    actions, _ = self.runner.infer(
                        state, self.cmd_vel, self.desired_qpos, self.mj_to_isaac
                    )

                    # 3. Use CommandProcessor for Sequenced Pipelining (Limit -> Sim -> ROS)
                    self.latest_targets[:] = self.command_processor.process(
                        actions, self.desired_qpos
                    )

            actuator_count += 1

            # Motor model matching MuJoCo
            targets = self.latest_targets
            kp, kd = 25.0, 0.5
            effort_limit, sat_effort, vel_lim = 23.5, 23.5, 30.0
            
            pos_err = targets - self.q
            raw_torques = kp * pos_err - kd * self.dq
            
            vel_at_lim = vel_lim * (1 + effort_limit / sat_effort)
            v_clamp = np.clip(self.dq, -vel_at_lim, vel_at_lim)
            t_top = sat_effort * (1.0 - v_clamp / vel_lim)
            t_bot = sat_effort * (-1.0 - v_clamp / vel_lim)
            
            pd_torques = np.clip(
                raw_torques, np.minimum(t_bot, -effort_limit), np.minimum(t_top, effort_limit)
            )

            # Using PD Controller by default as ActuatorNet was unstable in Gazebo
            self.latest_torques[:] = pd_torques

            # LOGGING FOR DIAGNOSIS (every 100 physics steps ~ 0.2s)
            if actuator_count % 100 == 0:
                print(
                    f"[GazeboBridge] t: {self.sim_time:.2f} | q[0]: {self.q[0]:.2f} | T_pd[0]: {pd_torques[0]:.2f}"
                )

            if count % 4 == 0:
                state = self.telemetry.parse_bridge_data(
                    self.q,
                    self.dq,
                    self.base_quat,
                    self.base_ang_vel,
                    self.base_lin_vel_b,
                    self.base_pos,
                )
                self.telemetry.publish(sim_time=self.sim_time, state=state)
            count += 1
            if count % 200 == 0:
                print(
                    f"\r[Bridge] t={self.sim_time:.2f} h={self.base_pos[2]:.2f} vx={self.base_lin_vel_b[0]:+.2f}   ",
                    end="",
                    flush=True,
                )

    def _cleanup(self):
        """Clean up Gazebo server and simulator processes."""
        self._stop.set()
        if hasattr(self, "gz_proc") and self.gz_proc:
            print(f"[GazeboDriver] Terminating simulator (PID {self.gz_proc.pid})...")
            try:
                os.killpg(os.getpgid(self.gz_proc.pid), signal.SIGTERM)
                self.gz_proc.wait(timeout=5)
            except Exception:
                subprocess.run(
                    ["pkill", "-9", "-f", "gz-sim-server"], stderr=subprocess.DEVNULL
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="go2")
    parser.add_argument("--world", type=str, default="quadruped_world")
    parser.add_argument(
        "--internal_policy",
        type=str,
        default=None,
        help="Path to policy checkpoint (Turbo Mode)",
    )
    parser.add_argument("--obs_dim", type=int, default=49)
    args = parser.parse_args()
    rclpy.init()
    node = Ros2GazeboDriver(
        args.robot, args.world, checkpoint=args.internal_policy, obs_dim=args.obs_dim
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._cleanup()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
