#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import torch
import argparse
import signal
import threading
import subprocess
from pathlib import Path
from scipy.spatial.transform import Rotation

# ROS 2 Standard Imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Vector3

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


class Ros2GazeboBridge(Node):
    def __init__(self, robot_type, world_name="quadruped_world"):
        super().__init__("gazebo_bridge_node")
        self.robot_type = robot_type
        self.world_name = world_name

        # 4. ROS 2 Initialization
        self.joint_pub = self.create_publisher(JointState, "/sensors/joint_states", 10)
        self.imu_pub = self.create_publisher(Imu, "/sensors/imu", 10)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.create_subscription(JointState, "/commands/joint_commands", self._command_cb, 10)

        # 2. State & Control Buffers
        self.latest_torques = np.zeros(12, dtype=np.float32)
        self.latest_targets = np.array(
            [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5],
            dtype=np.float32,
        )
        self.sim_time = 0.0
        self.q = np.zeros(12)
        self.dq = np.zeros(12)
        self.base_pos = np.zeros(3)
        self.base_quat = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
        self.base_ang_vel = np.zeros(3)
        self.base_lin_vel_b = np.zeros(3)

        # 3. Gazebo Transport & Subscriptions
        self.gz_node = GzTransportNode()
        self.joint_pubs = []
        for jname in JOINT_NAMES:
            topic = f"/model/{self.robot_type}/joint/{jname}/cmd_force"
            self.joint_pubs.append(self.gz_node.advertise(topic, double_pb2.Double))
        
        self.new_data_event = threading.Event()
        self._stop = threading.Event()
        self.world_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "scene.sdf"))

        # 4. Starting Threads
        self.physics_thread = threading.Thread(target=self._physics_loop, daemon=True)
        self.repeater_thread = threading.Thread(target=self._repeater_loop, daemon=True)
        self.physics_thread.start()
        self.repeater_thread.start()

        print(f"[Ros2GazeboBridge] Initialized for {robot_type}. Physics at 500Hz (Slave).")

    def _command_cb(self, msg):
        if len(msg.position) == 12:
            self.latest_targets[:] = msg.position

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
        for joint in msg.joint:
            if joint.name in JOINT_NAMES:
                idx = JOINT_NAMES.index(joint.name)
                try:
                     # Mapping joint state (Gazebo -> Isaac order)
                     self.q[idx] = joint.axis1.position
                     self.dq[idx] = joint.axis1.velocity
                     
                     # Check for asymmetry: Isaac Lab usually expects mirror-symmetric signs 
                     # (e.g. positive HAA = Outward for both sides). 
                     # Gazebo SDF axis 1 0 0 on both sides is asymmetric.
                     if "rf_" in joint.name or "rh_" in joint.name:
                         # Mirror the right side joints if they are asymmetric in SDF
                         # Most Go1 policies expect symmetric HAA/HFE signs
                         if "_haa_" in joint.name: self.q[idx] *= -1.0; self.dq[idx] *= -1.0
                except AttributeError:
                    pass
        # Signal the physics loop to step
        self.new_data_event.set()

    def _odom_cb(self, msg):
        self.base_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        # msg.pose.orientation is [x, y, z, w]
        q = msg.pose.orientation
        
        # Gazebo Sim2Sim Odometry: 
        # Harmonize with MuJoCo bridge: World velocity rotated to Body Frame.
        v_world = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
        
        # Pull orientation from the odom pose [x, y, z, w]
        q = msg.pose.orientation
        q_scipy = [q.x, q.y, q.z, q.w]
        
        try:
            R = Rotation.from_quat(q_scipy).as_matrix()
            # v_body = R^T @ v_world
            self.base_lin_vel_b[:] = R.T @ v_world
        except Exception:
            self.base_lin_vel_b[:] = v_world # Fallback
        
        # Log Suspected Drift
        if np.abs(self.base_lin_vel_b[0]) > 2.0 and self.sim_time < 0.2:
             print(f"\r[Bridge] WARNING: Extreme vx={self.base_lin_vel_b[0]:.2f}. Collision likely.")

    def _repeater_loop(self):
        """
        Background thread to repeat the latest torques at 1000Hz.
        Gazebo clears forces every step; this ensures the robot stays powered.
        """
        while not self._stop.is_set():
            if hasattr(self, 'joint_pubs') and self.joint_pubs:
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
        subprocess.run(["pkill", "-9", "-f", "gz-sim-server"], stderr=subprocess.DEVNULL)
        time.sleep(1.0)
        
        # 3. Environment Setup (Crucial for Resource Path and Partition)
        env = os.environ.copy()
        
        # Ensure a truly unique partition for THIS specific bridge instance
        unique_id = os.getpid() % 100000
        partition = f"quadruped_sim_{unique_id}"
        env["GZ_PARTITION"] = partition
        
        # Resource paths for Gazebo to find models and meshes
        model_path = os.path.join(root_dir, "models")
        env["GZ_SIM_RESOURCE_PATH"] = root_dir + os.pathsep + model_path + os.pathsep + env.get("GZ_SIM_RESOURCE_PATH", "")
        
        # VDI Rendering Fixes
        if os.environ.get("FORCE_SOFTWARE_RENDER", "1") == "1":
            env["LIBGL_ALWAYS_SOFTWARE"] = "1"
            env["QT_X11_NO_MITSHM"] = "1"

        gz_args = ["gz", "sim", self.world_path]
        if os.environ.get("GZ_HEADLESS", "0") == "1":
            gz_args.append("-s")
        
        print(f"[Ros2GazeboBridge] Partition: {partition}")
        print(f"[Ros2GazeboBridge] Launching: {' '.join(gz_args)}")
        
        # Launch Gazebo in a new process group with the ENRICHED env
        self.gz_proc = subprocess.Popen(gz_args, env=env, preexec_fn=os.setsid)
        
        # Match our own process partition for topic discovery
        os.environ["GZ_PARTITION"] = partition
        time.sleep(5.0)

        # 4. Bind Subscriptions
        self.gz_node.subscribe(imu_pb2.IMU, f"/model/{self.robot_type}/link/base/sensor/imu/imu", self._imu_cb)
        self.gz_node.subscribe(model_pb2.Model, f"/model/{self.robot_type}/joint_state", self._joint_cb)
        self.gz_node.subscribe(odometry_pb2.Odometry, f"/model/{self.robot_type}/odometry", self._odom_cb)
        self.gz_node.subscribe(world_stats_pb2.WorldStatistics, f"/world/{self.world_name}/stats", self._stats_cb)

        # 5. Wait for First State
        while (self.sim_time == 0 or np.all(self.q == 0)) and not self._stop.is_set():
            time.sleep(0.1)
        
        print("[Ros2GazeboBridge] First joint state received. Initializing targets.")
        self.latest_targets[:] = self.q
            
        pos_err_hist = np.zeros((3, 12), dtype=np.float32)
        vel_hist = np.zeros((3, 12), dtype=np.float32)
        last_processed_sim_time = -1.0
        count = 0

        while not self._stop.is_set():
            self.new_data_event.wait()
            self.new_data_event.clear()

            if self.sim_time == last_processed_sim_time:
                continue
            last_processed_sim_time = self.sim_time

            # State for ActuatorNet
            targets = self.latest_targets
            pos_err_hist = np.roll(pos_err_hist, 1, 0); pos_err_hist[0] = self.q - targets
            vel_hist = np.roll(vel_hist, 1, 0); vel_hist[0] = self.dq

            net_in = torch.zeros((12, 6))
            net_in[:, :3] = torch.from_numpy(pos_err_hist.T.copy())
            net_in[:, 3:] = torch.from_numpy(vel_hist.T.copy())

            with torch.no_grad():
                torques = self.act_net(net_in).squeeze().numpy()
            
            # Sign mirroring for right side HAA
            final_torques = torques.copy()
            for j_idx, jname in enumerate(JOINT_NAMES):
                if ("rf_" in jname or "rh_" in jname) and "_haa_" in jname:
                    final_torques[j_idx] *= -1.0
            
            self.latest_torques[:] = np.clip(final_torques, -23.7, 23.7)

            if count % 4 == 0:
                self._publish_ros2()
            count += 1
            if count % 200 == 0:
                print(f"\r[Bridge] t={self.sim_time:.2f} h={self.base_pos[2]:.2f} vx={self.base_lin_vel_b[0]:+.2f}   ", end="", flush=True)

    def _cleanup(self):
        """Clean up Gazebo server and simulator processes."""
        self._stop.set()
        if hasattr(self, 'gz_proc') and self.gz_proc:
            print(f"[Ros2GazeboBridge] Terminating simulator (PID {self.gz_proc.pid})...")
            try:
                os.killpg(os.getpgid(self.gz_proc.pid), signal.SIGTERM)
                self.gz_proc.wait(timeout=5)
            except Exception:
                subprocess.run(["pkill", "-9", "-f", "gz-sim-server"], stderr=subprocess.DEVNULL)

    def _publish_ros2(self):
        # Use Gazebo simulation time for all headers to ensure PolicyBridge sync
        msg_time = rclpy.time.Time(seconds=self.sim_time).to_msg()
        
        # Joint States
        js = JointState()
        js.header.stamp = msg_time
        js.name = JOINT_NAMES
        js.position = self.q.tolist()
        js.velocity = self.dq.tolist()
        self.joint_pub.publish(js)

        # IMU
        imu = Imu()
        imu.header.stamp = msg_time
        imu.orientation = Quaternion(w=float(self.base_quat[0]), x=float(self.base_quat[1]), y=float(self.base_quat[2]), z=float(self.base_quat[3]))
        imu.angular_velocity = Vector3(x=float(self.base_ang_vel[0]), y=float(self.base_ang_vel[1]), z=float(self.base_ang_vel[2]))
        self.imu_pub.publish(imu)

        # Odometry
        odom = Odometry()
        odom.header.stamp = msg_time
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base"
        odom.twist.twist.linear = Vector3(x=float(self.base_lin_vel_b[0]), y=float(self.base_lin_vel_b[1]), z=float(self.base_lin_vel_b[2]))
        self.odom_pub.publish(odom)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="go1")
    parser.add_argument("--world", type=str, default="quadruped_world")
    args = parser.parse_args()
    rclpy.init()
    node = Ros2GazeboBridge(args.robot, args.world)
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
