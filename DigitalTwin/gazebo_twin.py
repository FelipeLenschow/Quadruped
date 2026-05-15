import os
import sys
import time
import threading
import numpy as np
import argparse
import subprocess
import signal

# Ensure absolute path of the repository is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry

from Configs.config_loader import load_config
from Telemetry.estimator import rot_from_quat

# Gazebo Transport & Msgs
try:
    from gz.transport13 import Node as GzTransportNode
    from gz.msgs10 import (
        double_pb2,
        pose_pb2,
        vector3d_pb2,
        quaternion_pb2,
        pose_v_pb2
    )
except ImportError:
    print("[ERROR] Gazebo (Harmonic) Python bindings not found.")
    sys.exit(1)

JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

# Standard Isaac ordering for ROS 2 incoming messages
ISAAC_NAMES = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]

class GazeboTwinNode(Node):
    """
    Passive Gazebo Digital Twin Node.
    Listens to ROS 2 topics and updates the Gazebo visualization.
    """
    def __init__(self, robot_type="go2"):
        super().__init__("gazebo_twin_node")
        self.robot_type = robot_type

        # 1. State Variables
        self.base_pos = np.array([0.0, 0.0, 0.35], dtype=np.float64)
        self.base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.joint_pos = np.zeros(12, dtype=np.float64)
        self.base_lin_vel_body = np.zeros(3, dtype=np.float64)

        # Internal Gazebo state for PD tracking
        self.gz_q = np.zeros(12, dtype=np.float64)
        self.gz_dq = np.zeros(12, dtype=np.float64)

        self.last_time = time.time()
        self.lock = threading.Lock()
        self.running = True

        # 2. Start Gazebo Server
        self.world_name = "quadruped_world"
        self._stop = threading.Event()
        self.gz_proc = None
        self.physics_thread = threading.Thread(target=self._gazebo_loop, daemon=True)
        self.physics_thread.start()

        # 3. ROS Subscriptions
        self.create_subscription(JointState, "/sensors/joint_states", self.joint_cb, 10)
        self.create_subscription(Imu, "/sensors/imu", self.imu_cb, 10)
        self.create_subscription(Odometry, "/odom/simulator", self.odom_cb, 10)

    def joint_cb(self, msg: JointState):
        with self.lock:
            for i, name in enumerate(msg.name):
                try:
                    # We receive in Isaac order, map to our internal Gazebo order
                    isaac_idx = ISAAC_NAMES.index(name)
                    gz_idx = JOINT_NAMES.index(name)
                    self.joint_pos[gz_idx] = msg.position[i]
                except ValueError:
                    pass

    def imu_cb(self, msg: Imu):
        with self.lock:
            q = msg.orientation
            self.base_quat = np.array([q.w, q.x, q.y, q.z])

    def odom_cb(self, msg: Odometry):
        with self.lock:
            p = msg.pose.pose.position
            self.base_pos = np.array([p.x, p.y, p.z])
            v = msg.twist.twist.linear
            self.base_lin_vel_body = np.array([v.x, v.y, v.z])

    def _gz_joint_cb(self, msg):
        # High frequency internal Gazebo state
        for joint in msg.joint:
            if joint.name in JOINT_NAMES:
                idx = JOINT_NAMES.index(joint.name)
                try:
                    self.gz_q[idx] = joint.axis1.position
                    self.gz_dq[idx] = joint.axis1.velocity
                except AttributeError:
                    pass

    def _gazebo_loop(self):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Gazebo"))
        world_path = os.path.join(root_dir, "scene.sdf")

        # Create a zero-gravity twin world to prevent falling
        with open(world_path, "r") as f:
            sdf_content = f.read()
        sdf_content = sdf_content.replace("<world name=\"quadruped_world\">", "<world name=\"quadruped_world\">\n        <gravity>0 0 0</gravity>")
        twin_world_path = "/tmp/scene_twin.sdf"
        with open(twin_world_path, "w") as f:
            f.write(sdf_content)

        subprocess.run(["pkill", "-9", "-f", "gz-sim-server"], stderr=subprocess.DEVNULL)
        time.sleep(1.0)

        env = os.environ.copy()
        unique_id = os.getpid() % 100000
        partition = f"quadruped_twin_{unique_id}"
        env["GZ_PARTITION"] = partition
        env["GZ_SIM_RESOURCE_PATH"] = (
            root_dir + os.pathsep + os.path.join(root_dir, "models") + os.pathsep +
            os.path.abspath(os.path.join(root_dir, "..", "Unitree_Go2", "models")) +
            os.pathsep + env.get("GZ_SIM_RESOURCE_PATH", "")
        )

        gz_args = ["gz", "sim", twin_world_path]
        print(f"[GazeboTwin] Launching Gazebo Twin on partition: {partition} (Zero Gravity)")
        self.gz_proc = subprocess.Popen(gz_args, env=env, preexec_fn=os.setsid)

        os.environ["GZ_PARTITION"] = partition
        self.gz_node = GzTransportNode()
        time.sleep(5.0)

        # Subscribe to internal gazebo state for PD control
        from gz.msgs10 import model_pb2
        self.gz_node.subscribe(model_pb2.Model, f"/model/{self.robot_type}/joint_state", self._gz_joint_cb)

        # Publishers
        self.pose_pub = self.gz_node.advertise(f"/world/{self.world_name}/set_pose", pose_pb2.Pose)
        self.joint_pubs = []
        for jname in JOINT_NAMES:
            topic = f"/model/{self.robot_type}/joint/{jname}/cmd_force"
            self.joint_pubs.append(self.gz_node.advertise(topic, double_pb2.Double))
            
        print("[GazeboTwin] Syncing Telemetry with Gazebo...")

        while not self._stop.is_set():
            current_time = time.time()
            dt = current_time - self.last_time
            self.last_time = current_time

            with self.lock:
                # Update Base Pose
                pose_msg = pose_pb2.Pose()
                pose_msg.name = self.robot_type
                pose_msg.position.x = self.base_pos[0]
                pose_msg.position.y = self.base_pos[1]
                pose_msg.position.z = self.base_pos[2]
                pose_msg.orientation.w = self.base_quat[0]
                pose_msg.orientation.x = self.base_quat[1]
                pose_msg.orientation.y = self.base_quat[2]
                pose_msg.orientation.z = self.base_quat[3]
                self.pose_pub.publish(pose_msg)

                # Update Joints (using PD torque control)
                kp = 80.0
                kd = 2.0
                torques = kp * (self.joint_pos - self.gz_q) - kd * self.gz_dq
                
                for i, pub in enumerate(self.joint_pubs):
                    msg = double_pb2.Double()
                    msg.data = float(torques[i])
                    pub.publish(msg)

            time.sleep(0.001) # 1000 Hz PD sync

    def _cleanup(self):
        self._stop.set()
        if self.gz_proc:
            try:
                os.killpg(os.getpgid(self.gz_proc.pid), signal.SIGTERM)
                self.gz_proc.wait(timeout=5)
            except Exception:
                subprocess.run(["pkill", "-9", "-f", "gz-sim-server"], stderr=subprocess.DEVNULL)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="go2")
    args = parser.parse_args()

    rclpy.init()
    node = GazeboTwinNode(robot_type=args.robot)
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
