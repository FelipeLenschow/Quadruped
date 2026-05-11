import os
import sys
import time
import threading
import numpy as np
import argparse

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry

import mujoco
import mujoco.viewer

# Ensure absolute path of the repository is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Telemetry.estimator import rot_from_quat

class MujocoTwinNode(Node):
    """
    Passive MuJoCo Digital Twin Node.
    Listens to ROS 2 topics and updates the MuJoCo visualization.
    """
    def __init__(self, robot_type="go2"):
        super().__init__("mujoco_twin_node")
        self.robot_type = robot_type

        # 1. Load MuJoCo Model
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Mujoco"))
        mjcf_path = os.path.join(
            base_dir, "mujoco_menagerie", "unitree_go2", "scene.xml"
        )
        if not os.path.exists(mjcf_path):
            mjcf_path = os.path.join(base_dir, "scene.xml")

        self.get_logger().info(f"Loading MuJoCo Twin Model from {mjcf_path}")
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)

        # Disable Gravity and Collisions
        self.model.opt.gravity[:] = 0.0
        self.model.geom_conaffinity[:] = 0
        self.model.geom_contype[:] = 0

        # Resolve joint indices
        self.isaac_names = [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
        ]
        self.qpos_addr = np.zeros(12, dtype=int)
        for i, name in enumerate(self.isaac_names):
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j_id != -1:
                self.qpos_addr[i] = self.model.jnt_qposadr[j_id]

        # 2. State Variables
        self.base_pos = np.array([0.0, 0.0, 0.35], dtype=np.float64)
        self.base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.joint_pos = np.zeros(12, dtype=np.float64)
        self.base_lin_vel_body = np.zeros(3, dtype=np.float64)

        self.last_time = time.time()
        self.lock = threading.Lock()
        self.running = True

        # 3. ROS Subscriptions
        self.create_subscription(JointState, "/sensors/joint_states", self.joint_cb, 10)
        self.create_subscription(Imu, "/sensors/imu", self.imu_cb, 10)
        self.create_subscription(Odometry, "/odom", self.odom_cb, 10)

        # 4. Start Viewer Thread
        self.viewer_thread = threading.Thread(target=self._viewer_loop, daemon=True)
        self.viewer_thread.start()

    def joint_cb(self, msg: JointState):
        with self.lock:
            for i, name in enumerate(msg.name):
                try:
                    idx = self.isaac_names.index(name)
                    self.joint_pos[idx] = msg.position[i]
                except ValueError:
                    pass

    def imu_cb(self, msg: Imu):
        with self.lock:
            q = msg.orientation
            self.base_quat = np.array([q.w, q.x, q.y, q.z])

    def odom_cb(self, msg: Odometry):
        with self.lock:
            v = msg.twist.twist.linear
            self.base_lin_vel_body = np.array([v.x, v.y, v.z])

    def _viewer_loop(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            track_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
            if track_id == -1:
                track_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
            if track_id != -1:
                viewer.cam.trackbodyid = track_id

            while self.running and rclpy.ok() and viewer.is_running():
                current_time = time.time()
                dt = current_time - self.last_time
                self.last_time = current_time

                with self.lock:
                    # Integrate position
                    R = rot_from_quat(self.base_quat)
                    v_world = R @ self.base_lin_vel_body
                    self.base_pos[0] += v_world[0] * dt
                    self.base_pos[1] += v_world[1] * dt

                    # Update MuJoCo Data
                    self.data.qpos[0:3] = self.base_pos
                    self.data.qpos[3:7] = self.base_quat
                    for i, addr in enumerate(self.qpos_addr):
                        if addr > 0:
                            self.data.qpos[addr] = self.joint_pos[i]

                mujoco.mj_forward(self.model, self.data)
                viewer.sync()
                time.sleep(1.0 / 60.0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="go2")
    args = parser.parse_args()

    rclpy.init()
    node = MujocoTwinNode(robot_type=args.robot)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.running = False
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
