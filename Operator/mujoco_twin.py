import os
import sys
import time
import threading
import numpy as np
import argparse
import re

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
    def __init__(self, robot_type="go2", use_estimator=False):
        super().__init__("mujoco_twin_node")
        self.robot_type = robot_type

        # 1. Load MuJoCo Model
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Mujoco"))
        menagerie_dir = os.path.join(base_dir, "mujoco_menagerie", "unitree_go2")
        mjcf_path = os.path.join(menagerie_dir, "scene.xml")
        if not os.path.exists(mjcf_path):
            mjcf_path = os.path.join(base_dir, "scene.xml")
            menagerie_dir = base_dir

        # --- Generate Ghost Robot ---
        # 1. Generate go2_ghost.xml
        with open(os.path.join(menagerie_dir, "go2.xml"), "r") as f:
            go2_xml = f.read()
        
        go2_xml = go2_xml.replace('model="go2"', 'model="go2_ghost"')
        go2_xml = go2_xml.replace('name="', 'name="cmd_')
        go2_xml = go2_xml.replace('joint="', 'joint="cmd_')
        go2_xml = go2_xml.replace('child="', 'child="cmd_')
        go2_xml = go2_xml.replace('class="', 'class="cmd_')
        # Replace material definitions with transparent rgba (Greenish)
        go2_xml = re.sub(r'material="[^"]+"', 'rgba="0.2 0.8 0.2 0.4"', go2_xml)
        # Remove asset block to prevent duplicate mesh loads
        go2_xml = re.sub(r'<asset>.*?</asset>', '', go2_xml, flags=re.DOTALL)
        
        ghost_path = os.path.join(menagerie_dir, "twin_go2_ghost.xml")
        with open(ghost_path, "w") as f:
            f.write(go2_xml)
            
        # 2. Generate scene_twin.xml
        with open(mjcf_path, "r") as f:
            scene_xml = f.read()
            
        scene_xml = scene_xml.replace(
            '<include file="go2.xml"/>',
            '<include file="go2.xml"/>\n  <include file="twin_go2_ghost.xml"/>'
        )
        
        scene_twin_path = os.path.join(menagerie_dir, "twin_scene.xml")
        with open(scene_twin_path, "w") as f:
            f.write(scene_xml)
            
        mjcf_path = scene_twin_path
        # --- End Generate Ghost Robot ---

        self.get_logger().info(f"Loading MuJoCo Twin Model with Ghost from {mjcf_path}")
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
        self.cmd_qpos_addr = np.zeros(12, dtype=int)
        for i, name in enumerate(self.isaac_names):
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j_id != -1:
                self.qpos_addr[i] = self.model.jnt_qposadr[j_id]
                
            cmd_j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cmd_" + name)
            if cmd_j_id != -1:
                self.cmd_qpos_addr[i] = self.model.jnt_qposadr[cmd_j_id]

        # Resolve Base Addresses
        real_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if real_body_id != -1:
            self.real_base_addr = self.model.jnt_qposadr[self.model.body_jntadr[real_body_id]]
        else:
            self.real_base_addr = 0
            
        cmd_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cmd_base")
        if cmd_body_id != -1:
            self.cmd_base_addr = self.model.jnt_qposadr[self.model.body_jntadr[cmd_body_id]]
        else:
            self.cmd_base_addr = -1

        # 2. State Variables
        self.base_pos = np.array([0.0, 0.0, 0.50], dtype=np.float64)
        self.base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.joint_pos = np.zeros(12, dtype=np.float64)
        self.cmd_joint_pos = np.zeros(12, dtype=np.float64)
        self.base_lin_vel_body = np.zeros(3, dtype=np.float64)

        self.last_time = time.time()
        self.lock = threading.Lock()
        self.running = True

        # 3. ROS Subscriptions
        self.create_subscription(JointState, "/sensors/joint_states", self.joint_cb, 10)
        self.create_subscription(JointState, "/commands/joint_commands", self.cmd_joint_cb, 10)
        self.create_subscription(Imu, "/sensors/imu", self.imu_cb, 10)
        
        odom_topic = "/odom/state_estimator" if use_estimator else "/odom/state_simulator"
        self.get_logger().info(f"MuJoCo Twin subscribing to: {odom_topic}")
        self.create_subscription(Odometry, odom_topic, self.odom_cb, 10)

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

    def cmd_joint_cb(self, msg: JointState):
        with self.lock:
            for i, name in enumerate(msg.name):
                try:
                    idx = self.isaac_names.index(name)
                    self.cmd_joint_pos[idx] = msg.position[i]
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
                    # Update MuJoCo Data
                    if self.real_base_addr >= 0:
                        self.data.qpos[self.real_base_addr:self.real_base_addr+3] = self.base_pos
                        self.data.qpos[self.real_base_addr+3:self.real_base_addr+7] = self.base_quat
                    else:
                        self.data.qpos[0:3] = self.base_pos
                        self.data.qpos[3:7] = self.base_quat

                    if self.cmd_base_addr >= 0:
                        self.data.qpos[self.cmd_base_addr:self.cmd_base_addr+3] = self.base_pos
                        self.data.qpos[self.cmd_base_addr+3:self.cmd_base_addr+7] = self.base_quat

                    for i, addr in enumerate(self.qpos_addr):
                        if addr >= 0:
                            self.data.qpos[addr] = self.joint_pos[i]
                            
                    for i, addr in enumerate(self.cmd_qpos_addr):
                        if addr >= 0:
                            self.data.qpos[addr] = self.cmd_joint_pos[i]

                mujoco.mj_forward(self.model, self.data)
                viewer.sync()
                time.sleep(1.0 / 60.0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="go2")
    parser.add_argument("--use_estimator", action="store_true", help="Use estimated odometry instead of ground truth")
    args = parser.parse_args()

    rclpy.init()
    node = MujocoTwinNode(robot_type=args.robot, use_estimator=args.use_estimator)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.running = False
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
