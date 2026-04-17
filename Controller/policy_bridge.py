import os
import sys
import time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Twist, Vector3, Quaternion
from nav_msgs.msg import Odometry
import numpy as np
import os
import sys
import argparse
import time

# Add parent directory to sys.path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Controller.policy_runner import PolicyRunner, quat_to_rot_matrix


class CommandProcessor:
    """
    Centralized component to handle policy outputs (Actions -> Robot).
    Handles hardware-aware scaling, limiting, and sequenced publishing.
    """
    def __init__(self, node, robot_type="go2", joint_names=None, saturation=0.9):
        self.node = node
        self.robot_type = robot_type
        self.saturation = saturation
        self.joint_names = joint_names or [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"
        ]
        
        # Hardware Limits (Unitree Go2 Standard)
        self.hard_min = np.array([
            -1.047, -1.047, -1.047, -1.047, # HAA
            -1.571, -1.571,                 # FL/FR Thigh
            -0.524, -0.524,                 # RL/RR Thigh
            -2.723, -2.723, -2.723, -2.723  # Calf
        ], dtype=np.float32)
        self.hard_max = np.array([
            1.047, 1.047, 1.047, 1.047,   # HAA
            3.491, 3.491,                 # FL/FR Thigh
            4.538, 4.538,                 # RL/RR Thigh
            -0.838, -0.838, -0.838, -0.838 # Calf
        ], dtype=np.float32)
        
        self.center = (self.hard_min + self.hard_max) / 2.0
        self.half_range = (self.hard_max - self.hard_min) / 2.0
        self.soft_min = (self.center - self.half_range * self.saturation).astype(np.float32)
        self.soft_max = (self.center + self.half_range * self.saturation).astype(np.float32)
        
        self.cmd_pub = self.node.create_publisher(JointState, '/commands/joint_commands', 10)
        self.node.get_logger().info(f"[CommandProcessor] Initialized for {robot_type} (Sat: {saturation*100}%)")

    def process(self, actions, desired_qpos, action_scale=0.25, send_to_robot_cb=None):
        targets = actions * action_scale + desired_qpos
        limited_targets = np.clip(targets, self.soft_min, self.soft_max)
        
        if send_to_robot_cb:
            send_to_robot_cb(limited_targets)
            
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = limited_targets.tolist()
        self.cmd_pub.publish(msg)
        
        return limited_targets
