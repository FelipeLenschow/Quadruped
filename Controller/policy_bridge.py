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
from configs.config_loader import load_config
import yaml
from std_msgs.msg import Float32

class CommandProcessor:
    """
    Centralized component to handle policy outputs (Actions -> Robot).
    Handles hardware-aware scaling, limiting, and sequenced publishing.
    """
    def __init__(self, node, robot_type="go2", joint_names=None):
        self.node = node
        self.robot_type = robot_type
        
        # Load Configuration
        self.config = load_config()
        if not self.config:
            self.node.get_logger().error("[CommandProcessor] Failed to load config. Using hardcoded safety defaults.")

        self.saturation = self.config.get("saturation_limit", 0.9)
        self.action_scale = self.config.get("action_scale", 0.25)
        
        self.joint_names = joint_names or [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"
        ]
        
        # Hardware Limits from Config
        limits = self.config.get("joint_limits", {})
        abd = limits.get("abduction", {"min": -1.04, "max": 1.04})
        hip = limits.get("hip", {"min": -1.57, "max": 3.49})
        knee = limits.get("knee", {"min": -2.72, "max": -0.83})

        # Build limit arrays
        self.hard_min = np.array([
            abd["min"]] * 4 + [hip["min"]] * 4 + [knee["min"]] * 4, dtype=np.float32)
        self.hard_max = np.array([
            abd["max"]] * 4 + [hip["max"]] * 4 + [knee["max"]] * 4, dtype=np.float32)
        
        # RL Thigh limits are usually different in training (wider range)
        # But for safety we use the conservative ones from config
        
        self.center = (self.hard_min + self.hard_max) / 2.0
        self.half_range = (self.hard_max - self.hard_min) / 2.0
        self.soft_min = (self.center - self.half_range * self.saturation).astype(np.float32)
        self.soft_max = (self.center + self.half_range * self.saturation).astype(np.float32)
        
        self.cmd_pub = self.node.create_publisher(JointState, '/commands/joint_commands', 10)
        
        # Safety / Max Torque
        self.safety_cfg = self.config.get("safety", {})
        self.watchdog_timeout = self.safety_cfg.get("watchdog_timeout", 1.0)
        self.global_max_torque = self.safety_cfg.get("global_max_torque", 23.5)
        
        self.active_max_torque = 0.0     # Fail-safe start: Zero torque until supervisor says otherwise
        self.last_torque_msg_time = 0.0  # 0 means no supervisor heartbeat received yet
        self.has_received_supervisor_msg = False
        
        self.node.create_subscription(Float32, "/safety/max_torque", self.max_torque_cb, 10)
        
        self.node.get_logger().info(f"[CommandProcessor] Initialized for {robot_type} (Sat: {self.saturation*100}%)")
        self.node.get_logger().info(f"[CommandProcessor] Safety: Timeout={self.watchdog_timeout}s, GlobalMax={self.global_max_torque}Nm")

    def max_torque_cb(self, msg: Float32):
        # Apply global limit
        self.active_max_torque = min(msg.data, self.global_max_torque)
        self.last_torque_msg_time = time.time()
        self.has_received_supervisor_msg = True

    def process(self, actions, desired_qpos, action_scale=None, send_to_robot_cb=None):
        # Watchdog logic
        now = time.time()
        if not self.has_received_supervisor_msg:
            # Haven't received anything yet
            self.active_max_torque = 0.0
        elif now - self.last_torque_msg_time > self.watchdog_timeout:
            # Heartbeat lost
            if self.active_max_torque > 0.0:
                self.node.get_logger().warn(f"[CommandProcessor] Safety Watchdog triggered (> {self.watchdog_timeout}s)! Max torque set to 0.0", throttle_duration_sec=2.0)
            self.active_max_torque = 0.0
            
        # Use config action_scale unless overridden
        scale = action_scale if action_scale is not None else self.action_scale
        targets = actions * scale + desired_qpos
        limited_targets = np.clip(targets, self.soft_min, self.soft_max)
        
        if send_to_robot_cb:
            send_to_robot_cb(limited_targets)
            
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = limited_targets.tolist()
        msg.effort = [float(self.active_max_torque)] * len(self.joint_names)
        self.cmd_pub.publish(msg)
        
        return limited_targets
