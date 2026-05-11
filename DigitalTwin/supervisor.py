import os
import sys
import time
import argparse
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32

# Ensure absolute path of the repository is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config_loader import load_config
from Telemetry.estimator import rot_from_quat

class SupervisorNode(Node):
    """
    Supervisor Node.
    Subscribes to robot telemetry and runs a safety check loop. 
    Publishes max torque overrides to /safety/max_torque.
    """
    def __init__(self, robot_type="go2"):
        super().__init__("supervisor_node")
        self.robot_type = robot_type
        
        # 1. Load Configuration
        self.config = load_config()
        self.safety_cfg = self.config.get("safety", {})
        self.freq = self.safety_cfg.get("supervisor_frequency", 10.0)

        # State Variables
        self.base_pos = np.array([0.0, 0.0, 0.35], dtype=np.float64)
        self.base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64) # [w, x, y, z]
        self.joint_pos = np.zeros(12, dtype=np.float64)
        self.base_lin_vel_body = np.zeros(3, dtype=np.float64)
        
        # 2. ROS Subscriptions
        self.create_subscription(JointState, "/sensors/joint_states", self.joint_cb, 10)
        self.create_subscription(Imu, "/sensors/imu", self.imu_cb, 10)
        self.create_subscription(Odometry, "/odom", self.odom_cb, 10)

        # 3. ROS Publishers
        self.max_torque_pub = self.create_publisher(Float32, "/safety/max_torque", 10)

        # 4. Timer Loops
        # Safety / NN Loop
        self.timer_period = 1.0 / self.freq
        self.create_timer(self.timer_period, self.safety_loop)

        self.get_logger().info(f"Supervisor Node initialized at {self.freq}Hz. Safety loop running.")

    def joint_cb(self, msg: JointState):
        # We don't strictly need this for safety yet, but good for future NN checks
        pass

    def imu_cb(self, msg: Imu):
        q = msg.orientation
        self.base_quat = np.array([q.w, q.x, q.y, q.z])

    def odom_cb(self, msg: Odometry):
        v = msg.twist.twist.linear
        self.base_lin_vel_body = np.array([v.x, v.y, v.z])

    def safety_loop(self):
        """
        Future home of the Safety Neural Network.
        For now, it just continuously publishes the default max torque.
        """
        # TODO: Implement NN check here
        robot_is_safe = True 
        
        msg = Float32()
        if robot_is_safe:
            msg.data = 23.5  # Typical Go2 max torque (will be clipped by CommandProcessor if needed)
        else:
            msg.data = 0.0
            
        self.max_torque_pub.publish(msg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="go2")
    args = parser.parse_args()

    rclpy.init()
    node = SupervisorNode(robot_type=args.robot)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
