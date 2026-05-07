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

from Controller.Utils.state_estimator import rot_from_quat
from DigitalTwin.mujoco_twin import MujocoTwin

class SupervisorNode(Node):
    """
    Supervisor Node.
    Subscribes to robot telemetry, feeds data to the Digital Twin for rendering,
    and runs a safety check loop. Publishes max torque overrides if necessary.
    """
    def __init__(self, robot_type="go2"):
        super().__init__("supervisor_node")
        self.robot_type = robot_type

        # 1. Initialize the Twin
        self.twin = MujocoTwin(robot_type=robot_type)
        
        self.isaac_names = self.twin.isaac_names

        # State Variables
        self.base_pos = np.array([0.0, 0.0, 0.35], dtype=np.float64)
        self.base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64) # [w, x, y, z]
        self.joint_pos = np.zeros(12, dtype=np.float64)
        self.base_lin_vel_body = np.zeros(3, dtype=np.float64)
        
        self.last_odom_time = time.time()

        # 2. ROS Subscriptions
        self.create_subscription(JointState, "/sensors/joint_states", self.joint_cb, 10)
        self.create_subscription(Imu, "/sensors/imu", self.imu_cb, 10)
        self.create_subscription(Odometry, "/odom", self.odom_cb, 10)

        # 3. ROS Publishers
        self.max_torque_pub = self.create_publisher(Float32, "/safety/max_torque", 10)

        # 4. Timer Loops
        # Physics update loop (~60 Hz to match rendering)
        self.create_timer(1.0 / 60.0, self.update_twin_loop)
        
        # Safety / NN Loop (~10 Hz)
        self.create_timer(0.1, self.safety_loop)

        self.get_logger().info("Supervisor Node initialized. Twin is running.")

    def joint_cb(self, msg: JointState):
        for i, name in enumerate(msg.name):
            try:
                idx = self.isaac_names.index(name)
                self.joint_pos[idx] = msg.position[i]
            except ValueError:
                pass

    def imu_cb(self, msg: Imu):
        q = msg.orientation
        self.base_quat = np.array([q.w, q.x, q.y, q.z])

    def odom_cb(self, msg: Odometry):
        v = msg.twist.twist.linear
        self.base_lin_vel_body = np.array([v.x, v.y, v.z])

    def update_twin_loop(self):
        current_time = time.time()
        dt = current_time - self.last_odom_time
        self.last_odom_time = current_time

        # Integrate position (rotate body velocity to world frame)
        R = rot_from_quat(self.base_quat)
        v_world = R @ self.base_lin_vel_body
        
        self.base_pos[0] += v_world[0] * dt
        self.base_pos[1] += v_world[1] * dt
        
        # Feed data to the twin
        self.twin.update_state(self.joint_pos, self.base_quat, self.base_pos)

    def safety_loop(self):
        """
        Future home of the Safety Neural Network.
        For now, it just continuously publishes the default max torque.
        """
        # TODO: Implement NN check here
        robot_is_safe = True 
        
        msg = Float32()
        if robot_is_safe:
            msg.data = 23.5  # Typical Go2 max torque
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
        node.twin.stop()
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
