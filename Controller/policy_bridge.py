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


class PolicyController(Node):
    """
    Unified ROS 2 Controller for Quadruped Locomotion.
    This node is 100% agnostic to the backend (Sim or Real).
    It only communicates via ROS 2 topics.
    """

    def __init__(self, checkpoint, robot_key, obs_dim=49):
        super().__init__("policy_controller")

        # 1. Initialize Policy
        self.runner = PolicyRunner(checkpoint, obs_dim=obs_dim, robot_type=robot_key)
        print(
            f"[PolicyController] Initialized for {robot_key} with policy {checkpoint}"
        )

        # 2. State Buffers
        self.imu_data = None
        self.joint_data = None
        self.base_lin_vel = [0.0, 0.0, 0.0]
        self.cmd_vel = [0.0, 0.0, 0.0, 0.0]  # vx, vy, wz, dummy
        self.last_actions = np.zeros(12, dtype=np.float32)
        self.last_targets = None

        # Default Pose (Type-Grouped: Hips, Thighs, Calves)
        self.desired_qpos = np.array(
            [
                0.1,
                -0.1,
                0.1,
                -0.1,  # Hips
                0.8,
                0.8,
                1.0,
                1.0,  # Thighs
                -1.5,
                -1.5,
                -1.5,
                -1.5,  # Calves
            ],
            dtype=np.float32,
        )

        # Mappings: Standardized (No re-mapping needed as both are Type-Grouped)
        self.mj_to_isaac = np.arange(12) 
        self.tg_to_isaac = np.arange(12) 

        # 3. ROS Subscriptions
        self.create_subscription(Imu, "/sensors/imu", self.imu_cb, 10)
        self.create_subscription(JointState, "/sensors/joint_states", self.joint_cb, 10)
        self.create_subscription(Odometry, "/odom", self.odom_cb, 10)
        self.create_subscription(Twist, "/cmd_vel", self.teleop_cb, 10)

        # 4. ROS Publisher
        self.command_pub = self.create_publisher(JointState, "/commands/joint_commands", 10)

        # Synchronization
        self.last_fire_time = -1.0
        self.FIRE_RATE = 1.0 / 50.0  # 50Hz AI control
        
        print("[PolicyController] Pure ROS controller started (1-to-1 mapping).")

    def imu_cb(self, msg):
        self.imu_data = {
            "quaternion": [msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z],
            "gyroscope": [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
        }

    def joint_cb(self, msg):
        self.joint_data = {"q": np.array(msg.position), "dq": np.array(msg.velocity)}
        # Trigger control loop based on joint states (primary heartbeat)
        self.control_loop()

    def odom_cb(self, msg):
        self.base_lin_vel = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]

    def teleop_cb(self, msg):
        # Clamp to training distribution [-1.0, 1.0]
        self.cmd_vel[0] = float(np.clip(msg.linear.x, -1.0, 1.0))
        self.cmd_vel[1] = float(np.clip(msg.linear.y, -1.0, 1.0))
        self.cmd_vel[2] = float(np.clip(msg.angular.z, -1.0, 1.0))

    def control_loop(self):
        """Main control loop triggered by joint state updates."""
        if self.imu_data is None or self.joint_data is None:
            return

        # Rate control: Ensure we only run at 50Hz regardless of sensor frequency
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.last_fire_time < self.FIRE_RATE:
            return
        self.last_fire_time = now

        # 1. Build Obs
        state = type("obj", (object,), {
                "imu": type("obj", (object,), self.imu_data),
                "motorState": [type("obj", (object,), {"q": q, "dq": dq})
                               for q, dq in zip(self.joint_data["q"], self.joint_data["dq"])],
                "base_lin_vel": self.base_lin_vel,
            })
        obs = self.runner.build_obs(state, self.cmd_vel, self.last_actions, self.desired_qpos, self.mj_to_isaac)

        # 2. Inference
        actions = self.runner.get_action(obs)
        self.last_actions[:] = actions

        # 3. Apply Scaling & Nominal Pose
        targets = actions * 0.25 + self.desired_qpos

        # 4. Publish
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"
        ]
        msg.position = targets.tolist()
        self.command_pub.publish(msg)

        # Console Output (Throttle to 25Hz for readability)
        if not hasattr(self, "_print_counter"): self._print_counter = 0
        self._print_counter += 1
        if self._print_counter % 2 == 0:
             # Tracking error (Actual - PREVIOUS target)
             if self.last_targets is not None:
                 t_err = np.mean(np.abs(self.joint_data["q"] - self.last_targets))
             else:
                 t_err = 0.0
             
             self.last_targets = targets.copy()
             act_mean = np.mean(np.abs(actions))
            
             # Detailed debug
             print(f"[Controller] cmd: {self.cmd_vel[:3]} | vx: {obs[0]:.3f} {obs[1]:.3f} | act: {act_mean:.3f} | grav: {obs[6]:.2f} {obs[7]:.2f} {obs[8]:.2f} | t_err: {t_err:.3f}   ", end="\r", flush=True)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--robot", type=str, default="go2")
    parser.add_argument("--obs_dim", type=int, default=49)
    args = parser.parse_args()

    rclpy.init()
    node = PolicyController(args.checkpoint, args.robot, args.obs_dim)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
