import os
import sys
import time
import numpy as np
import torch
import argparse
import threading
import subprocess

# ROS 2 (for Sim mode)
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Header

# Project Imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Deployment.policy_runner import PolicyRunner


class PolicyBridge(Node):
    def __init__(self, backend, checkpoint, robot_key, obs_dim=49, sim_type=None):
        super().__init__("policy_bridge")
        self.backend = backend  # 'sim' or 'real'
        self.robot_key = robot_key

        # 1. Initialize Policy
        self.runner = PolicyRunner(checkpoint, obs_dim=obs_dim, robot_type=robot_key)
        print(
            f"[PolicyBridge] Backend: {backend.upper()} | Robot: {robot_key} | Policy: {checkpoint}"
        )

        # 2. State Buffers & Constants
        self.imu_data = None
        self.joint_data = None
        self.cmd_vel = [0.0, 0.0, 0.0, 0.0]  # vx, vy, wz, dummy
        self.base_lin_vel = [0.0, 0.0, 0.0]
        self.last_actions = np.zeros(12, dtype=np.float32)

        # Synchronization tracking
        self.timestamps = {"imu": 0, "joint": 0, "odom": 0}
        self.last_sync_timestamp = -1

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
        # Mapping (ROS topic is Type-Grouped)
        self.mj_to_isaac = np.arange(12)

        # 3. Backend Specific Setup
        if self.backend == "sim":
            self._init_sim(sim_type)
        else:
            self._init_real()

        # 4. Timer Loop (50Hz - Only for REAL robot)
        if self.backend != "sim":
            self.timer = self.create_timer(0.02, self.control_loop)

    def _init_sim(self, sim_type):
        # ROS 2 Subscriptions
        self.create_subscription(Imu, "/sensors/imu", self.imu_cb, 10)
        self.create_subscription(JointState, "/sensors/joint_states", self.joint_cb, 10)
        self.create_subscription(Odometry, "/odom", self.odom_cb, 10)
        # Note: In Sim mode, teleop often comes from another ROS topic (/cmd_vel)
        self.create_subscription(Twist, "/cmd_vel", self.teleop_cb, 10)

        # ROS 2 Publisher
        self.command_pub = self.create_publisher(
            JointState, "/commands/joint_commands", 10
        )

        # Auto-launch Sim Bridge if requested
        if sim_type:
            bridge_script = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "Mujoco" if sim_type == "mujoco" else "Gazebo",
                    f"ros2_{sim_type}_bridge.py",
                )
            )
            print(f"[PolicyBridge] Launching {sim_type} simulation bridge...")
            sim_env = os.environ.copy()
            sim_env["LIBGL_ALWAYS_SOFTWARE"] = "1"
            sim_env["QT_X11_NO_MITSHM"] = "1"
            sim_env["GZ_PARTITION"] = "quadruped_sim"
            sim_env["GZ_HEADLESS"] = "0"
            sim_env["FORCE_SOFTWARE_RENDER"] = "1"
            self.sim_proc = subprocess.Popen(
                ["/usr/bin/python3", bridge_script, f"--robot={self.robot_key}"],
                env=sim_env,
            )
        else:
            self.sim_proc = None

    def _init_real(self):
        # Unitree SDK Setup
        try:
            import unitree_legged_sdk as sdk

            self.sdk = sdk
            self.udp = sdk.UDP(sdk.LOWLEVEL, 8080, "192.168.123.10", 8007)
            self.low_cmd = sdk.LowCmd()
            self.low_state = sdk.LowState()
            self.udp.InitCmdData(self.low_cmd)
            print("[PolicyBridge] Unitree SDK Initialized.")
        except ImportError:
            print("[Error] Unitree SDK not found. Real mode will fail.")
            sys.exit(1)

    # Callbacks
    def imu_cb(self, msg):
        self.imu_data = {
            "quaternion": [
                msg.orientation.w,
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
            ],
            "gyroscope": [
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ],
        }

    def joint_cb(self, msg):
        self.joint_data = {"q": np.array(msg.position), "dq": np.array(msg.velocity)}
        # Joint state is the primary trigger — only fire if we have all required data
        if self.backend == "sim" and self.imu_data is not None:
            self.control_loop()

    def teleop_cb(self, msg):
        # Clamp to training distribution ranges (from QuadrupedEnvCfg):
        # command_x_range = (-1.0, 1.0), command_y_range = (-1.0, 1.0), command_yaw_range = (-1.0, 1.0)
        self.cmd_vel[0] = float(np.clip(msg.linear.x, -1.0, 1.0))
        self.cmd_vel[1] = float(np.clip(msg.linear.y, -1.0, 1.0))
        self.cmd_vel[2] = float(np.clip(msg.angular.z, -1.0, 1.0))

    def odom_cb(self, msg):
        self.base_lin_vel = [
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
        ]

    def control_loop(self):
        if self.backend == "sim":
            self._control_step_sim()
        else:
            self._control_step_real()

    def _control_step_sim(self):
        if self.imu_data is None or self.joint_data is None:
            return

        # 1. Build Obs
        state = type(
            "obj",
            (object,),
            {
                "imu": type("obj", (object,), self.imu_data),
                "motorState": [
                    type("obj", (object,), {"q": q, "dq": dq})
                    for q, dq in zip(self.joint_data["q"], self.joint_data["dq"])
                ],
                "base_lin_vel": self.base_lin_vel,
            },
        )
        obs = self.runner.build_obs(
            state, self.cmd_vel, self.last_actions, self.desired_qpos, self.mj_to_isaac
        )

        # DEBUG
        if not hasattr(self, "_print_counter"):
            self._print_counter = 0
        self._print_counter += 1
        if self._print_counter % 25 == 0:
            print(
                f"\r[PolicyBridge] Active cmd_vel: {self.cmd_vel} | vx={self.base_lin_vel[0]:.3f} | actions_mean={np.mean(self.last_actions):.3f}   ",
                end="",
                flush=True,
            )

        # 2. Inference
        actions = self.runner.get_action(obs)
        self.last_actions[:] = actions

        # 3. Apply Scaling & Nominal Pose
        # Standard Isaac Lab pattern: target = action * scale + default_pose
        targets = actions * 0.25 + self.desired_qpos

        # 4. Publish
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = targets.tolist()
        self.command_pub.publish(msg)

    def _control_step_real(self):
        # 1. Receive State
        self.udp.Recv()
        self.udp.GetRecv(self.low_state)

        # 2. Build Obs (Matching Sim structure for symmetry)
        # Ported from legacy real2policy_bridge.py
        state_obj = type(
            "obj",
            (object,),
            {
                "imu": self.low_state.imu,
                "motorState": self.low_state.motorState,
                "base_lin_vel": [0.0, 0.0, 0.0],
            },
        )
        obs = self.runner.build_obs(
            state_obj,
            self.cmd_vel,
            self.last_actions,
            self.desired_qpos,
            self.mj_to_isaac,
        )

        # 3. Inference
        actions = self.runner.get_action(obs)
        self.last_actions[:] = actions

        # 4. Map Actions to SDK Command
        # Apply scaling and nominal pose offset
        targets = actions * 0.25 + self.desired_qpos

        # For Go1/A1, we send position targets
        for i in range(12):
            self.low_cmd.motorCmd[i].q = float(targets[i])
            self.low_cmd.motorCmd[i].Kp = 80.0
            self.low_cmd.motorCmd[i].Kd = 3.0
            self.low_cmd.motorCmd[i].tau = 0.0

        # 5. Send Command
        self.udp.SetSend(self.low_cmd)
        self.udp.Send()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="sim", choices=["sim", "real"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--robot", type=str, default="go2")
    parser.add_argument("--obs_dim", type=int, default=49)
    parser.add_argument("--sim", type=str, default=None, choices=["mujoco", "gazebo"])
    args = parser.parse_args()

    # Handle policy path
    ckpt = args.checkpoint
    if not os.path.isabs(ckpt):
        p_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "policies", ckpt)
        )
        if os.path.exists(p_path):
            ckpt = p_path

    rclpy.init()
    node = PolicyBridge(args.backend, ckpt, args.robot, args.obs_dim, args.sim)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    if hasattr(node, "sim_proc") and node.sim_proc:
        node.sim_proc.terminate()
    if rclpy.ok():
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
