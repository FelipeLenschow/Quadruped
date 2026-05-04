import os
import sys
import time
import numpy as np
import argparse
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "unitree_sdk2_python"))
)

from Controller.policy_runner import PolicyRunner
from Controller.policy_bridge import CommandProcessor
from Controller.Utils.telemetry import TelemetryManager

# SDK2 Imports
from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


class RealDriver(Node):
    def __init__(self, robot="go2", internal_policy=None, obs_dim=45):
        super().__init__("real_driver")
        self.robot_type = robot

        # 1. SDK2 Initialization
        # On-robot, we usually use the default network interface
        ChannelFactoryInitialize(0)
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.low_state_handler, 10)

        self.low_state = None
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.crc = CRC()

        # 2. Controller Utilities
        self.telemetry = TelemetryManager(self, robot_type=robot)
        self.command_processor = CommandProcessor(self, robot_type=robot)

        # 3. Internal Inference
        self.policy = None
        self.desired_qpos = np.array(
            [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5],
            dtype=np.float32,
        )
        if internal_policy:
            self.policy = PolicyRunner(
                internal_policy, obs_dim=obs_dim, robot_type=robot
            )
            self.policy.decimation = 4  # Typical for hardware at 200Hz
            self.desired_qpos = np.array(
                [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5],
                dtype=np.float32,
            )
            self.sdk_to_isaac = list(range(12))  # Verified mapping should go here

        # 4. Teleop Subscription
        self.create_subscription(Twist, "/cmd_vel", self.teleop_cb, 10)
        self.cmds_vel = np.zeros(3)

        # 5. Initialization logic for SDK
        self._init_low_cmd()

        # 6. Control Loop (50Hz to match policy, SDK internal runs faster)
        self.create_timer(0.02, self.control_loop)
        self.get_logger().info(
            f"[RealDriver] Initialized for {robot}. Ready for deployment."
        )

    def _init_low_cmd(self):
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01
            self.low_cmd.motor_cmd[i].q = 2.146e9  # PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = 1.6e4  # VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def low_state_handler(self, msg: LowState_):
        self.low_state = msg

    def teleop_cb(self, msg):
        self.cmds_vel = np.array([msg.linear.x, msg.linear.y, msg.angular.z])

    def control_loop(self):
        """Internal inference logic."""
        if self.low_state is None:
            return

        state = self.telemetry.standardize(self.low_state, backend="generic")

        # Republish robot state to ROS2 for monitoring (rviz2, rqt, ros2 topic echo)
        self.telemetry.publish(sim_time=time.time(), state=state)

        if self.policy:
            if self.policy.should_step():
                actions, _ = self.policy.infer(
                    state, self.cmds_vel, self.desired_qpos, self.sdk_to_isaac
                )

                # Apply commands
                cmds = self.command_processor.process(actions, self.desired_qpos)
                self.send_to_sdk(cmds)

    def send_to_sdk(self, joint_targets):
        """Map ROS Type-Grouped targets to SDK2 motor commands."""
        # Generic map from TelemetryManager: [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
        # (This matches the order we used in sim_bridges)
        ros_to_sdk = [
            1,
            5,
            9,
            0,
            4,
            8,
            3,
            7,
            11,
            2,
            6,
            10,
        ]  # This needs to be carefully verified for SDK2

        # Re-using the logic from legacy bridge for now, but targeting motor_cmd
        sdk_indices = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
        for i, ros_idx in enumerate(sdk_indices):
            self.low_cmd.motor_cmd[i].q = float(joint_targets[ros_idx])
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = 45.0  # Typical Go2 gains
            self.low_cmd.motor_cmd[i].kd = 1.0
            self.low_cmd.motor_cmd[i].tau = 0.0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="go2")
    parser.add_argument("--internal_policy", type=str, default=None)
    parser.add_argument("--obs_dim", type=int, default=45)
    args = parser.parse_args()

    rclpy.init()
    node = RealDriver(args.robot, args.internal_policy, args.obs_dim)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
