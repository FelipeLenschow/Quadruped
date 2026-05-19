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

from pipeline import LocomotionPipeline

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
    def __init__(self, robot="go2", internal_policy=None, obs_dim=45, interface=None):
        super().__init__("real_driver")
        self.robot_type = robot

        # 1. SDK2 Initialization
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.low_state_handler, 10)

        self.low_state = None
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.crc = CRC()

        # 2. Locomotion Pipeline
        try:
            self.pipeline = LocomotionPipeline(
                node=self,
                robot_type=robot,
                checkpoint=internal_policy,
                obs_dim=obs_dim,
                use_estimator=True  # Usually True on physical hardware
            )
        except ImportError:
            self.get_logger().error("[RealDriver] PyTorch not found. Internal policy disabled. Running in TELEMETRY ONLY mode.")
            self.pipeline = LocomotionPipeline(
                node=self,
                robot_type=robot,
                checkpoint=None,
                obs_dim=obs_dim,
                use_estimator=True
            )

        # 4. Teleop Subscription
        self.create_subscription(Twist, "/cmd_vel", self.teleop_cb, 10)
        self.cmds_vel = np.zeros(3)

        # 5. Initialization logic for SDK
        self._init_low_cmd()

        # 6. Control Loop (200Hz to match simulation)
        self.create_timer(0.005, self.control_loop)
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

    def _get_raw_sensor_data(self):
        """Standardizes LowState into raw vectors for the TelemetryManager."""
        raw = self.low_state
        q = [float(raw.motor_state[i].q) for i in range(12)]
        dq = [float(raw.motor_state[i].dq) for i in range(12)]
        quat = raw.imu_state.quaternion   # [w, x, y, z]
        gyro = raw.imu_state.gyroscope    # body frame
        accel = raw.imu_state.accelerometer # body frame

        # Foot contact from FSR sensors (int16)
        # Unitree LowState foot_force order: [FR, FL, RR, RL] -> reorder to [FL, FR, RL, RR]
        contact = [0.0, 0.0, 0.0, 0.0]
        if hasattr(raw, 'foot_force'):
            ff = raw.foot_force
            thr = 50
            contact = [
                float(ff[1] > thr),  # FL
                float(ff[0] > thr),  # FR
                float(ff[3] > thr),  # RL
                float(ff[2] > thr),  # RR
            ]

        return {
            'q': q, 'dq': dq, 'quat': quat, 'gyro': gyro, 'accel': accel, 'contact': contact
        }

    def control_loop(self):
        """Internal inference logic."""
        if self.low_state is None:
            return

        raw_data = self._get_raw_sensor_data()

        cmds = self.pipeline.step(
            raw_state_kwargs=raw_data,
            cmd_vel=self.cmds_vel,
            sim_time=time.time()
        )
        
        if "main" in self.pipeline.policy_manager.policies:
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
        max_torque = self.pipeline.safety_processor.active_max_torque
        
        for i, ros_idx in enumerate(sdk_indices):
            self.low_cmd.motor_cmd[i].q = float(joint_targets[ros_idx])
            self.low_cmd.motor_cmd[i].dq = 0.0
            
            if max_torque <= 0.1:
                self.low_cmd.motor_cmd[i].kp = 0.0
                self.low_cmd.motor_cmd[i].kd = 0.0
                self.low_cmd.motor_cmd[i].tau = 0.0
            else:
                self.low_cmd.motor_cmd[i].kp = 45.0  # Typical Go2 gains
                self.low_cmd.motor_cmd[i].kd = 1.0
                self.low_cmd.motor_cmd[i].tau = 0.0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="go2")
    parser.add_argument("--interface", type=str, default=None)
    parser.add_argument("--internal_policy", type=str, default=None)
    parser.add_argument("--obs_dim", type=int, default=45)
    args = parser.parse_args()

    # 1. SDK2 Initialization (Must happen BEFORE rclpy.init to claim the DDS domain)
    # Clear ROS 2 config to prevent conflicts with SDK's internal XML
    os.environ.pop("CYCLONEDDS_URI", None)
    try:
        ChannelFactoryInitialize(0, networkInterface=args.interface)
    except Exception as e:
        print(f"[SDK2] Failed to initialize ChannelFactory: {e}")
        sys.exit(1)

    # 2. ROS 2 Initialization (Use a different Domain ID to avoid conflict with SDK)
    os.environ["ROS_DOMAIN_ID"] = "1"
    rclpy.init()
    node = RealDriver(args.robot, args.internal_policy, args.obs_dim, interface=args.interface)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
