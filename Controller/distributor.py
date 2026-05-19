import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import JointState


class Distributor:
    """
    Publishes final joint commands to ROS 2 and optionally invokes a
    simulator/SDK callback.

    This is the single exit-point for motor commands leaving the pipeline.
    It does NOT make safety decisions — those are handled upstream by the
    CommandSafetyProcessor.
    """

    def __init__(self, node: Node, joint_names: list = None):
        self.node = node
        self.joint_names = joint_names or [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"
        ]

        self.cmd_pub = self.node.create_publisher(JointState, '/commands/joint_commands', 10)
        self.node.get_logger().info("[Distributor] Initialized.")

    def send(self, targets: np.ndarray, max_torque: float, send_to_robot_cb=None):
        """
        Publish final targets + torque to the ROS 2 topic, and optionally
        invoke a driver-specific callback (SDK write, sim actuator set, etc.).

        Args:
            targets:          12-dim array of joint position targets.
            max_torque:       Maximum torque (Nm) to apply.
            send_to_robot_cb: Optional callable(targets) for driver-specific output.
        """
        if send_to_robot_cb:
            send_to_robot_cb(targets)

        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = targets.tolist()
        msg.effort = [float(max_torque)] * len(self.joint_names)
        self.cmd_pub.publish(msg)
