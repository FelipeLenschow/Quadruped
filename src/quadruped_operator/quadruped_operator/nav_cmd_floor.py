"""Nav command floor - Nav2's velocity (/cmd_vel/nav_raw) to twist_mux (/cmd_vel/nav), never slower than the walking
policy moves.

Nav2's MPPI slows to a few cm/s and a few deg/s when it nears a goal or hesitates. The clock-gait policies keep
stepping at any command, so the robot then stepped in place: 18% of Nav2's commands in trial new_qwen3-8-27b_20 were
under 0.03 m/s forward or 0.1 rad/s turning (2026-11-19). MPPI has no minimum speed, and Nav2's velocity smoother has
only a deadband to zero, where the policy also steps in place.

A command whose largest part (forward over min_linear, sideways over min_linear, turning over min_angular) is under 1
is scaled up until that part reaches its minimum: all parts by the same factor, so the curve MPPI chose stays. Under
zero_below of the minimum it is zero (Nav2 holding still). Settings: config.yaml nav_cmd.
"""

import signal

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from quadruped_common.config_loader import load_config


def floor(vx, vy, wz, min_linear, min_angular, zero_below):
    """(vx, vy, wz) with its largest part, relative to its minimum, at least 1; zeros under zero_below."""
    r = max(abs(vx) / min_linear, abs(vy) / min_linear, abs(wz) / min_angular)
    if r < zero_below:
        return 0.0, 0.0, 0.0
    if r >= 1.0:
        return vx, vy, wz
    return vx / r, vy / r, wz / r


class NavCmdFloor(Node):
    def __init__(self, cfg):
        super().__init__("nav_cmd_floor")
        self.min_linear = float(cfg.get("min_linear", 0.15))
        self.min_angular = float(cfg.get("min_angular", 0.3))
        self.zero_below = float(cfg.get("zero_below", 0.1))
        self.pub = self.create_publisher(Twist, "/cmd_vel/nav", 10)
        self.create_subscription(Twist, "/cmd_vel/nav_raw", self._cb, 10)
        self.get_logger().info(f"/cmd_vel/nav_raw -> /cmd_vel/nav, at least {self.min_linear} m/s or "
                               f"{self.min_angular} rad/s, zero under {self.zero_below:.0%} of that")

    def _cb(self, msg):
        out = Twist()
        out.linear.x, out.linear.y, out.angular.z = floor(msg.linear.x, msg.linear.y, msg.angular.z,
                                                          self.min_linear, self.min_angular, self.zero_below)
        self.pub.publish(out)


def main():
    rclpy.init()
    node = NavCmdFloor(load_config().get("nav_cmd") or {})
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
