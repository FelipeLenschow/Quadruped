"""Auto localize - once navigation starts on a saved map with an unknown pose, run the localize skill.

Waits for AMCL and the skill server, and for the operator to put the robot in policy mode. Retries a
failed localize, then exits.
"""

import argparse
import json

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from std_msgs.msg import String
from std_srvs.srv import Empty

from quadruped_interfaces.srv import Skill


class AutoLocalize(Node):
    def __init__(self):
        super().__init__("auto_localize")
        self.mode = None
        self.create_subscription(String, "/robot_state", self._state_cb, 10)
        self.skill = self.create_client(Skill, "/skill")
        self.amcl = self.create_client(Empty, "/reinitialize_global_localization")

    def _state_cb(self, msg):
        self.mode = json.loads(msg.data).get("mode")

    def call(self, name):
        future = self.skill.call_async(Skill.Request(name=name))
        rclpy.spin_until_future_complete(self, future)
        res = future.result()
        self.get_logger().info(f"{name}: {'ok' if res.success else 'failed'} - {res.message}")
        return res.success

    def wait(self, what, cond):
        self.get_logger().info(f"waiting for {what}...")
        while rclpy.ok() and not cond():
            rclpy.spin_once(self, timeout_sec=0.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--attempts", type=int, default=3)
    args = ap.parse_args(remove_ros_args()[1:])

    rclpy.init()
    node = AutoLocalize()
    try:
        node.wait("AMCL", lambda: node.amcl.service_is_ready())
        node.wait("the skill server", lambda: node.skill.service_is_ready())
        node.wait("/robot_state", lambda: node.mode is not None)
        node.wait("policy mode", lambda: node.mode == "policy")
        for _ in range(args.attempts):
            if node.call("localize"):
                break
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
