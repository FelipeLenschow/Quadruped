"""Map save - writes /map to maps/<name>.pgm and .yaml when a name arrives on /map/save.

The same files as nav2's map_saver_cli (trinary: free, occupied, unknown), so launcher [N] lists the map
and AMCL can localize on it. The answer goes out on /map/save/result: "saved <path>" or "error: <why>".
An existing map is never overwritten.
"""

import os
import re
import signal

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from quadruped_common import paths
from quadruped_common.map_files import write_map

NAME = re.compile(r"[A-Za-z0-9_-]{1,40}")


class MapSave(Node):
    def __init__(self):
        super().__init__("map_save")
        self.grid = None
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, "/map", self._map_cb, latched)
        self.create_subscription(String, "/map/save", self._save_cb, 10)
        self.result = self.create_publisher(String, "/map/save/result", 10)
        self.get_logger().info(f"/map/save <name> -> {paths.repo('maps')}/<name>.pgm/.yaml")

    def _map_cb(self, msg):
        self.grid = msg

    def _save_cb(self, msg):
        name = msg.data.strip()
        base = paths.repo("maps", name)
        if not NAME.fullmatch(name):
            reply = "error: name must be 1-40 letters, digits, _ or -"
        elif self.grid is None:
            reply = "error: no /map yet"
        elif os.path.exists(base + ".yaml"):
            reply = f"error: maps/{name} exists; pick another name"
        else:
            try:
                os.makedirs(os.path.dirname(base), exist_ok=True)
                write_map(self.grid, base)
                info = self.grid.info
                reply = (f"saved maps/{name} ({info.width}x{info.height} @ {info.resolution:.2f} m)")
            except OSError as e:
                reply = f"error: {e}"
        if reply.startswith("saved"):
            self.get_logger().info(reply)
        else:
            self.get_logger().warn(reply)
        self.result.publish(String(data=reply))


def main():
    rclpy.init()
    node = MapSave()
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
