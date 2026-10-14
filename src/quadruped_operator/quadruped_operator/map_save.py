"""Map save - writes /map to maps/<name>.pgm and .yaml when a name arrives on /map/save.

The same files as nav2's map_saver_cli (trinary: free, occupied, unknown), so launcher [N] lists the map
and AMCL can localize on it. The answer goes out on /map/save/result: "saved <path>" or "error: <why>".
An existing map is never overwritten.
"""

import math
import os
import re
import signal

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from quadruped_common import paths

NAME = re.compile(r"[A-Za-z0-9_-]{1,40}")
# map_server reads a pixel as free below free_thresh: 0.196 keeps unknown's grey 205 ((255 - 205) / 255) unknown.
FREE_THRESH, OCCUPIED_THRESH = 0.196, 0.65
FREE, OCCUPIED, UNKNOWN = 254, 0, 205


def write_map(grid, base):
    """grid: OccupancyGrid. Writes base.pgm and base.yaml."""
    info = grid.info
    cells = np.asarray(grid.data, np.int8).reshape(info.height, info.width)
    image = np.full(cells.shape, UNKNOWN, np.uint8)
    image[(cells >= 0) & (cells <= 25)] = FREE
    image[cells >= OCCUPIED_THRESH * 100] = OCCUPIED
    with open(base + ".pgm", "wb") as f:
        f.write(f"P5\n{info.width} {info.height}\n255\n".encode())
        f.write(np.flipud(image).tobytes())
    q = info.origin.orientation
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    o = info.origin.position
    with open(base + ".yaml", "w") as f:
        f.write(f"image: {os.path.basename(base)}.pgm\nmode: trinary\nresolution: {info.resolution:.6g}\n"
                f"origin: [{o.x:.6g}, {o.y:.6g}, {yaw:.6g}]\nnegate: 0\n"
                f"occupied_thresh: {OCCUPIED_THRESH}\nfree_thresh: {FREE_THRESH}\n")


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
