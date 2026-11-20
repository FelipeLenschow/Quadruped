"""Map open - SLAM's /map with the space ahead that the lidar sees open, on /map_open (the app's and mcp_server's map).

A beam with no return is inf in /scan, and slam_toolbox (Karto) skips it: in a long corridor the L1 saw both walls but
nothing ahead, and /map kept an unknown V in front of the robot down the corridor; the LLM agent's go_to_place then
walked 1 m at a time, "as far as the map goes" (trial new_qwen3-8-27b-think_3, 2026-11-19). Here, for each /scan, the
unknown cells along each beam ahead (open_ahead_deg each side) with no return within open_ahead_m, and none within
3 deg of it (the L1 fills under half the beams of one scan: a missing beam beside an obstacle is not open), are kept
as free, never past a wall SLAM saw. Before 2026-11-20 only mcp_server did this, for the agent's tools, and the app
still showed the V. Nav2's global costmap reads /map_open too (nav.launch.py); /map itself (map_save) is
unchanged. Settings: config.yaml map_open.
"""

import array
import math
import signal

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener

from quadruped_common.config_loader import load_config


def open_beams(ranges, angle_min, angle_increment, range_min, reach, half_rad, spread_rad=math.radians(3.0)):
    """Angles (rad, in the scan's frame) of the beams within half_rad of ahead with no return under reach, nor any
    within spread_rad of them."""
    r = np.asarray(ranges, float)
    a = angle_min + np.arange(len(r)) * angle_increment
    a = (a + math.pi) % (2 * math.pi) - math.pi
    near = np.isfinite(r) & (r >= range_min) & (r < reach)
    spread = max(1, int(round(spread_rad / abs(angle_increment))))
    near = np.convolve(np.r_[near[-spread:], near, near[:spread]], np.ones(2 * spread + 1), "valid") > 0
    return a[(np.abs(a) <= half_rad) & ~near]


def walls_of(grid):
    """The grid's walls (>= 65), one cell thicker each way, flat: a beam slipped through the corner gaps of a
    diagonal wall one cell thick, and filled the far side of the lab wall (new_qwen3-8-27b-think_16)."""
    info = grid.info
    walls = (np.asarray(grid.data, np.int8) >= 65).astype(np.uint8).reshape(info.height, info.width)
    return cv2.dilate(walls, np.ones((3, 3), np.uint8)).ravel().astype(bool)


def open_cells(grid, x0, y0, yaw, beams, reach, walls=None):
    """World cell keys (floor(x / res), floor(y / res)) of the unknown cells along the beams from (x0, y0) facing
    yaw, out to reach, each beam ending at the first wall (walls_of(grid), or given)."""
    info = grid.info
    w, h, res = info.width, info.height, info.resolution
    ox, oy = info.origin.position.x, info.origin.position.y
    if walls is None:
        walls = walls_of(grid)
    # From the robot itself, not 0.3 m out: at the lab door the wall was nearer, and the lidar sees nothing that
    # near, so those beams had no return and went on through it (_16).
    d = np.arange(res / 2, reach, res / 2)
    ang = yaw + np.asarray(beams)[:, None]
    x, y = x0 + d[None, :] * np.cos(ang), y0 + d[None, :] * np.sin(ang)
    i, j = np.floor((x - ox) / res).astype(int), np.floor((y - oy) / res).astype(int)
    inside = (i >= 0) & (i < w) & (j >= 0) & (j < h)
    cells = np.asarray(grid.data, np.int8)
    k = np.clip(j, 0, h - 1) * w + np.clip(i, 0, w - 1)
    v = np.where(inside, cells[k], -1)
    wall = inside & walls[k]
    keep = (np.cumsum(wall, axis=1) == 0) & (v < 0)
    return set(zip(np.floor(x[keep] / res).astype(int).tolist(), np.floor(y[keep] / res).astype(int).tolist()))


def with_open(grid, keys):
    """grid's cells (int8 array) with the unknown ones among keys made free."""
    info = grid.info
    w, h, res = info.width, info.height, info.resolution
    ox, oy = info.origin.position.x, info.origin.position.y
    cells = np.array(grid.data, np.int8)
    if keys:
        k = np.array(list(keys), float)
        i = np.floor(((k[:, 0] + 0.5) * res - ox) / res).astype(int)  # each key's cell centre
        j = np.floor(((k[:, 1] + 0.5) * res - oy) / res).astype(int)
        ok = (i >= 0) & (i < w) & (j >= 0) & (j < h)
        idx = j[ok] * w + i[ok]
        cells[idx[cells[idx] < 0]] = 0
    return cells


class MapOpen(Node):
    def __init__(self, cfg):
        super().__init__("map_open")
        self.reach = float(cfg.get("open_ahead_m", 6.0))
        self.half = math.radians(float(cfg.get("open_ahead_deg", 60.0)))
        self.period = float(cfg.get("publish_s", 1.0))
        self.grid, self.walls, self.keys, self.dirty = None, None, set(), False
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(OccupancyGrid, "/map_open", latched)
        self.create_subscription(OccupancyGrid, "/map", self._map_cb, latched)
        self.create_subscription(LaserScan, "/scan", self._scan_cb, qos_profile_sensor_data)
        self.create_timer(self.period, self._publish)
        self.get_logger().info(f"/map + open ahead (no return within {self.reach:g} m, "
                               f"+-{math.degrees(self.half):.0f} deg) -> /map_open")

    def _map_cb(self, msg):
        self.grid, self.walls, self.dirty = msg, walls_of(msg), True

    def _scan_cb(self, msg):
        if self.grid is None:
            return
        beams = open_beams(msg.ranges, msg.angle_min, msg.angle_increment, msg.range_min, self.reach, self.half)
        if not len(beams):
            return
        for when in (Time.from_msg(msg.header.stamp), Time()):
            try:
                tf = self.tf_buffer.lookup_transform(self.grid.header.frame_id or "map", msg.header.frame_id,
                                                     when).transform
                break
            except TransformException:
                tf = None
        if tf is None:
            return
        q = tf.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        new = open_cells(self.grid, tf.translation.x, tf.translation.y, yaw, beams, self.reach, self.walls) - self.keys
        if new:
            self.keys |= new
            self.dirty = True

    def _publish(self):
        if not self.dirty or self.grid is None:
            return
        self.dirty = False
        out = OccupancyGrid()
        out.header, out.info = self.grid.header, self.grid.info
        out.data = array.array("b", with_open(self.grid, self.keys).tobytes())
        self.pub.publish(out)


def main():
    rclpy.init()
    node = MapOpen(load_config().get("map_open") or {})
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
