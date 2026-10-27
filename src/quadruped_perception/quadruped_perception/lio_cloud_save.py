"""A LIO's registered scans to one PCD file, written while mapping: the map without holding it in memory.

Subscribes /lio/cloud (remap it: FAST-LIO's /helios_lio/cloud), each scan already de-skewed and in the LIO's world
frame, and appends its x, y, z, intensity (float32) to maps/<name>_<date>_<time>.pcd. The header's point count is
rewritten every --header_period s and on exit, so the file is valid even if the node is killed. FAST-LIO's own save
on exit does nothing in its ROS 2 branch (the points are never collected), and holding a long run in memory would
not fit next to DLS2 on the Go2's Jetson anyway.

The scans are what the LIO matched: after its voxel filter (FAST-LIO's filter_size_surf), ~1.6k points each. For
every raw point, build the map from a recorded bag instead.
"""

import argparse
import os
import signal
from datetime import datetime

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import PointCloud2

from quadruped_common import paths

FLOAT32 = 7
COUNT_WIDTH = 12   # the header's counts are this many digits, so they can be rewritten in place


def header(points):
    return ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n"
            "TYPE F F F F\nCOUNT 1 1 1 1\n"
            f"WIDTH {points:0{COUNT_WIDTH}d}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
            f"POINTS {points:0{COUNT_WIDTH}d}\nDATA binary\n").encode()


def xyzi(msg):
    """The cloud's finite points as an (n, 4) float32 array: x, y, z, intensity (0 if it has none)."""
    fields = {f.name: f for f in msg.fields}
    if any(fields.get(n) is None or fields[n].datatype != FLOAT32 for n in "xyz") or msg.is_bigendian:
        return None
    names, offsets = ["x", "y", "z"], [fields[n].offset for n in "xyz"]
    has_intensity = "intensity" in fields and fields["intensity"].datatype == FLOAT32
    if has_intensity:
        names.append("i")
        offsets.append(fields["intensity"].offset)
    dtype = np.dtype({"names": names, "formats": ["<f4"] * len(names), "offsets": offsets,
                      "itemsize": msg.point_step})
    a = np.frombuffer(bytes(msg.data), dtype=dtype, count=msg.width * msg.height)
    out = np.empty((len(a), 4), np.float32)
    out[:, 0], out[:, 1], out[:, 2] = a["x"], a["y"], a["z"]
    out[:, 3] = a["i"] if has_intensity else 0.0
    return out[np.isfinite(out[:, :3]).all(axis=1)]


class LioCloudSave(Node):
    def __init__(self, args):
        super().__init__("lio_cloud_save")
        self.path = args.out or paths.repo("maps", f"{args.name}_{datetime.now():%Y%m%d_%H%M%S}.pcd")
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self.file = open(self.path, "wb")
        self.file.write(header(0))
        self.points = 0
        self.scans = 0
        self.frame = None
        self.written_points = -1
        self.warned_format = False
        # Reliable: a lost scan is a hole in the map
        reliable = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(PointCloud2, "/lio/cloud", self.on_cloud, reliable)
        self.create_timer(args.header_period, self.write_header)
        self.get_logger().info(f"/lio/cloud -> {self.path} (while mapping)")

    def on_cloud(self, msg):
        pts = xyzi(msg)
        if pts is None:
            if not self.warned_format:
                self.warned_format = True
                self.get_logger().error("/lio/cloud has no float32 x, y, z: not saved")
            return
        if self.frame is None:
            self.frame = msg.header.frame_id
            self.get_logger().info(f"first scan: {len(pts)} points, frame {self.frame}")
        elif msg.header.frame_id != self.frame:
            self.get_logger().warn(f"frame changed {self.frame} -> {msg.header.frame_id}: one file, two frames")
            self.frame = msg.header.frame_id
        self.file.seek(0, os.SEEK_END)
        self.file.write(pts.tobytes())
        self.points += len(pts)
        self.scans += 1

    def write_header(self):
        if self.file.closed or self.points == self.written_points:
            return
        self.file.seek(0)
        self.file.write(header(self.points))
        self.file.flush()
        os.fsync(self.file.fileno())
        self.written_points = self.points

    def close(self):
        if self.file.closed:
            return
        self.write_header()
        self.file.close()
        if self.points == 0:
            os.remove(self.path)
            print(f"[lio_cloud_save] no scans arrived: {self.path} not kept", flush=True)
        else:
            mb = os.path.getsize(self.path) / 1e6
            print(f"[lio_cloud_save] saved {self.points:,} points from {self.scans} scans to {self.path} ({mb:.0f} MB)",
                  flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", default="lio", help="file name prefix: maps/<name>_<date>_<time>.pcd")
    ap.add_argument("--out", default="", help="the file instead")
    ap.add_argument("--header_period", type=float, default=5.0, help="s between point count updates in the file")
    args = ap.parse_args(remove_ros_args()[1:])
    rclpy.init()
    node = LioCloudSave(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
