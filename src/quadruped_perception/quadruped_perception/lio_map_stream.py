"""Point-LIO's map, small enough for Wi-Fi: occupied voxels of /lio/cloud, sent once each.

/lio/map_voxels: every occupied voxel's centre, transient_local, every --snapshot_period s. A receiver
replaces its map with it (a smaller one means the mapper restarted).
/lio/map_voxels/delta: only voxels occupied since the last delta, every --delta_period s. Added to the map.

Both are PointCloud2 with x, y, z float32 (12 bytes a point) in lio_odom. A voxel counts once --min_hits
scans have hit it, which drops one-off returns. A gap of --reset_gap s in /lio/cloud means Point-LIO
restarted in a new frame: the map starts over.
"""

import argparse
import array
import signal

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import PointCloud2, PointField

FIELDS = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")]
BITS = 21
BIAS = 1 << (BITS - 1)
MASK = (1 << BITS) - 1
MAX_CANDIDATES = 500_000


def keys_of(idx):
    """Voxel indices (n, 3) to one int64 each."""
    i = (idx + BIAS).astype(np.int64)
    return (i[:, 0] << (2 * BITS)) | (i[:, 1] << BITS) | i[:, 2]


def centres_of(keys, size):
    k = np.asarray(keys, np.int64)
    idx = np.column_stack([(k >> (2 * BITS)) & MASK, (k >> BITS) & MASK, k & MASK]) - BIAS
    return ((idx + 0.5) * size).astype(np.float32)


def xyz(msg):
    off = {f.name: f.offset for f in msg.fields}
    f4 = ">f4" if msg.is_bigendian else "<f4"
    pts = np.frombuffer(bytes(msg.data), np.dtype({"names": list("xyz"), "formats": [f4] * 3,
                                                   "offsets": [off[c] for c in "xyz"], "itemsize": msg.point_step}),
                        count=msg.width * msg.height)
    out = np.column_stack([pts[c] for c in "xyz"])
    return out[np.isfinite(out).all(axis=1)]


class LioMapStream(Node):
    def __init__(self, args):
        super().__init__("lio_map_stream")
        self.args = args
        self.occupied = set()
        self.candidates = {}
        self.pending = []
        self.frame = "lio_odom"
        self.last_cloud = None
        self.full_warned = False
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.snapshot_pub = self.create_publisher(PointCloud2, "/lio/map_voxels", latched)
        self.delta_pub = self.create_publisher(PointCloud2, "/lio/map_voxels/delta", 10)
        self.create_subscription(PointCloud2, "/lio/cloud", self._cloud_cb, 20)
        self.create_timer(args.delta_period, self._send_delta)
        self.create_timer(args.snapshot_period, self._send_snapshot)
        self.get_logger().info(f"/lio/cloud -> /lio/map_voxels(+/delta), {args.voxel:.2f} m voxels, "
                               f"{args.min_hits} hits, deltas every {args.delta_period:g} s, "
                               f"snapshots every {args.snapshot_period:g} s")

    def _cloud_cb(self, msg):
        now = self.get_clock().now()
        if self.last_cloud is not None and (now - self.last_cloud).nanoseconds * 1e-9 > self.args.reset_gap:
            self.get_logger().info(f"no /lio/cloud for {self.args.reset_gap:g} s: Point-LIO restarted, new map")
            self.occupied.clear()
            self.candidates.clear()
            self.pending.clear()
            self._send_snapshot()
        self.last_cloud = now
        self.frame = msg.header.frame_id or self.frame
        pts = xyz(msg)
        if not len(pts):
            return
        keys = np.unique(keys_of(np.floor(pts / self.args.voxel)))
        need = self.args.min_hits
        for k in keys.tolist():
            if k in self.occupied:
                continue
            hits = self.candidates.get(k, 0) + 1
            if hits >= need:
                self.candidates.pop(k, None)
                if len(self.occupied) >= self.args.max_voxels:
                    if not self.full_warned:
                        self.full_warned = True
                        self.get_logger().warn(f"map full ({self.args.max_voxels} voxels): no new voxels added")
                    continue
                self.occupied.add(k)
                self.pending.append(k)
            else:
                self.candidates[k] = hits
        if len(self.candidates) > MAX_CANDIDATES:
            self.candidates.clear()

    def _cloud(self, keys):
        pts = centres_of(keys, self.args.voxel) if keys else np.zeros((0, 3), np.float32)
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame
        msg.height, msg.width = 1, len(pts)
        msg.fields = FIELDS
        msg.point_step, msg.row_step = 12, 12 * len(pts)
        msg.is_dense = True
        data = array.array("B")
        data.frombytes(pts.tobytes())
        msg.data = data
        return msg

    def _send_delta(self):
        if self.pending:
            self.delta_pub.publish(self._cloud(self.pending))
            self.pending = []

    def _send_snapshot(self):
        self.snapshot_pub.publish(self._cloud(list(self.occupied)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--voxel", type=float, default=0.10)
    ap.add_argument("--min_hits", type=int, default=2)
    ap.add_argument("--delta_period", type=float, default=1.0)
    ap.add_argument("--snapshot_period", type=float, default=10.0)
    ap.add_argument("--reset_gap", type=float, default=5.0)
    ap.add_argument("--max_voxels", type=int, default=300_000)
    args = ap.parse_args(remove_ros_args()[1:])

    rclpy.init()
    node = LioMapStream(args)
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
