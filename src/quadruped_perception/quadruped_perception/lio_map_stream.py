"""Point-LIO's map, small enough for Wi-Fi: occupied voxels of /lio/cloud, with what appears and disappears.

/lio/map_voxels: transient_local. A receiver replaces its map with it. Sent when Point-LIO restarts, when a
new receiver subscribes, and every --snapshot_period s if set.
/lio/map_voxels/delta: changes, every --delta_period s. Each message is a MARK_REMOVE point, the voxels to
remove, a MARK_ADD point, the voxels to add. Both marks have x = NaN; y is 0 (remove) or 1 (add).
/lio/map_voxels/size: the voxel edge in m (Float32, transient_local), for drawing and keying them.

The clouds are PointCloud2 with x, y, z float32 (12 bytes a point) in lio_odom: voxel centres.

Each voxel keeps a hit/miss score, as OctoMap does: a point in it is a hit, a beam from the lidar passing
through it a miss. It shows once its score reaches --min_hits hits, and goes again when misses bring it back
down: an opened door, someone who walked by, or the L1's range noise thickening surfaces over minutes. A gap
of --reset_gap s in /lio/cloud means Point-LIO restarted in a new frame: the map starts over.

Nothing goes out faster than --max_send voxels per --delta_period: a whole map in one message held the Wi-Fi
for seconds and starved the safety heartbeat. A snapshot carries the voxels nearest the robot, and the rest
follow in the deltas.

The L1's floor is a few cm thick (more at grazing range), so voxels would stack it 2-3 deep, and grazing beams
would eat into it. Points within --ground_band of the floor under the robot (from /lio/odom and the nearby
cloud) are kept as a height map instead: one voxel per column, at the column's mean height. Lower obstacles
merge into it. A column moves to another voxel only once its mean is well past the edge.
"""

import argparse
import array
import math
import signal
from collections import deque

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.utilities import remove_ros_args
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float32

from quadruped_core.config_loader import load_config

FIELDS = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")]
BITS = 21
BIAS = 1 << (BITS - 1)
MASK = (1 << BITS) - 1
FLOOR_RADIUS = 1.0
FLOOR_BELOW = (0.15, 1.0)
FLOOR_MIN_POINTS = 20
ODOM_MATCH_S = 0.2
SUBSCRIBER_POLL_S = 0.2
FLOOR_KEEP = 0.8  # a floor column moves to another voxel once its mean is this many voxels from the centre
MARK_REMOVE = (math.nan, 0.0, 0.0)
MARK_ADD = (math.nan, 1.0, 0.0)
# Log-odds in steps of 0.05. Hit 0.7 as OctoMap; miss 0.45, half its weight, so beams grazing low or thin things
# seen from afar do not wear them away; clamped to 0.12 .. 0.92, so an opened door clears in ~5 s, not ~20.
HIT, MISS, LOW, HIGH = 17, -4, -40, 50
BLOCK = 4  # blocks of 16^3 voxels


def keys_of(idx):
    """Voxel indices (n, 3) to one int64 each."""
    i = (idx + BIAS).astype(np.int64)
    return (i[:, 0] << (2 * BITS)) | (i[:, 1] << BITS) | i[:, 2]


def indices_of(keys):
    k = np.asarray(keys, np.int64)
    return np.column_stack([(k >> (2 * BITS)) & MASK, (k >> BITS) & MASK, k & MASK]) - BIAS


def centres_of(keys, size):
    return ((indices_of(keys) + 0.5) * size).astype(np.float32)


def unique_keys(pts, size):
    return np.unique(keys_of(np.floor(pts / size).astype(np.int64)))


def stamp_s(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def xyz(msg):
    off = {f.name: f.offset for f in msg.fields}
    f4 = ">f4" if msg.is_bigendian else "<f4"
    pts = np.frombuffer(bytes(msg.data), np.dtype({"names": list("xyz"), "formats": [f4] * 3,
                                                   "offsets": [off[c] for c in "xyz"], "itemsize": msg.point_step}),
                        count=msg.width * msg.height)
    out = np.column_stack([pts[c] for c in "xyz"])
    return out[np.isfinite(out).all(axis=1)]


def rotate(q, v):
    x, y, z, w = q
    u = np.array([x, y, z])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


class Occupancy:
    """Hit/miss scores per voxel, int8, in blocks of 16^3 made on a block's first hit. Beams through space
    no point ever landed in change nothing, so memory follows the surfaces, not the air."""

    def __init__(self, threshold):
        self.threshold = threshold
        self.rows = {}
        self.store = np.zeros((256, 1 << (3 * BLOCK)), np.int8)

    def _rows(self, idx, create):
        bkeys = keys_of(idx >> BLOCK)
        unique, inv = np.unique(bkeys, return_inverse=True)
        rows = np.empty(len(unique), np.int64)
        for i, k in enumerate(unique.tolist()):
            r = self.rows.get(k, -1)
            if r < 0 and create:
                r = len(self.rows)
                if r == len(self.store):
                    self.store = np.concatenate([self.store, np.zeros_like(self.store)])
                self.rows[k] = r
            rows[i] = r
        return rows[inv.ravel()]

    def score(self, keys):
        rows = self._rows(indices_of(keys), False)
        out = np.zeros(len(keys), np.int16)
        known = rows >= 0
        out[known] = self.store.reshape(-1)[self._flat(rows[known], indices_of(keys[known]))]
        return out

    @staticmethod
    def _flat(rows, idx):
        local = idx & ((1 << BLOCK) - 1)
        return rows * (1 << (3 * BLOCK)) + (local[:, 0] << (2 * BLOCK)) + (local[:, 1] << BLOCK) + local[:, 2]

    def update(self, hits, misses):
        """Voxel keys hit and passed through, each listed once. Returns the keys that became occupied and
        those that stopped being."""
        hit_idx, miss_idx = indices_of(hits), indices_of(misses)
        keys = np.concatenate([hits, misses])
        idx = np.concatenate([hit_idx, miss_idx])
        step = np.concatenate([np.full(len(hit_idx), HIT, np.int16), np.full(len(miss_idx), MISS, np.int16)])
        rows = np.concatenate([self._rows(hit_idx, True), self._rows(miss_idx, False)])
        known = rows >= 0
        keys, idx, step, rows = keys[known], idx[known], step[known], rows[known]
        flat = self._flat(rows, idx)
        cells = self.store.reshape(-1)
        before = cells[flat].astype(np.int16)
        after = np.clip(before + step, LOW, HIGH)
        cells[flat] = after
        was, now = before >= self.threshold, after >= self.threshold
        return keys[now & ~was], keys[was & ~now]


class VoxelMap:
    """The map lio_map_stream sends: the occupancy grid plus the floor height map, and which voxels changed."""

    def __init__(self, voxel, min_hits=2, ground_band=0.10, max_voxels=300_000, max_range=10.0, sensor=(0, 0, 0)):
        self.voxel, self.min_hits, self.ground_band = voxel, min_hits, ground_band
        self.max_voxels, self.max_range = max_voxels, max_range
        self.sensor = np.asarray(sensor, float)
        self.full = False
        self.clear()

    def clear(self):
        self.grid = Occupancy(HIT * self.min_hits + MISS)
        self.occupied = set()
        self.ground = {}
        self.ground_voxel = {}
        self.ground_keys = set()
        self.changed = set()
        self.floor = None

    def size(self):
        return len(self.occupied) + len(self.ground_keys)

    def is_occupied(self, k):
        return k in self.occupied or k in self.ground_keys

    def all_keys(self):
        return self.occupied | self.ground_keys

    def insert(self, pts, pose):
        """One registered scan (n, 3) in lio_odom, and the base pose then: (x, y, z, qx, qy, qz, qw), or None."""
        floor, rest = self._split_floor(pts, pose)
        if len(floor):
            self._add_floor(floor)
        v = self.voxel
        hit_pts = rest
        misses = np.zeros(0, np.int64)
        if pose is not None:
            origin = np.asarray(pose[:3], float) + rotate(pose[3:], self.sensor)
            d = pts - origin
            length = np.linalg.norm(d, axis=1)
            hit_pts = rest[np.linalg.norm(rest - origin, axis=1) <= self.max_range]
            step = 0.5 * v
            n = np.floor(np.minimum(length, self.max_range) / step).astype(np.int64)
            ray = np.repeat(np.arange(len(pts)), n)
            i = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
            samples = origin + d[ray] / length[ray, None] * ((i + 0.5) * step)[:, None]
            misses = unique_keys(samples, v)
            hits = self._hits_on_surfaces(hit_pts, origin)
        else:
            hits = unique_keys(hit_pts, v)
        misses = np.setdiff1d(misses, hits, assume_unique=True)  # a beam's own end voxel included
        appeared, gone = self.grid.update(hits, misses)
        for k in gone.tolist():
            if k in self.occupied:
                self.occupied.discard(k)
                self.changed.add(k)
        for k in appeared.tolist():
            if self._room():
                self.occupied.add(k)
                self.changed.add(k)

    def _hits_on_surfaces(self, pts, origin):
        """Range noise puts some returns a voxel behind the surface they came from, where no beam ever passes
        to clear them: a hit right behind an occupied voxel on its own beam counts for that voxel."""
        v = self.voxel
        keys = keys_of(np.floor(pts / v).astype(np.int64))
        d = pts - origin
        back = keys_of(np.floor((pts - d / np.linalg.norm(d, axis=1)[:, None] * v) / v).astype(np.int64))
        snap = (back != keys) & (self.grid.score(back) >= self.grid.threshold)
        return np.unique(np.where(snap, back, keys))

    def _room(self):
        if self.size() < self.max_voxels:
            return True
        self.full = True
        return False

    def _split_floor(self, pts, pose):
        """(points near the floor under the robot, the rest)."""
        if pose is None or self.ground_band <= 0:
            return pts[:0], pts
        x, y, z = pose[:3]
        dz = z - pts[:, 2]
        near = (np.hypot(pts[:, 0] - x, pts[:, 1] - y) < FLOOR_RADIUS) & (dz > FLOOR_BELOW[0]) & (dz < FLOOR_BELOW[1])
        if near.sum() >= FLOOR_MIN_POINTS:
            self.floor = float(np.median(pts[near, 2])) - z
        if self.floor is None:
            return pts[:0], pts
        on = np.abs(pts[:, 2] - (z + self.floor)) < self.ground_band
        return pts[on], pts[~on]

    def _add_floor(self, pts):
        v = self.voxel
        cols = np.floor(pts[:, :2] / v).astype(np.int64)
        keys, first, inv = np.unique(keys_of(np.column_stack([cols, np.zeros(len(cols), np.int64)])),
                                     return_index=True, return_inverse=True)
        sums = np.bincount(inv.ravel(), weights=pts[:, 2])
        counts = np.bincount(inv.ravel())
        for k, s, n, i in zip(keys.tolist(), sums.tolist(), counts.tolist(), first.tolist()):
            total = self.ground.get(k, (0.0, 0))
            total = (total[0] + s, total[1] + n)
            self.ground[k] = total
            if total[1] < self.min_hits:
                continue
            mean = total[0] / total[1]
            old = self.ground_voxel.get(k)
            if old is not None and abs(mean - ((old & MASK) - BIAS + 0.5) * v) < FLOOR_KEEP * v:
                continue
            key = (k & ~MASK) | (math.floor(mean / v) + BIAS)
            if old == key or (old is None and not self._room()):
                continue
            if old is not None:
                self.ground_keys.discard(old)
                self.changed.add(old)
            self.ground_voxel[k] = key
            self.ground_keys.add(key)
            self.changed.add(key)


class LioMapStream(Node):
    def __init__(self, args, sensor):
        super().__init__("lio_map_stream")
        self.args = args
        self.map = VoxelMap(args.voxel, args.min_hits, args.ground_band, args.max_voxels, args.max_range, sensor)
        self.shown = set()
        self.poses = deque(maxlen=100)
        self.frame = "lio_odom"
        self.last_cloud = None
        self.full_warned = False
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.snapshot_pub = self.create_publisher(PointCloud2, "/lio/map_voxels", latched)
        self.delta_pub = self.create_publisher(PointCloud2, "/lio/map_voxels/delta", 10)
        self.size_pub = self.create_publisher(Float32, "/lio/map_voxels/size", latched)
        self.size_pub.publish(Float32(data=float(args.voxel)))
        self.create_subscription(PointCloud2, "/lio/cloud", self._cloud_cb, 20)
        self.create_subscription(Odometry, "/lio/odom", self._odom_cb, 50)
        self.subscribers = 0
        self.create_timer(args.delta_period, self._send_delta)
        self.create_timer(SUBSCRIBER_POLL_S, self._watch_subscribers)
        if args.snapshot_period > 0:
            self.create_timer(args.snapshot_period, self._send_snapshot)
        self.get_logger().info(f"/lio/cloud -> /lio/map_voxels(+/delta), {args.voxel:.2f} m voxels, "
                               f"{args.min_hits} hits, beams clear to {args.max_range:g} m, deltas every "
                               f"{args.delta_period:g} s, at most {args.max_send} voxels each, "
                               f"floor band {args.ground_band:g} m")

    def _odom_cb(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.poses.append((stamp_s(msg.header.stamp), (p.x, p.y, p.z, q.x, q.y, q.z, q.w)))

    def _pose_at(self, t):
        if not self.poses:
            return None
        best = min(self.poses, key=lambda q: abs(q[0] - t))
        return best[1] if abs(best[0] - t) < ODOM_MATCH_S else None

    def _cloud_cb(self, msg):
        now = self.get_clock().now()
        if self.last_cloud is not None and (now - self.last_cloud).nanoseconds * 1e-9 > self.args.reset_gap:
            self.get_logger().info(f"no /lio/cloud for {self.args.reset_gap:g} s: Point-LIO restarted, new map")
            self.map.clear()
            self.shown.clear()
            self._send_snapshot()
        self.last_cloud = now
        self.frame = msg.header.frame_id or self.frame
        pts = xyz(msg)
        if len(pts):
            self.map.insert(pts, self._pose_at(stamp_s(msg.header.stamp)))
        if self.map.full and not self.full_warned:
            self.full_warned = True
            self.get_logger().warn(f"map full ({self.args.max_voxels} voxels): no new voxels added")

    def _cloud(self, keys, removed=None):
        pts = centres_of(keys, self.args.voxel) if len(keys) else np.zeros((0, 3), np.float32)
        if removed is not None:
            gone = centres_of(removed, self.args.voxel) if len(removed) else np.zeros((0, 3), np.float32)
            pts = np.vstack([np.array([MARK_REMOVE], np.float32), gone, np.array([MARK_ADD], np.float32), pts])
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame
        msg.height, msg.width = 1, len(pts)
        msg.fields = FIELDS
        msg.point_step, msg.row_step = 12, 12 * len(pts)
        msg.is_dense = removed is None
        data = array.array("B")
        data.frombytes(pts.tobytes())
        msg.data = data
        return msg

    def _nearest_first(self, keys):
        keys = np.asarray(keys, np.int64)
        if len(keys) and self.poses:
            x, y = self.poses[-1][1][:2]
            c = centres_of(keys, self.args.voxel)
            keys = keys[np.argsort(np.hypot(c[:, 0] - x, c[:, 1] - y))]
        return keys

    def _send_delta(self):
        changed = self.map.changed
        if not changed:
            return
        keys = np.fromiter(changed, np.int64, len(changed))
        if len(keys) > self.args.max_send:
            keys = self._nearest_first(keys)[:self.args.max_send]
        added, removed = [], []
        for k in keys.tolist():
            changed.discard(k)
            if self.map.is_occupied(k):
                if k not in self.shown:
                    self.shown.add(k)
                    added.append(k)
            elif k in self.shown:
                self.shown.discard(k)
                removed.append(k)
        if added or removed:
            self.delta_pub.publish(self._cloud(added, removed))

    def _send_snapshot(self):
        keys = self._nearest_first(np.fromiter(self.map.all_keys(), np.int64))
        n = self.args.max_send
        self.snapshot_pub.publish(self._cloud(keys[:n]))
        self.shown = set(keys[:n].tolist())
        self.map.changed.update(keys[n:].tolist())

    def _watch_subscribers(self):
        n = self.snapshot_pub.get_subscription_count()
        if n > self.subscribers:
            self._send_snapshot()
        self.subscribers = n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--min_hits", type=int, default=2)
    ap.add_argument("--max_range", type=float, default=10.0, help="m; beams clear voxels and add points up to this")
    ap.add_argument("--delta_period", type=float, default=1.0)
    ap.add_argument("--snapshot_period", type=float, default=0.0, help="s; 0: only on restart or a new receiver")
    ap.add_argument("--max_send", type=int, default=8000, help="voxels per message (12 bytes each)")
    ap.add_argument("--reset_gap", type=float, default=5.0)
    ap.add_argument("--max_voxels", type=int, default=300_000)
    ap.add_argument("--ground_band", type=float, default=0.10, help="m around the floor kept as a height map; 0: off")
    args = ap.parse_args(remove_ros_args()[1:])
    # Beams start at the lidar: lio_imu is the base, and the lidar sits at real_lidar.xyz on it.
    sensor = (load_config().get("real_lidar") or {}).get("xyz", (0.0, 0.0, 0.0))

    rclpy.init()
    node = LioMapStream(args, sensor)
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
