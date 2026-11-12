"""Lidar fan: the lidar cloud as the ranges a lidar policy (ClockScan) was trained on, on /lidar/fan_ranges.

The policy's fan (from its .qpol) is rays from a fixed point in the base frame, rows by depression and columns by
azimuth, turning with the whole base. Points stay in the base frame: each /odom/state_estimator message moves the
kept ones by the base's motion since the last (the IMU's rotation change and the estimator's velocity, with its
vertical part; the odom TF's height is the base above the stance feet, which does not follow a climb). At rate_hz
the new points and each ray's KEEP nearest (while younger than window_s) are binned into each ray's cone (half the
gap to the next ray each way). A cone with any new point drops its older ones, and the ray takes the shortest
range in its cone, as a ray's first hit (a point carried into the next cone by the motion is one of its). A ray with no point reads the surface of the nearest rays that
have one: their mean height below the fan, level in the world (the IMU's attitude), along this ray; a ray that does
not go down in the world reads max_range, as the training rays that hit nothing. With no points at all a ray reads the range to
flat ground with the fan prior_height above it (an empty ray read as max_range stopped the policy climbing in
training). Points whose beam passed a leg are dropped, so a bin a leg hides is filled the same way. Each point is moved from
its own time (the cloud's 'time' field: the L1 measures a cloud over 65 ms).
Nothing is published while the cloud or the motion is older than cloud_timeout_s, so the controller stops the policy.
"""

import argparse
from collections import deque
import os
import signal
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from rclpy.time import Time
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import JointState, PointCloud2
from std_msgs.msg import Float32MultiArray
from tf2_ros import Buffer, TransformException, TransformListener

from quadruped_common import paths
from quadruped_common.config_loader import load_config, load_policies
from quadruped_perception.lidar_filter import leg_links, leg_shadow, rotation

def _edges(centers, circular):
    """Bin edges around sorted centers: midpoints, the outer ones half a gap out (or wrapped, if circular)."""
    mid = 0.5 * (centers[1:] + centers[:-1])
    if circular:
        lo = 0.5 * (centers[0] + centers[-1] - 360.0)
        return np.concatenate([[lo], mid, [lo + 360.0]])
    return np.concatenate([[1.5 * centers[0] - 0.5 * centers[1]], mid, [1.5 * centers[-1] - 0.5 * centers[-2]]])


class FanBins:
    """Points in the base frame -> one range per fan ray, flat index row * len(azimuths) + column."""

    def __init__(self, pos, depressions, azimuths, max_range, prior_height):
        self.pos = np.asarray(pos, dtype=np.float64)
        self.max_range = float(max_range)
        dep = np.asarray(depressions, dtype=np.float64)
        az = np.asarray(azimuths, dtype=np.float64)
        self.n_az = len(az)
        self.dep_order = np.argsort(dep)
        self.dep = dep[self.dep_order]
        self.dep_edges = _edges(self.dep, False)
        self.az_order = np.argsort(az)
        self.az = az[self.az_order]
        gap = self.az[1] - self.az[0] if len(az) > 1 else 360.0
        self.circular = abs(self.az[-1] - self.az[0] + gap - 360.0) < 1e-3
        self.az_edges = _edges(self.az, self.circular)
        self.dep_rank, self.az_rank = np.argsort(self.dep_order), np.argsort(self.az_order)
        d, a = np.meshgrid(np.radians(dep), np.radians(az), indexing="ij")
        self.dirs = np.stack([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), -np.sin(d)], -1).reshape(-1, 3)
        self.prior = np.minimum(prior_height / np.maximum(-self.dirs[:, 2], 1e-3), self.max_range).astype(np.float32)

    def update(self, pts, keep=1, new=None):
        """Range per ray (NaN with no point): the shortest in its cone, where a cone with any new point (mask over
        pts) drops its older ones. picked, picked_rays: each ray's point; kept: up to keep nearest per ray."""
        self.picked = self.picked_rays = self.kept = np.zeros(0, dtype=np.int64)
        out = np.full(len(self.prior), np.nan, dtype=np.float32)
        v = pts - self.pos
        r = np.linalg.norm(v, axis=1)
        idx = np.flatnonzero(r > 1e-3)
        v, r = v[idx], r[idx]
        dep = np.degrees(np.arcsin(np.clip(-v[:, 2] / r, -1.0, 1.0)))
        az = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
        if self.circular:
            az = (az - self.az_edges[0]) % 360.0 + self.az_edges[0]
        i = np.searchsorted(self.dep_edges, dep) - 1
        j = np.searchsorted(self.az_edges, az) - 1
        ok = (i >= 0) & (i < len(self.dep)) & (j >= 0) & (j < len(self.az))
        idx, r, ray = idx[ok], r[ok], self.dep_order[i[ok]] * self.n_az + self.az_order[j[ok]]
        if new is not None:
            fresh = new[idx]
            ok = fresh | ~np.isin(ray, ray[fresh])
            idx, r, ray = idx[ok], r[ok], ray[ok]
        if not len(r):
            return out
        order = np.lexsort((r, ray))
        ray, r, idx = ray[order], r[order], idx[order]
        first = np.concatenate([[True], ray[1:] != ray[:-1]])
        rank = np.arange(len(ray)) - np.flatnonzero(first)[np.cumsum(first) - 1]
        out[ray[first]] = np.minimum(r[first], self.max_range)
        self.picked, self.picked_rays, self.kept = idx[first], ray[first], idx[rank < keep]
        return out

    def fill(self, rays, src_rays, src_pts, up=(0.0, 0.0, 1.0), k=4):
        """Ranges for rays from the k nearest src_rays (grid distance): their points' mean height below the fan along
        up (world up in the base frame), as a level plane met by each ray."""
        row, col = np.divmod(rays, self.n_az)
        srow, scol = np.divmod(src_rays, self.n_az)
        dc = np.abs(self.az_rank[col][:, None] - self.az_rank[scol][None, :])
        if self.circular:
            dc = np.minimum(dc, self.n_az - dc)
        d2 = (self.dep_rank[row][:, None] - self.dep_rank[srow][None, :]) ** 2 + dc ** 2
        k = min(k, len(src_rays))
        near = np.argpartition(d2, k - 1, axis=1)[:, :k]
        w = 1.0 / np.maximum(np.take_along_axis(d2, near, axis=1), 1)
        up = np.asarray(up, dtype=np.float64)
        h = (w * ((self.pos - src_pts[near]) @ up)).sum(axis=1) / w.sum(axis=1)
        down = -(self.dirs[rays] @ up)
        hit = (h > 0.0) & (down > 1e-3)
        return np.where(hit, np.minimum(h / np.where(hit, down, 1.0), self.max_range), self.max_range)


class FanTracker:
    """FanBins over time: points in the base frame, moved with it; each tick bins the new ones and each ray's KEEP
    nearest from before, and fills the rays with no point."""

    KEEP = 3

    def __init__(self, bins, window):
        self.bins, self.window = bins, window
        self.new = []
        self.kept = np.zeros((0, 3))
        self.kept_t = np.zeros(0)
        self.ray_t = np.full(len(bins.prior), -np.inf)
        self.stale = np.ones(len(bins.prior), dtype=bool)

    def add(self, t, pts):
        self.new.append((t, pts))

    def move(self, dR, d):
        """The base moved: old base <- new base = (dR, d)."""
        self.kept = (self.kept - d) @ dR
        self.new = [(t, (p - d) @ dR) for t, p in self.new]

    def ranges(self, t, up=(0.0, 0.0, 1.0)):
        young = t - self.kept_t <= self.window
        base = np.concatenate([self.kept[young]] + [p for _, p in self.new])
        stamp = np.concatenate([self.kept_t[young]] + [np.full(len(p), s) for s, p in self.new])
        new = np.arange(len(base)) >= young.sum()
        self.new = []
        b = self.bins
        out = b.update(base, self.KEEP, new)
        self.kept, self.kept_t = base[b.kept], stamp[b.kept]
        self.best, self.best_t = base[b.picked], stamp[b.picked]
        self.ray_t[b.picked_rays] = self.best_t
        self.stale = np.isnan(out)
        if self.stale.any():
            out[self.stale] = (b.fill(np.flatnonzero(self.stale), b.picked_rays, self.best, up) if len(b.picked)
                               else b.prior[self.stale])
        return out


class FanInput:
    """lidar_fan without ROS (also stairs_mujoco's): clouds, base motion and leg links in, the fan's ranges out."""

    def __init__(self, fan, cfg):
        self.window = float(cfg.get("window_s", 0.5))
        self.timeout = float(cfg.get("cloud_timeout_s", 0.5))
        self.leg_radius = float(cfg.get("leg_radius", 0.045))
        self.bins = FanBins(fan.pos, fan.depressions, fan.azimuths, fan.max_range, float(cfg.get("prior_height", 0.273)))
        self.tracker = FanTracker(self.bins, self.window)
        self.links = None
        self.last_cloud = self.last_motion = None
        self.att = self.att_t = None
        self.steps = deque()

    def motion(self, stamp, R, v, now):
        """Base attitude R (odom <- base) and velocity v (base frame) at stamp: moves the kept points."""
        if self.att is not None and 0.0 < stamp - self.att_t < self.timeout:
            step = (self.att.T @ R, np.asarray(v, dtype=np.float64) * (stamp - self.att_t))
            self.tracker.move(*step)
            self.steps.append((stamp, *step))
            while stamp - self.steps[0][0] > self.timeout:
                self.steps.popleft()
            self.last_motion = now
        self.att, self.att_t = R, stamp

    def cloud(self, stamp, pts, origin, now, times=None):
        """A cloud measured from stamp on, each point times after it: points in the base frame at that time, the
        lidar at origin."""
        times = np.zeros(len(pts)) if times is None else np.asarray(times, dtype=np.float64)
        keep = pts[:, 2] < self.bins.pos[2]
        pts, times = pts[keep], times[keep]
        if self.links is not None and self.leg_radius > 0 and len(pts):
            keep = ~leg_shadow(pts, origin, self.links, self.leg_radius)
            pts, times = pts[keep], times[keep]
        if self.steps and self.steps[-1][0] - stamp < self.timeout:
            measured = stamp + times
            for s, dR, d in self.steps:
                later = measured < s
                if later.any():
                    pts[later] = (pts[later] - d) @ dR
        self.tracker.add(now, pts)
        self.last_cloud = now

    def ranges(self, now):
        """The fan's ranges, or None while the cloud or the motion is older than cloud_timeout_s."""
        if None in (self.last_cloud, self.last_motion) or now - min(self.last_cloud, self.last_motion) > self.timeout:
            return None
        return self.tracker.ranges(now, self.att[2])


def load_fan(checkpoint, wait_s):
    """The lidar fan of a .pt (its .qpol, which real_driver exports) or .qpol, or None if it has none."""
    import quadruped_control as qc

    path = checkpoint if os.path.isabs(checkpoint) else paths.repo(checkpoint)
    qpol = os.path.splitext(path)[0] + ".qpol" if path.endswith(".pt") else path
    deadline = time.monotonic() + wait_s
    while not os.path.exists(qpol) or (path != qpol and os.path.exists(path)
                                        and os.path.getmtime(qpol) < os.path.getmtime(path)):
        if time.monotonic() > deadline:
            return None
        time.sleep(1.0)
    try:
        return qc.PolicySpec.from_file(qpol).lidar
    except Exception as e:
        print(f"[lidar_fan] {qpol}: {e}")
        return None


def find_fan(checkpoint, wait_s):
    """The launcher's checkpoint if it has a lidar, else the first policies.json entry whose .qpol has one."""
    if checkpoint:
        fan = load_fan(checkpoint, wait_s)
        if fan is not None:
            return checkpoint, fan
    for name, entry in (load_policies().get("policies") or {}).items():
        fan = load_fan(entry["path"], 0.0)
        if fan is not None:
            return name, fan
    return None, None


class LidarFan(Node):
    def __init__(self, fan, cfg):
        super().__init__("lidar_fan")
        self.frame = cfg.get("frame", "base")
        self.fan = FanInput(fan, cfg)
        self.mounts = {}
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(Float32MultiArray, "/lidar/fan_ranges", 10)
        self.create_subscription(PointCloud2, "/lidar/points", self._cloud_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/odom/state_estimator", self._motion_cb, 50)
        if self.fan.leg_radius > 0:
            self.create_subscription(JointState, "/sensors/joint_states", self._joints_cb, 10)
        self.create_timer(1.0 / float(cfg.get("rate_hz", 50.0)), self._tick)
        self.get_logger().info(f"{len(fan.depressions)} x {len(fan.azimuths)} rays from {list(fan.pos)}, "
                               f"window {self.fan.window:.2f} s -> /lidar/fan_ranges")

    def _joints_cb(self, msg):
        links = leg_links(msg)
        if links is not None:
            self.fan.links = links

    def _motion_cb(self, msg):
        v = msg.twist.twist.linear
        self.fan.motion(Time.from_msg(msg.header.stamp).nanoseconds * 1e-9, rotation(msg.pose.pose.orientation),
                        (v.x, v.y, v.z), time.monotonic())

    def _cloud_cb(self, msg):
        if msg.header.frame_id not in self.mounts:
            try:
                tf = self.tf_buffer.lookup_transform(self.frame, msg.header.frame_id, Time()).transform
            except TransformException:
                return
            self.mounts[msg.header.frame_id] = (rotation(tf.rotation),
                                                np.array([tf.translation.x, tf.translation.y, tf.translation.z]))
        R, t = self.mounts[msg.header.frame_id]
        offsets = {f.name: f.offset for f in msg.fields}
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
        pts = np.stack([raw[:, offsets[n]:offsets[n] + 4].copy().view(np.float32)[:, 0] for n in "xyz"], axis=1)
        times = (raw[:, offsets["time"]:offsets["time"] + 4].copy().view(np.float32)[:, 0] if "time" in offsets
                 else np.zeros(len(pts), dtype=np.float32))
        ok = np.isfinite(pts).all(axis=1)
        self.fan.cloud(Time.from_msg(msg.header.stamp).nanoseconds * 1e-9, pts[ok].astype(np.float64) @ R.T + t, t,
                       time.monotonic(), times[ok])

    def _tick(self):
        ranges = self.fan.ranges(time.monotonic())
        if ranges is not None and rclpy.ok():
            self.pub.publish(Float32MultiArray(data=ranges.tolist()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", default="", help="policy .pt or .qpol; default: the first lidar policy in policies.json")
    ap.add_argument("--wait", type=float, default=60.0, help="seconds to wait for the checkpoint's .qpol")
    args = ap.parse_args(remove_ros_args()[1:])

    name, fan = find_fan(args.checkpoint, args.wait)
    if fan is None:
        print("[lidar_fan] No lidar policy (launcher checkpoint or policies.json): nothing to do.")
        return
    print(f"[lidar_fan] Fan from {name}")
    rclpy.init()
    node = LidarFan(fan, load_config().get("lidar_fan") or {})
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
