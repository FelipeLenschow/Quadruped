"""Lidar body filter: drops the points inside a box around the robot's own body and legs.

Everything outside the box is kept, so a person's legs next to the robot survive for a follow
behaviour. The cloud is republished in the robot's base frame on /lidar/points_filtered.

With leg_radius set, it also drops every point whose beam passed within leg_radius of a thigh or calf
(placed by forward kinematics from /sensors/joint_states). A beam that clips a leg's edge returns a
mix of the leg and the floor behind it, a point out along the beam past the leg and outside the box;
on the real Go2 these came off the front-left thigh at 0.10-0.13 m and marked phantom obstacles.

With --accumulate N the last N clouds are merged, each moved through odom into the newest cloud's base
frame. The real L1 covers only a patch of its field of view per cloud (~44 of 360 scan beams), so one
cloud alone leaves most of /scan empty.
"""

import argparse
import array
import signal
from collections import deque

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import JointState, PointCloud2, PointField
from tf2_ros import Buffer, TransformException, TransformListener

from quadruped_core.config_loader import load_config
from quadruped_core.telemetry.kinematics import leg_chain_body

FIELDS = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
          for i, n in enumerate(("x", "y", "z", "intensity"))]
ODOM_FRAME = "odom"
LEGS = ("FL", "FR", "RL", "RR")


def rotation(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class LidarFilter(Node):
    def __init__(self, cfg, accumulate=1):
        super().__init__("lidar_filter")
        self.frame = cfg.get("frame", "base")
        box = cfg.get("body_box") or {}
        self.box_min = np.array([box.get(k, (-0.45, 0.45))[0] for k in "xyz"])
        self.box_max = np.array([box.get(k, (-0.45, 0.45))[1] for k in "xyz"])
        self.max_range = float(cfg.get("max_range", 20.0))
        self.leg_radius = float(cfg.get("leg_radius", 0.0))
        self.links = None  # (8, 2, 3): thigh and calf end points in base, from the latest joint state
        self.history = deque(maxlen=accumulate) if accumulate > 1 else None
        self.transforms = {}
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(PointCloud2, "/lidar/points_filtered", qos_profile_sensor_data)
        self.create_subscription(PointCloud2, "/lidar/points", self._cloud_cb, qos_profile_sensor_data)
        if self.leg_radius > 0:
            self.create_subscription(JointState, "/sensors/joint_states", self._joints_cb, 10)
        self.get_logger().info(f"body box {self.box_min.tolist()} .. {self.box_max.tolist()} in {self.frame}"
                               + (f", leg shadows {self.leg_radius:.3f} m" if self.leg_radius > 0 else "")
                               + (f", merging the last {accumulate} clouds" if self.history is not None else ""))

    def _joints_cb(self, msg):
        pos = dict(zip(msg.name, msg.position))
        try:
            q = [np.array([pos[f"{leg}_{j}_joint"] for j in ("hip", "thigh", "calf")]) for leg in LEGS]
        except KeyError:
            return
        chains = [leg_chain_body(i, q[i]) for i in range(len(LEGS))]
        self.links = np.array([(c[k], c[k + 1]) for c in chains for k in (0, 1)])

    def _leg_shadow(self, pts, origin):
        """Points whose beam (origin -> point) passed within leg_radius of a leg link, by the closest
        distance between the two segments (Ericson, Real-Time Collision Detection 5.1.9)."""
        d1 = pts - origin
        a = np.maximum(np.einsum("ij,ij->i", d1, d1), 1e-12)
        hit = np.zeros(len(pts), bool)
        for A, B in self.links:
            d2, r = B - A, origin - A
            e, f = d2 @ d2, d2 @ r
            b, c = d1 @ d2, d1 @ r
            denom = a * e - b * b
            s = np.clip(np.where(denom > 1e-12, (b * f - c * e) / np.maximum(denom, 1e-12), 0.0), 0.0, 1.0)
            t = (b * s + f) / e
            s = np.where(t < 0, np.clip(-c / a, 0, 1), np.where(t > 1, np.clip((b - c) / a, 0, 1), s))
            t = np.clip(t, 0, 1)
            gap = (origin + d1 * s[:, None]) - (A + d2 * t[:, None])
            hit |= np.einsum("ij,ij->i", gap, gap) < self.leg_radius ** 2
        return hit

    def _transform(self, source):
        if source not in self.transforms:
            try:
                t = self.tf_buffer.lookup_transform(self.frame, source, Time()).transform
            except TransformException:
                return None
            self.transforms[source] = (rotation(t.rotation),
                                       np.array([t.translation.x, t.translation.y, t.translation.z]))
        return self.transforms[source]

    def _odom_pose(self, stamp):
        """odom <- base at the cloud's time, or the latest one if TF has not caught up with it yet."""
        for t in (Time.from_msg(stamp), Time()):
            try:
                tf = self.tf_buffer.lookup_transform(ODOM_FRAME, self.frame, t).transform
            except TransformException:
                continue
            return rotation(tf.rotation), np.array([tf.translation.x, tf.translation.y, tf.translation.z])
        return None

    def _merge(self, out, stamp):
        """This cloud and the ones before it, all in this cloud's base frame."""
        pose = self._odom_pose(stamp)
        if pose is None:
            return out
        R, t = pose
        self.history.append((out[:, :3].astype(np.float64) @ R.T + t, out[:, 3]))
        pts = np.concatenate([h[0] for h in self.history])
        intensity = np.concatenate([h[1] for h in self.history])
        return np.column_stack([(pts - t) @ R, intensity]).astype(np.float32)

    def _cloud_cb(self, msg):
        tf = self._transform(msg.header.frame_id)
        if tf is None:
            return
        offsets = {f.name: f.offset for f in msg.fields}
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
        cols = [raw[:, offsets[n]:offsets[n] + 4].copy().view(np.float32)[:, 0] for n in ("x", "y", "z")]
        intensity = (raw[:, offsets["intensity"]:offsets["intensity"] + 4].copy().view(np.float32)[:, 0]
                     if "intensity" in offsets else np.zeros(len(raw), np.float32))
        pts = np.stack(cols, axis=1)
        keep = np.isfinite(pts).all(axis=1) & (np.linalg.norm(pts, axis=1) < self.max_range)
        pts = pts[keep] @ tf[0].T + tf[1]
        intensity = intensity[keep]
        outside = ~((pts > self.box_min) & (pts < self.box_max)).all(axis=1)
        if self.links is not None:
            outside[outside] = ~self._leg_shadow(pts[outside], tf[1])
        out = np.column_stack([pts[outside], intensity[outside]]).astype(np.float32)
        if self.history is not None:
            out = self._merge(out, msg.header.stamp)

        cloud = PointCloud2()
        cloud.header.stamp = msg.header.stamp
        cloud.header.frame_id = self.frame
        cloud.height, cloud.width = 1, len(out)
        cloud.fields = FIELDS
        cloud.point_step, cloud.row_step = 16, 16 * len(out)
        cloud.is_dense = True
        data = array.array("B")
        data.frombytes(out.tobytes())
        cloud.data = data
        self.pub.publish(cloud)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--accumulate", type=int, default=1, help="merge the last N clouds through odom")
    args = ap.parse_args(remove_ros_args()[1:])

    rclpy.init()
    node = LidarFilter(load_config().get("lidar_filter") or {}, args.accumulate)
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
