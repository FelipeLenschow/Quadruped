"""Lidar body filter: drops the points inside a box around the robot's own body and legs.

Everything outside the box is kept, so a person's legs next to the robot survive for a follow
behaviour. The cloud is republished in the robot's base frame on /lidar/points_filtered.
"""

import argparse
import array
import signal

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import Buffer, TransformException, TransformListener

from quadruped_core.config_loader import load_config

FIELDS = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
          for i, n in enumerate(("x", "y", "z", "intensity"))]


def rotation(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class LidarFilter(Node):
    def __init__(self, cfg):
        super().__init__("lidar_filter")
        self.frame = cfg.get("frame", "base")
        box = cfg.get("body_box") or {}
        self.box_min = np.array([box.get(k, (-0.45, 0.45))[0] for k in "xyz"])
        self.box_max = np.array([box.get(k, (-0.45, 0.45))[1] for k in "xyz"])
        self.max_range = float(cfg.get("max_range", 20.0))
        self.transforms = {}
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(PointCloud2, "/lidar/points_filtered", qos_profile_sensor_data)
        self.create_subscription(PointCloud2, "/lidar/points", self._cloud_cb, qos_profile_sensor_data)
        self.get_logger().info(f"body box {self.box_min.tolist()} .. {self.box_max.tolist()} in {self.frame}")

    def _transform(self, source):
        if source not in self.transforms:
            try:
                t = self.tf_buffer.lookup_transform(self.frame, source, Time()).transform
            except TransformException:
                return None
            self.transforms[source] = (rotation(t.rotation),
                                       np.array([t.translation.x, t.translation.y, t.translation.z]))
        return self.transforms[source]

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
        out = np.column_stack([pts[outside], intensity[outside]]).astype(np.float32)

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
    ap.parse_args(remove_ros_args()[1:])

    rclpy.init()
    node = LidarFilter(load_config().get("lidar_filter") or {})
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
