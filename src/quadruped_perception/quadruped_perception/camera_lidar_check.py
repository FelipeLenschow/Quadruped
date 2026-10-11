"""Camera mount check: draws the lidar points on a camera's image, coloured by distance (near red, far blue).

If the camera's mount and calibration are right, the points land on the objects they hit: a cone's points
on the cone, a wall's edge on the edge. Publishes /camera/<name>/lidar_check/compressed at --rate; view it
with rqt_image_view or rviz. The last --accumulate clouds are drawn together (the L1 covers only a patch of
its view per cloud), so keep the robot still.
"""

import argparse
import array
import signal
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CameraInfo, CompressedImage, PointCloud2
from tf2_ros import Buffer, TransformException, TransformListener

from quadruped_perception.lidar_filter import rotation


def xyz(msg):
    off = {f.name: f.offset for f in msg.fields}
    f4 = ">f4" if msg.is_bigendian else "<f4"
    pts = np.frombuffer(bytes(msg.data), np.dtype({"names": ["x", "y", "z"], "formats": [f4] * 3,
                                                   "offsets": [off["x"], off["y"], off["z"]],
                                                   "itemsize": msg.point_step}), count=msg.width * msg.height)
    out = np.column_stack([pts[c] for c in "xyz"]).astype(np.float64)
    return out[np.isfinite(out).all(axis=1)]


def project(info, pts):
    """Points in the camera's optical frame to pixels of info's image, with its distortion."""
    k = np.array(info.k).reshape(3, 3)
    d = np.array(info.d, float)
    zero = np.zeros(3)
    if info.distortion_model == "equidistant":
        uv, _ = cv2.fisheye.projectPoints(pts[:, None, :], zero, zero, k, d[:4])
    else:
        uv, _ = cv2.projectPoints(pts, zero, zero, k, d)
    return uv.reshape(-1, 2)


def pixel_ray(info, u, v):
    """The direction (unnormalised, z = 1) in the camera's optical frame that pixel (u, v) looks along."""
    k = np.array(info.k).reshape(3, 3)
    d = np.array(info.d, float)
    p = np.array([[[u, v]]], np.float64)
    if info.distortion_model == "equidistant":
        x, y = cv2.fisheye.undistortPoints(p, k, d[:4]).ravel()
    else:
        x, y = cv2.undistortPoints(p, k, d).ravel()
    return np.array([x, y, 1.0])


class CameraLidarCheck(Node):
    def __init__(self, args):
        super().__init__("camera_lidar_check")
        self.max_range = args.max_range
        self.info = None
        self.image = None
        self.clouds = deque(maxlen=args.accumulate)
        self.cloud_frame = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        base = f"/camera/{args.camera}"
        self.create_subscription(CompressedImage, f"{base}/image_raw/compressed", self._image_cb, 2)
        self.create_subscription(CameraInfo, f"{base}/camera_info", self._info_cb, 2)
        self.create_subscription(PointCloud2, args.cloud, self._cloud_cb, qos_profile_sensor_data)
        self.pub = self.create_publisher(CompressedImage, f"{base}/lidar_check/compressed", 1)
        self.create_timer(1.0 / args.rate, self._draw)
        self.get_logger().info(f"{base}/image_raw/compressed + {args.cloud} -> {base}/lidar_check/compressed")

    def _info_cb(self, m):
        self.info = m

    def _image_cb(self, m):
        self.image = m

    def _cloud_cb(self, m):
        self.cloud_frame = m.header.frame_id
        self.clouds.append(xyz(m))

    def _draw(self):
        if self.image is None:
            self.get_logger().warn("waiting for the image", throttle_duration_sec=5.0)
            return
        if self.info is None or not any(self.info.k):
            self.get_logger().warn("no camera_info: calibrate the camera first", throttle_duration_sec=5.0)
            return
        if not self.clouds:
            self.get_logger().warn("waiting for the lidar", throttle_duration_sec=5.0)
            return
        frame = self.image.header.frame_id
        try:
            t = self.tf_buffer.lookup_transform(frame, self.cloud_frame, Time()).transform
        except TransformException as e:
            self.get_logger().warn(f"{self.cloud_frame} -> {frame}: {e}", throttle_duration_sec=5.0)
            return
        image = cv2.imdecode(np.frombuffer(bytes(self.image.data), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return
        pts = np.vstack(self.clouds) @ rotation(t.rotation).T + (t.translation.x, t.translation.y, t.translation.z)
        dist = np.linalg.norm(pts, axis=1)
        keep = (pts[:, 2] > 0.1) & (dist < self.max_range)
        pts, dist = pts[keep], dist[keep]
        uv = project(self.info, pts) if len(pts) else np.zeros((0, 2))
        h, w = image.shape[:2]
        sx, sy = w / self.info.width, h / self.info.height
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < self.info.width) & (uv[:, 1] >= 0) & (uv[:, 1] < self.info.height)
        colors = cv2.applyColorMap((np.clip(dist / self.max_range, 0, 1) * 255).astype(np.uint8)[:, None],
                                   cv2.COLORMAP_JET)[:, 0]
        for (u, v), c in zip(uv[inside][::-1], colors[inside][::-1]):
            cv2.circle(image, (int(u * sx), int(v * sy)), 2, tuple(int(x) for x in c), -1)
        cv2.putText(image, f"{inside.sum()} points, max {self.max_range:.0f} m", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        msg = CompressedImage(format="bgr8; jpeg compressed bgr8")
        msg.header = self.image.header
        msg.data = array.array("B", cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
        self.pub.publish(msg)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", default="front")
    ap.add_argument("--cloud", default="/lidar/points")
    ap.add_argument("--accumulate", type=int, default=20, help="clouds drawn together")
    ap.add_argument("--max_range", type=float, default=6.0)
    ap.add_argument("--rate", type=float, default=2.0)
    args = ap.parse_args(remove_ros_args()[1:])

    rclpy.init()
    node = CameraLidarCheck(args)
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
