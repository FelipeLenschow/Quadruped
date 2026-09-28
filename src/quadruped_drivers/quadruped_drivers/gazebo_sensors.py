"""Gazebo sensors to ROS 2: the Go2's lidar and front RGB-D camera, stamped with sim time."""

import argparse
import array
import math
import signal
import traceback

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from tf2_ros import StaticTransformBroadcaster

from gz.msgs10.camera_info_pb2 import CameraInfo as GzCameraInfo
from gz.msgs10.image_pb2 import Image as GzImage
from gz.msgs10.pointcloud_packed_pb2 import PointCloudPacked
from gz.transport13 import Node as GzNode

# base -> sensor mounts, as in the Unitree Go2 URDF and model.sdf
MOUNTS = {
    "radar": ((0.28945, 0.0, -0.046825), (0.0, 2.8782, 0.0)),
    "front_camera": ((0.32715, 0.0, 0.04297), (0.0, 0.0, 0.0)),
}
IMAGE_ENCODINGS = {3: "rgb8", 13: "32FC1"}


def quaternion(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


class GazeboSensors(Node):
    def __init__(self, args):
        super().__init__("gazebo_sensors")
        self.base_frame = args.base_frame
        self.optical_frame = "front_camera_optical"
        self.cloud_pub = self.create_publisher(PointCloud2, "/lidar/points", 5)
        self.image_pub = self.create_publisher(Image, "/camera/image_raw", 5)
        # self.depth_pub = self.create_publisher(Image, "/camera/depth/image_raw", 5)
        self.info_pub = self.create_publisher(CameraInfo, "/camera/camera_info", 5)
        self._publish_static_tf()

        self.gz = GzNode()
        self.gz.subscribe(PointCloudPacked, "/lidar/points", self._guard(self._cloud_cb))
        self.gz.subscribe(GzImage, "/front_camera/image", self._guard(lambda m: self._image_cb(m, self.image_pub)))
        # self.gz.subscribe(GzImage, "/front_camera/depth_image", self._guard(lambda m: self._image_cb(m, self.depth_pub)))
        self.gz.subscribe(GzCameraInfo, "/front_camera/camera_info", self._guard(self._info_cb))
        self.get_logger().info("/lidar/points, /camera/{image_raw,camera_info}")

    def _guard(self, cb):
        failed = []

        def wrapped(msg):
            try:
                if rclpy.ok():
                    cb(msg)
            except Exception:
                if not failed and rclpy.ok():
                    failed.append(True)
                    self.get_logger().error(traceback.format_exc())
        return wrapped

    def close(self):
        for topic in self.gz.subscribed_topics():
            self.gz.unsubscribe(topic)

    def _publish_static_tf(self):
        transforms = []
        for child, (xyz, rpy) in MOUNTS.items():
            transforms.append(self._transform(self.base_frame, child, xyz, rpy))
        transforms.append(self._transform("front_camera", self.optical_frame, (0.0, 0.0, 0.0),
                                          (-math.pi / 2, 0.0, -math.pi / 2)))
        self.tf = StaticTransformBroadcaster(self)
        self.tf.sendTransform(transforms)

    def _transform(self, parent, child, xyz, rpy):
        t = TransformStamped()
        t.header.frame_id, t.child_frame_id = parent, child
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = xyz
        q = quaternion(*rpy)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q
        return t

    @staticmethod
    def _bytes(data):
        # Assigning bytes to a uint8[] field copies element by element (~90 ms per image).
        out = array.array("B")
        out.frombytes(data)
        return out

    @staticmethod
    def _stamp(msg, gz_header):
        msg.header.stamp.sec = gz_header.stamp.sec
        msg.header.stamp.nanosec = gz_header.stamp.nsec

    def _cloud_cb(self, m):
        msg = PointCloud2()
        self._stamp(msg, m.header)
        msg.header.frame_id = "radar"
        msg.height, msg.width = m.height, m.width
        msg.fields = [PointField(name=f.name, offset=f.offset, datatype=f.datatype + 1, count=max(f.count, 1))
                      for f in m.field]
        msg.is_bigendian = m.is_bigendian
        msg.point_step, msg.row_step = m.point_step, m.row_step
        msg.is_dense = m.is_dense
        msg.data = self._bytes(m.data)
        self.cloud_pub.publish(msg)

    def _image_cb(self, m, pub):
        encoding = IMAGE_ENCODINGS.get(m.pixel_format_type)
        if encoding is None:
            return
        msg = Image()
        self._stamp(msg, m.header)
        msg.header.frame_id = self.optical_frame
        msg.height, msg.width, msg.step = m.height, m.width, m.step
        msg.encoding = encoding
        msg.data = self._bytes(m.data)
        pub.publish(msg)

    def _info_cb(self, m):
        msg = CameraInfo()
        self._stamp(msg, m.header)
        msg.header.frame_id = self.optical_frame
        msg.height, msg.width = m.height, m.width
        msg.distortion_model = "plumb_bob"
        msg.d = list(m.distortion.k)[:5] or [0.0] * 5
        msg.k = list(m.intrinsics.k)
        msg.r = list(m.rectification_matrix) or [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = list(m.projection.p)
        self.info_pub.publish(msg)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base_frame", default="base")
    args = ap.parse_args(remove_ros_args()[1:])

    rclpy.init()
    node = GazeboSensors(args)
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
