"""Gazebo sensors to ROS 2: the Go2's lidar and front RGB-D camera, stamped with sim time."""

import argparse
import array
import signal
import traceback

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField

from gz.msgs10.camera_info_pb2 import CameraInfo as GzCameraInfo
from gz.msgs10.image_pb2 import Image as GzImage
from gz.msgs10.pointcloud_packed_pb2 import PointCloudPacked
from gz.transport13 import Node as GzNode

from quadruped_drivers.sensor_mounts import OPTICAL_FRAME, publish_static_tf

IMAGE_ENCODINGS = {3: "rgb8", 13: "32FC1"}


class GazeboSensors(Node):
    def __init__(self, args):
        super().__init__("gazebo_sensors")
        self.base_frame = args.base_frame
        self.optical_frame = OPTICAL_FRAME
        self.cloud_pub = self.create_publisher(PointCloud2, "/lidar/points", 5)
        self.image_pub = self.create_publisher(Image, "/camera/image_raw", 5)
        # self.depth_pub = self.create_publisher(Image, "/camera/depth/image_raw", 5)
        self.info_pub = self.create_publisher(CameraInfo, "/camera/camera_info", 5)
        self.tf = publish_static_tf(self, self.base_frame)

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
