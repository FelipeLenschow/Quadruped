"""Real Go2 sensors to ROS 2: the L1 lidar cloud and IMU from Unitree's DDS, on this machine's clock.

Both keep the robot's own stamps, shifted by one offset onto this clock (RobotClock), so Point-LIO sees the
cloud and IMU on the same timeline.
"""

import argparse
import array
import os
import sys
import time
from collections import deque
from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types
import numpy as np
import rclpy
from cyclonedds.core import Policy, Qos
from cyclonedds.domain import DomainParticipant
from cyclonedds.internal import InvalidSample
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import Imu, PointCloud2, PointField
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Quaternion_, Vector3_
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import Header_, String_

from quadruped_core.config_loader import load_config
from quadruped_drivers.sensor_mounts import MOUNTS, publish_static_tf, transform

LIDAR_SWITCH_TOPIC = "rt/utlidar/switch"
STALE_S = 1.0
MIN_RANGE = 0.05  # the L1's own minimum range; nearer after the range correction is not a real return
CLOCK_WINDOW_S = 10.0


@dataclass
@annotate.final
@annotate.autoid("sequential")
class Imu_(idl.IdlStruct, typename="sensor_msgs.msg.dds_.Imu_"):
    header: Header_
    orientation: Quaternion_
    orientation_covariance: types.array[types.float64, 9]
    angular_velocity: Vector3_
    angular_velocity_covariance: types.array[types.float64, 9]
    linear_acceleration: Vector3_
    linear_acceleration_covariance: types.array[types.float64, 9]


class RobotClock:
    """Maps the robot's stamps onto this clock: offset = the smallest (arrival - stamp) seen in the last
    CLOCK_WINDOW_S, i.e. the stamp plus the least transport delay. The window follows slow clock drift."""

    def __init__(self, node):
        self.node = node
        self.deltas = deque()

    def stamp(self, header):
        now = self.node.get_clock().now()
        robot_ns = header.stamp.sec * 10**9 + header.stamp.nanosec
        if robot_ns == 0:
            return now.to_msg()
        t = now.nanoseconds * 1e-9
        delta = now.nanoseconds - robot_ns
        while self.deltas and self.deltas[-1][1] >= delta:
            self.deltas.pop()
        self.deltas.append((t, delta))
        while self.deltas[0][0] < t - CLOCK_WINDOW_S:
            self.deltas.popleft()
        return Time(nanoseconds=robot_ns + self.deltas[0][1]).to_msg()


class RealSensors(Node):
    def __init__(self, args, cfg):
        super().__init__("real_sensors")
        self.frame = args.frame
        self.cloud_pub = self.create_publisher(PointCloud2, "/lidar/points", 5)
        self.imu_pub = self.create_publisher(Imu, "/lidar/imu", 200)
        self.clock = RobotClock(self)
        xyz, rpy = MOUNTS["radar"]
        lidar_cfg = cfg.get("real_lidar") or {}
        mount = (tuple(lidar_cfg.get("xyz", xyz)), tuple(lidar_cfg.get("rpy", rpy)))
        self.range_offset = float(lidar_cfg.get("range_offset", 0.0))
        self.imu_accel = bool(lidar_cfg.get("imu_accel", False))
        # Point-LIO's world is the lidar's axes at its start; lio_world puts it upright for viewing.
        self.tf = publish_static_tf(self, args.base_frame, {"radar": mount},
                                    [transform("lio_world", "lio_odom", *mount)])
        self.get_logger().info(f"lidar mount xyz {list(mount[0])} rpy {list(mount[1])}, "
                               f"range offset {self.range_offset:.3f} m")

        participant = DomainParticipant(0)
        self._participant = participant
        self.reader = DataReader(participant, Topic(participant, args.topic, PointCloud2_),
                                 qos=Qos(Policy.History.KeepLast(1)))
        self.imu_reader = DataReader(participant, Topic(participant, args.imu_topic, Imu_),
                                     qos=Qos(Policy.History.KeepLast(100)))
        self.imu_count = 0
        self.switch = None
        if args.switch_on:
            self.switch = ChannelPublisher(LIDAR_SWITCH_TOPIC, String_)
            self.switch.Init()

        self.last_cloud = None
        self.stale = False
        self.last_switch = 0.0
        self.create_timer(0.02, self._poll)
        self.imu_frame = args.imu_frame
        self.started = time.monotonic()
        self.create_timer(5.0, self._report_imu)
        self.get_logger().info(f"{args.topic} -> /lidar/points ({self.frame}), {args.imu_topic} -> /lidar/imu")

    def _poll(self):
        now = time.monotonic()
        for m in self.imu_reader.take(N=100):
            if not isinstance(m, InvalidSample):
                self._publish_imu(m)
        samples = self.reader.take(1)
        if samples and not isinstance(samples[0], InvalidSample):
            if self.last_cloud is None:
                m = samples[0]
                self.get_logger().info(f"first cloud: frame '{m.header.frame_id}', {m.width * m.height} points, "
                                       f"fields {[f.name for f in m.fields]}")
            elif self.stale:
                self.get_logger().info("lidar cloud resumed")
            self.last_cloud, self.stale = now, False
            self._publish(samples[0])
        elif self.last_cloud is not None and not self.stale and now - self.last_cloud > STALE_S:
            self.stale = True
            self.get_logger().warn(f"no lidar cloud for {STALE_S:.0f} s")

        if self.switch is not None and self.last_cloud is None and now - self.last_switch > 1.0:
            self.last_switch = now
            self.switch.Write(String_("ON"))

    def _report_imu(self):
        rate = self.imu_count / 5.0
        if self.imu_count == 0:
            self.get_logger().warn("no lidar IMU samples (Point-LIO needs them)", once=True)
        elif not getattr(self, "_imu_reported", False):
            self._imu_reported = True
            self.get_logger().info(f"lidar IMU {rate:.0f} Hz")
        self.imu_count = 0

    def _publish_imu(self, m):
        self.imu_count += 1
        msg = Imu()
        msg.header.stamp = self.clock.stamp(m.header)
        msg.header.frame_id = self.imu_frame
        q, w, a = m.orientation, m.angular_velocity, m.linear_acceleration
        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = q.x, q.y, q.z, q.w
        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = w.x, w.y, w.z
        if self.imu_accel:
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = a.x, a.y, a.z
        msg.orientation_covariance = list(m.orientation_covariance)
        msg.angular_velocity_covariance = list(m.angular_velocity_covariance)
        msg.linear_acceleration_covariance = list(m.linear_acceleration_covariance)
        self.imu_pub.publish(msg)

    def _publish(self, m):
        msg = PointCloud2()
        msg.header.stamp = self.clock.stamp(m.header)
        msg.header.frame_id = self.frame
        msg.height, msg.width = m.height, m.width
        msg.fields = [PointField(name=f.name, offset=f.offset, datatype=f.datatype, count=f.count) for f in m.fields]
        msg.is_bigendian = m.is_bigendian
        msg.point_step, msg.row_step = m.point_step, m.row_step
        data = bytearray(m.data)
        dense = self._correct_range(m, data) if self.range_offset else True
        msg.is_dense = m.is_dense and dense
        msg.data = array.array("B", data)
        self.cloud_pub.publish(msg)

    def _correct_range(self, m, data):
        """The L1 reports every range range_offset too long: pulls each point back along its ray, in place.
        Points left nearer than MIN_RANGE become NaN. Returns False if any did."""
        off = {f.name: f.offset for f in m.fields}
        f4 = ">f4" if m.is_bigendian else "<f4"
        pts = np.frombuffer(data, np.dtype({"names": ["x", "y", "z"], "formats": [f4] * 3,
                                            "offsets": [off["x"], off["y"], off["z"]], "itemsize": m.point_step}),
                            count=m.width * m.height)
        r = np.sqrt(pts["x"].astype(np.float64) ** 2 + pts["y"] ** 2 + pts["z"] ** 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            k = np.where(r - self.range_offset >= MIN_RANGE, 1.0 - self.range_offset / r, np.nan)
        for c in "xyz":
            pts[c] *= k
        return not np.isnan(k).any()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interface", default=None)
    ap.add_argument("--topic", default="rt/utlidar/cloud")
    ap.add_argument("--frame", default="radar", help="frame the cloud is published in")
    ap.add_argument("--imu_topic", default="rt/utlidar/imu")
    ap.add_argument("--imu_frame", default="radar_imu")
    ap.add_argument("--base_frame", default="base")
    ap.add_argument("--no_switch_on", dest="switch_on", action="store_false", help="don't turn the lidar on")
    args = ap.parse_args(remove_ros_args()[1:])

    os.environ.pop("CYCLONEDDS_URI", None)
    try:
        ChannelFactoryInitialize(0, networkInterface=args.interface)
    except Exception as e:
        print(f"[SDK2] Failed to initialize ChannelFactory: {e}")
        sys.exit(1)

    cfg = load_config()
    os.environ["ROS_DOMAIN_ID"] = str(cfg.get("network", {}).get("ros_domain_id", "1"))
    rclpy.init()
    node = RealSensors(args, cfg)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
