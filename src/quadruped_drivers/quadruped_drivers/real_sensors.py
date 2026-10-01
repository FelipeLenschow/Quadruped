"""Real Go2 sensors to ROS 2: the L1 lidar cloud from Unitree's DDS, restamped with this machine's clock."""

import argparse
import array
import os
import sys
import time

import numpy as np
import rclpy
from cyclonedds.core import Policy, Qos
from cyclonedds.domain import DomainParticipant
from cyclonedds.internal import InvalidSample
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import PointCloud2, PointField
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

from quadruped_core.config_loader import load_config
from quadruped_drivers.sensor_mounts import MOUNTS, publish_static_tf

LIDAR_SWITCH_TOPIC = "rt/utlidar/switch"
STALE_S = 1.0
MIN_RANGE = 0.05  # the L1's own minimum range; nearer after the range correction is not a real return


class RealSensors(Node):
    def __init__(self, args, cfg):
        super().__init__("real_sensors")
        self.frame = args.frame
        self.cloud_pub = self.create_publisher(PointCloud2, "/lidar/points", 5)
        xyz, rpy = MOUNTS["radar"]
        lidar_cfg = cfg.get("real_lidar") or {}
        mount = (tuple(lidar_cfg.get("xyz", xyz)), tuple(lidar_cfg.get("rpy", rpy)))
        self.range_offset = float(lidar_cfg.get("range_offset", 0.0))
        self.tf = publish_static_tf(self, args.base_frame, {"radar": mount})
        self.get_logger().info(f"lidar mount xyz {list(mount[0])} rpy {list(mount[1])}, "
                               f"range offset {self.range_offset:.3f} m")

        participant = DomainParticipant(0)
        self._participant = participant
        self.reader = DataReader(participant, Topic(participant, args.topic, PointCloud2_),
                                 qos=Qos(Policy.History.KeepLast(1)))
        self.switch = None
        if args.switch_on:
            self.switch = ChannelPublisher(LIDAR_SWITCH_TOPIC, String_)
            self.switch.Init()

        self.last_cloud = None
        self.stale = False
        self.last_switch = 0.0
        self.create_timer(0.02, self._poll)
        self.get_logger().info(f"{args.topic} -> /lidar/points ({self.frame})")

    def _poll(self):
        now = time.monotonic()
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

    def _publish(self, m):
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
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
