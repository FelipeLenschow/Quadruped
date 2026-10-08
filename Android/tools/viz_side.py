"""Publishes fake Nav2/robot viz topics for viz_check and prints the goal it receives.

Also plays Point-LIO's 3D map: a 5 x 4 m box room on /lio/map_voxels (latched snapshot),
a box growing on /lio/map_voxels/delta at 1 Hz, and /lio/odom driving a 1 m circle.
`--big N` pads the snapshot to N points, to test a large transfer. `--seconds S` runs longer
(for a phone on the same network, set ROS_DOMAIN_ID to the app's domain).
"""
import argparse
import math
import time

import numpy as np

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import JointState, LaserScan, PointCloud2, PointField
from tf2_msgs.msg import TFMessage

ap = argparse.ArgumentParser()
ap.add_argument("--big", type=int, default=0)
ap.add_argument("--seconds", type=float, default=7.0)
args = ap.parse_args()

rclpy.init()
node = rclpy.create_node("viz_side")
latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

grid = OccupancyGrid()
grid.header.frame_id = "map"
grid.info.resolution = 0.05
grid.info.width, grid.info.height = 300, 200
grid.info.origin.position.x, grid.info.origin.position.y = -7.5, -5.0
grid.data = [(-1, 0, 100)[i % 3] for i in range(300 * 200)]
node.create_publisher(OccupancyGrid, "/map", latched).publish(grid)


def tf(parent, child, x, y, z, yaw=0.0):
    t = TransformStamped()
    t.header.frame_id, t.child_frame_id = parent, child
    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = x, y, z
    t.transform.rotation.z, t.transform.rotation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return t


node.create_publisher(TFMessage, "/tf_static", latched).publish(TFMessage(transforms=[tf("base", "radar", 0.29, 0.0, -0.05)]))
tf_pub = node.create_publisher(TFMessage, "/tf", 100)
scan_pub = node.create_publisher(LaserScan, "/scan", qos_profile_sensor_data)
plan_pub = node.create_publisher(Path, "/plan", 10)
js_pub = node.create_publisher(JointState, "/sensors/joint_states", 10)
node.create_subscription(PoseStamped, "/goal_pose",
                         lambda m: print(f"goal: frame={m.header.frame_id} x={m.pose.position.x} y={m.pose.position.y} "
                                         f"yaw={2 * math.atan2(m.pose.orientation.z, m.pose.orientation.w):.4f}"), 10)


def tick():
    tf_pub.publish(TFMessage(transforms=[tf("map", "odom", 1.0, 0.0, 0.0, math.pi / 2),
                                         tf("odom", "base_footprint", 2.0, 0.0, 0.0),
                                         tf("odom", "base", 2.0, 0.0, 0.33)]))
    scan = LaserScan()
    scan.header.frame_id = "base_footprint"
    scan.angle_min, scan.angle_increment, scan.range_min, scan.range_max = -math.pi, math.pi / 180, 0.2, 15.0
    scan.ranges = [1.5] * 360
    scan_pub.publish(scan)
    path = Path()
    path.header.frame_id = "map"
    for x, y in [(0.0, 0.0), (0.5, 0.25), (1.0, 1.0)]:
        p = PoseStamped()
        p.pose.position.x, p.pose.position.y = x, y
        path.poses.append(p)
    plan_pub.publish(path)
    js = JointState()
    js.name = [f"{l}_{j}_joint" for j in ("hip", "thigh", "calf") for l in ("FL", "FR", "RL", "RR")]
    js.position = [0.1 * i for i in range(12)]
    js_pub.publish(js)


VOXEL = 0.1


def cloud(points):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    m = PointCloud2()
    m.header.frame_id = "lio_odom"
    m.header.stamp = node.get_clock().now().to_msg()
    m.height, m.width = 1, len(pts)
    m.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")]
    m.is_bigendian, m.point_step, m.row_step, m.is_dense = False, 12, 12 * len(pts), True
    m.data = pts.tobytes()
    return m


def room():
    # Voxel centres, (k + 0.5) * VOXEL, like lio_map_stream. lio_odom starts at the IMU,
    # about 0.3 m above the floor.
    xs, ys = (np.arange(-15, 35) + 0.5) * VOXEL, (np.arange(-15, 25) + 0.5) * VOXEL
    zs = (np.arange(-3, 20) + 0.5) * VOXEL
    pts = [(x, y, zs[0]) for x in xs for y in ys]
    for z in zs:
        pts += [(x, y, z) for x in xs for y in (ys[0], ys[-1])]
        pts += [(x, y, z) for y in ys[1:-1] for x in (xs[0], xs[-1])]
    if args.big > len(pts):
        # Extra floors above the room, only to make the snapshot heavy.
        level = 25
        while len(pts) < args.big:
            pts += [(x, y, (level + 0.5) * VOXEL) for x in xs for y in ys][: args.big - len(pts)]
            level += 1
    return pts


def box(k):
    # A box growing one row of voxels a second; each delta repeats the last row too,
    # so the app's dedupe gets exercised.
    return [(VOXEL * (25.5 + i), VOXEL * (-9.5 + j), VOXEL * (z + 0.5)) for j in (max(k - 1, 0), k)
            for i in range(3) for z in range(-3, 3)]


reliable = QoSReliabilityPolicy.RELIABLE
snap_pub = node.create_publisher(PointCloud2, "/lio/map_voxels",
                                 QoSProfile(depth=1, reliability=reliable, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))
delta_pub = node.create_publisher(PointCloud2, "/lio/map_voxels/delta", QoSProfile(depth=10, reliability=reliable))
odom_pub = node.create_publisher(Odometry, "/lio/odom", qos_profile_sensor_data)
mapped = room()
snap_pub.publish(cloud(mapped))
start = time.monotonic()
deltas = [0]


def lio_delta():
    pts = box(deltas[0])
    deltas[0] += 1
    mapped.extend(pts)
    delta_pub.publish(cloud(pts))


def lio_snapshot():
    snap_pub.publish(cloud(sorted(set(mapped))))


def lio_odom():
    th = 0.5 * (time.monotonic() - start)
    o = Odometry()
    o.header.frame_id, o.child_frame_id = "lio_odom", "lio_imu"
    o.header.stamp = node.get_clock().now().to_msg()
    o.pose.pose.position.x, o.pose.pose.position.y = math.sin(th), 1.0 - math.cos(th)
    o.pose.pose.orientation.z, o.pose.pose.orientation.w = math.sin(th / 2), math.cos(th / 2)
    o.twist.twist.linear.x, o.twist.twist.angular.z = 0.5, 0.5
    odom_pub.publish(o)


node.create_timer(0.1, tick)
node.create_timer(1.0, lio_delta)
node.create_timer(3.0, lio_snapshot)
node.create_timer(1 / 15, lio_odom)
end = time.monotonic() + args.seconds
while time.monotonic() < end:
    rclpy.spin_once(node, timeout_sec=0.05)
