"""Publishes fake Nav2/robot viz topics for viz_check and prints the goal it receives."""
import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import JointState, LaserScan
from tf2_msgs.msg import TFMessage

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


node.create_timer(0.1, tick)
end = time.monotonic() + 7
while time.monotonic() < end:
    rclpy.spin_once(node, timeout_sec=0.05)
