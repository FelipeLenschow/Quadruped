"""ROS side of the link check: publishes fake robot topics and counts what the link sends."""
import sys
import time

import rclpy
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Bool, Float32, Float32MultiArray, String

rclpy.init()
node = rclpy.create_node("link_check_ros_side")
counts = {}


def counter(name):
    def cb(msg):
        counts[name] = counts.get(name, 0) + 1
    return cb


for topic, typ in [("/safety/heartbeat", Float32), ("/safety/estop", Bool),
                   ("/safety/max_torque_percent", Float32), ("/pipeline/mode", String),
                   ("/pose/command", String), ("/cmd_vel/phone", Twist)]:
    node.create_subscription(typ, topic, counter(topic), 10)

state = node.create_publisher(String, "/robot_state", 10)
estop_state = node.create_publisher(Bool, "/safety/estop_state", 10)
vel = node.create_publisher(Vector3, "/estimator/base_lin_vel", 10)
contact = node.create_publisher(Float32MultiArray, "/estimator/feet_contact", 10)
estop_latched = [False]


def on_estop(msg):
    estop_latched[0] = estop_latched[0] or msg.data


node.create_subscription(Bool, "/safety/estop", on_estop, 10)
last_twist = [None]
node.create_subscription(Twist, "/cmd_vel/phone", lambda m: last_twist.__setitem__(0, m), 10)


def tick():
    state.publish(String(data='{"mode": "pose", "posture": "standing", "tilt_deg": 1.5}'))
    estop_state.publish(Bool(data=estop_latched[0]))
    vel.publish(Vector3(x=0.01, y=0.0, z=0.0))
    contact.publish(Float32MultiArray(data=[1.0, 1.0, 1.0, 1.0]))


node.create_timer(0.1, tick)
gaps = []
last = [None]


def hb(msg):
    now = time.monotonic()
    if last[0] is not None:
        gaps.append(now - last[0])
    last[0] = now


node.create_subscription(Float32, "/safety/heartbeat", hb, 10)
end = time.monotonic() + float(sys.argv[1] if len(sys.argv) > 1 else 10)
while time.monotonic() < end:
    rclpy.spin_once(node, timeout_sec=0.05)
print("received:", counts)
if gaps:
    print(f"heartbeat gap: mean {sum(gaps) / len(gaps) * 1000:.0f} ms, max {max(gaps) * 1000:.0f} ms")
print("estop latched:", estop_latched[0])
if last_twist[0] is not None:
    t = last_twist[0]
    print(f"last twist: vx {t.linear.x} vy {t.linear.y} wz {t.angular.z}")
