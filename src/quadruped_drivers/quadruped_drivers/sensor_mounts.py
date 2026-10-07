"""Go2 sensor mounts on the base, shared by the Gazebo and real sensor bridges."""

import math

from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster

# base -> sensor mounts, as in the Unitree Go2 URDF and model.sdf
MOUNTS = {
    "radar": ((0.28945, 0.0, -0.046825), (0.0, 2.8782, 0.0)),
    "front_camera": ((0.32715, 0.0, 0.04297), (0.0, 0.0, 0.0)),
}
OPTICAL_FRAME = "front_camera_optical"


def quaternion(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


def transform(parent, child, xyz, rpy):
    t = TransformStamped()
    t.header.frame_id, t.child_frame_id = parent, child
    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = xyz
    q = quaternion(*rpy)
    t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q
    return t


def publish_static_tf(node, base_frame, mounts=None, extra=()):
    """Returns the broadcaster; keep it alive with the node. extra: more TransformStamped, sent in the same
    message (a second sendTransform would replace the first for late subscribers)."""
    mounts = {**MOUNTS, **(mounts or {})}
    transforms = [transform(base_frame, child, xyz, rpy) for child, (xyz, rpy) in mounts.items()]
    transforms.append(transform("front_camera", OPTICAL_FRAME, (0.0, 0.0, 0.0), (-math.pi / 2, 0.0, -math.pi / 2)))
    transforms += list(extra)
    broadcaster = StaticTransformBroadcaster(node)
    broadcaster.sendTransform(transforms)
    return broadcaster
