"""Lidar level check: fits the floor in the lidar cloud and prints the mount angles that make it level.

Stand the robot still on flat, open ground. The floor is levelled against gravity (the IMU, through the odom
frame), so the body's own lean does not end up in the mount. Only roll and pitch come from the floor; yaw is kept.
"""

import argparse
import math

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import Imu, PointCloud2
from tf2_ros import Buffer, TransformException, TransformListener

from quadruped_perception.lidar_filter import rotation


def points(msg):
    offsets = {f.name: f.offset for f in msg.fields}
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
    pts = np.stack([raw[:, offsets[n]:offsets[n] + 4].copy().view(np.float32)[:, 0] for n in "xyz"], axis=1)
    return pts[np.isfinite(pts).all(axis=1)]


def fit_floor(pts, up, max_tilt, iters=500, tol=0.03):
    """Plane n.p + d = 0 with n near `up` and the lidar above it (d > 0), by RANSAC then least squares."""
    rng = np.random.default_rng(0)
    cos_max = math.cos(math.radians(max_tilt))
    best = None
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-6:
            continue
        n /= norm
        if n @ up < 0:
            n = -n
        d = -n @ a
        if n @ up < cos_max or d <= 0:
            continue
        inliers = np.abs(pts @ n + d) < tol
        if best is None or inliers.sum() > best.sum():
            best = inliers
    if best is None:
        return None
    floor = pts[best]
    centre = floor.mean(axis=0)
    n = np.linalg.eigh(np.cov((floor - centre).T))[1][:, 0]
    if n @ up < 0:
        n = -n
    return n, -n @ centre, int(best.sum())


def min_rotation(a, b):
    """Smallest rotation taking unit vector a to unit vector b."""
    v, c = np.cross(a, b), float(a @ b)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def rpy(R):
    """Of the two equal roll-pitch-yaw triples, the one with the least roll and yaw (the URDF's form)."""
    r, p, y = math.atan2(R[2, 1], R[2, 2]), math.asin(-np.clip(R[2, 0], -1, 1)), math.atan2(R[1, 0], R[0, 0])
    wrap = lambda a: math.atan2(math.sin(a), math.cos(a))
    alt = (wrap(r + math.pi), math.pi - p, wrap(y + math.pi))
    return min((r, p, y), alt, key=lambda v: abs(v[0]) + abs(v[2]))


def check_imu(up_lidar, accels):
    """At rest the accelerometer reads +g along up, so extrinsic_R (IMU <- lidar) maps the floor's up to it."""
    if len(accels) < 10:
        print("IMU: no /lidar/imu samples")
        return
    a = np.mean(accels, axis=0)
    if np.linalg.norm(a) < 1.0:
        print("IMU: /lidar/imu carries no acceleration (config.yaml real_lidar.imu_accel: false)")
        return
    up_imu = a / np.linalg.norm(a)
    print(f"IMU: |accel| {np.linalg.norm(a):.2f}, up in IMU frame {np.round(up_imu, 3).tolist()}, "
          f"up in lidar frame {np.round(up_lidar, 3).tolist()}")
    for name, R in (("identity", np.eye(3)), ("180 deg yaw", np.diag([-1.0, -1.0, 1.0]))):
        err = math.degrees(math.acos(np.clip(up_imu @ (R @ up_lidar), -1, 1)))
        print(f"  extrinsic_R {name}: {err:.1f} deg off")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clouds", type=int, default=10)
    ap.add_argument("--base_frame", default="base")
    ap.add_argument("--world_frame", default="odom", help="gravity-aligned frame; base assumes the body is level")
    ap.add_argument("--max_range", type=float, default=4.0)
    ap.add_argument("--max_tilt", type=float, default=45.0, help="deg the floor may be off from the current mount")
    ap.add_argument("--imu", action="store_true", help="also check the lidar IMU's rotation (Point-LIO extrinsic_R)")
    args = ap.parse_args(remove_ros_args()[1:])

    rclpy.init()
    node = rclpy.create_node("lidar_level")
    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)
    clouds = []
    accels = []
    if args.imu:
        node.create_subscription(Imu, "/lidar/imu", lambda m: accels.append(
            [m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z]), qos_profile_sensor_data)

    def keep(msg):
        if len(clouds) < args.clouds:
            clouds.append(msg)

    sub = node.create_subscription(PointCloud2, "/lidar/points", keep, qos_profile_sensor_data)
    node.get_logger().info(f"collecting {args.clouds} clouds from /lidar/points...")
    while rclpy.ok() and len(clouds) < args.clouds:
        rclpy.spin_once(node, timeout_sec=0.5)
    node.destroy_subscription(sub)

    frame = clouds[0].header.frame_id
    node.get_logger().info(f"{sum(m.width * m.height for m in clouds)} points; looking up {args.base_frame} <- {frame}")
    t = body = None
    for _ in range(20):
        try:
            t = tf_buffer.lookup_transform(args.base_frame, frame, Time()).transform
            body = tf_buffer.lookup_transform(args.world_frame, args.base_frame, Time()).transform
            break
        except TransformException:
            rclpy.spin_once(node, timeout_sec=0.5)
    if body is None:
        node.get_logger().error(f"no TF {args.world_frame} <- {args.base_frame} <- {frame}; are the driver and "
                                "real_sensors running?")
        rclpy.try_shutdown()
        return
    R_mount = rotation(t.rotation)
    R_body = rotation(body.rotation)
    R = R_body @ R_mount
    print(f"body {math.degrees(math.acos(np.clip(R_body[2, 2], -1, 1))):.1f} deg off level (IMU)")

    pts = np.concatenate([points(m) for m in clouds])
    r = np.linalg.norm(pts, axis=1)
    pts = pts[(r > 0.3) & (r < args.max_range)]
    if len(pts) > 30000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 30000, replace=False)]
    fit = fit_floor(pts, R.T @ np.array([0.0, 0.0, 1.0]), args.max_tilt)
    if fit is None:
        print(f"no floor found within {args.max_tilt} deg of the current mount; try --max_tilt 90")
    else:
        n, d, count = fit
        up_now = R @ n
        R_new = R_body.T @ min_rotation(up_now, np.array([0.0, 0.0, 1.0])) @ R
        roll, pitch, yaw = rpy(R_new)
        print(f"floor: {count} points, {math.degrees(math.acos(np.clip(up_now[2], -1, 1))):.1f} deg off level "
              f"with the current mount ({frame} -> {args.world_frame})")
        print(f"  tilts {'down' if up_now[0] > 0 else 'up'} at the front {math.degrees(math.asin(abs(up_now[0]))):.1f} deg, "
              f"{'left' if up_now[1] > 0 else 'right'} side down {math.degrees(math.asin(abs(up_now[1]))):.1f} deg")
        print(f"  lidar {d:.3f} m above the floor")
        print(f"level mount rpy: [{roll:.4f}, {pitch:.4f}, {yaw:.4f}]")
        if args.imu:
            check_imu(n, accels)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
