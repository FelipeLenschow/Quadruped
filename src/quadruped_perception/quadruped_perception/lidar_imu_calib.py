"""Lidar IMU calibration: the L1's IMU rotation on the body, from its gyro against the body IMU's gyro.

Both IMUs sit rigidly on the body, so they measure the same rotation in different axes. Record while the
robot turns both ways and walks (the gait's roll and pitch sway gives the other axes), then this prints
Point-LIO's extrinsic_R (lidar -> IMU) from that rotation and the base -> radar mount in TF.
"""

import argparse
import math

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import Imu
from tf2_ros import Buffer, TransformException, TransformListener

from quadruped_perception.lidar_filter import rotation

MAX_LAG_S = 0.2


def kabsch(a, b):
    """R minimising |a - R b|, and the singular values of the cross-covariance (how many axes were excited)."""
    u, s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, d]) @ vt, s


def smooth(t, w, window):
    """Mean of w over the last `window` seconds at each sample, to bring 250 Hz down to the body's 50 Hz."""
    c = np.vstack([np.zeros((1, 3)), np.cumsum(w, axis=0)])
    i0 = np.searchsorted(t, t - window)
    n = (np.arange(len(t)) + 1 - i0)[:, None]
    return (c[1:] - c[i0]) / n


def fit(body, lidar):
    tb, wb = body
    tl, wl = lidar[0] - 0.01, smooth(lidar[0], lidar[1], 0.02)
    keep = (tb > tl[0] + MAX_LAG_S) & (tb < tl[-1] - MAX_LAG_S)
    tb, wb = tb[keep], wb[keep]
    best = None
    for lag in np.arange(-MAX_LAG_S, MAX_LAG_S, 0.002):
        wi = np.column_stack([np.interp(tb + lag, tl, wl[:, k]) for k in range(3)])
        R, s = kabsch(wb, wi)
        rms = math.sqrt(np.mean(np.sum((wb - wi @ R.T) ** 2, axis=1)))
        if best is None or rms < best[0]:
            best = (rms, lag, R, s, wi)
    return best + (wb,)


def rpy(R):
    return (math.atan2(R[2, 1], R[2, 2]), math.asin(-np.clip(R[2, 0], -1, 1)), math.atan2(R[1, 0], R[0, 0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--base_frame", default="base")
    ap.add_argument("--lidar_frame", default="radar")
    args = ap.parse_args(remove_ros_args()[1:])

    rclpy.init()
    node = rclpy.create_node("lidar_imu_calib")
    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)
    data = {"body": [], "lidar": [], "acc": []}

    def record(key):
        def cb(m):
            t = Time.from_msg(m.header.stamp).nanoseconds * 1e-9
            w = m.angular_velocity
            data[key].append((t, w.x, w.y, w.z))
            if key == "lidar":
                a = m.linear_acceleration
                data["acc"].append(math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z))
        return cb

    node.create_subscription(Imu, "/sensors/imu", record("body"), qos_profile_sensor_data)
    node.create_subscription(Imu, "/lidar/imu", record("lidar"), qos_profile_sensor_data)
    print(f"recording {args.seconds:.0f} s: turn in place both ways, then walk forward and back...")
    start = node.get_clock().now()
    while rclpy.ok() and (node.get_clock().now() - start).nanoseconds * 1e-9 < args.seconds:
        rclpy.spin_once(node, timeout_sec=0.1)

    mount = None
    for _ in range(20):
        try:
            mount = rotation(tf_buffer.lookup_transform(args.base_frame, args.lidar_frame, Time()).transform.rotation)
            break
        except TransformException:
            rclpy.spin_once(node, timeout_sec=0.5)
    node.destroy_node()
    rclpy.try_shutdown()

    if len(data["body"]) < 100 or len(data["lidar"]) < 500:
        print(f"too few samples: {len(data['body'])} /sensors/imu, {len(data['lidar'])} /lidar/imu")
        return
    body, lidar = (np.array(data[k]) for k in ("body", "lidar"))
    rms, lag, R_base_imu, s, wi, wb = fit((body[:, 0], body[:, 1:]), (lidar[:, 0], lidar[:, 1:]))

    scale = np.linalg.norm(wb, axis=1).sum() / max(np.linalg.norm(wi, axis=1).sum(), 1e-9)
    print(f"samples: {len(body)} body, {len(lidar)} lidar IMU; lidar IMU |accel| mean {np.mean(data['acc']):.2f}")
    print(f"lidar IMU lag {lag * 1000:+.0f} ms, gyro RMS error {rms:.3f} rad/s, scale body/lidar {scale:.3f}")
    print(f"motion per axis (singular values): {np.round(s, 1).tolist()}")
    if s[1] < 0.05 * s[0]:
        print("  only one axis moved: walk too, or the rotation about it is a guess")
    r, p, y = rpy(R_base_imu)
    print(f"base <- lidar IMU rpy: [{r:.4f}, {p:.4f}, {y:.4f}]")
    if mount is None:
        print(f"no TF {args.base_frame} <- {args.lidar_frame}; is real_sensors running?")
        return
    ext = R_base_imu.T @ mount
    rows = ",\n                          ".join(", ".join(f"{v:.4f}" for v in row) for row in ext)
    print(f"point_lio_go2.yaml:\n            extrinsic_R: [{rows}]")


if __name__ == "__main__":
    main()
