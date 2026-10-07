"""Lidar IMU calibration: the L1's gyro and accelerometer axes on the body, against the body IMU.

Both IMUs sit rigidly on the body. The gyros measure the same rotation in different axes; the
accelerometers the same specific force, once the lidar's offset from the body IMU is accounted for.
The L1 reports its accelerometer in other axes than its gyro, so each gets its own fit.

Record while the robot turns both ways and walks (the gyro), and while it is held still in several
tilts, front-back and left-right (the accelerometer: only still moments compare, as gravity). Prints
Point-LIO's extrinsic_R and config.yaml's real_lidar.imu_acc_R, which maps the accelerometer into the
gyro's axes.
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
STILL_WINDOW_S = 0.5
STILL_GYRO = 0.05


def kabsch(a, b, proper=True):
    """R minimising |a - R b|, and the singular values of the cross-covariance (how many axes were excited).
    proper=False gives the best mirror (det -1) instead of the best rotation."""
    u, s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt)) * (1.0 if proper else -1.0)
    return u @ np.diag([1.0, 1.0, d]) @ vt, s


def smooth(t, w, window):
    """Mean of w over the last `window` seconds at each sample."""
    c = np.vstack([np.zeros((1, 3)), np.cumsum(w, axis=0)])
    i0 = np.searchsorted(t, t - window)
    n = (np.arange(len(t)) + 1 - i0)[:, None]
    return (c[1:] - c[i0]) / n


def resample(t_from, v, t_to, window):
    """v smoothed over `window` (centred) and read at t_to."""
    vs = smooth(t_from, v, window)
    return np.column_stack([np.interp(t_to, t_from - window / 2, vs[:, k]) for k in range(3)])


def fit_gyro(tb, wb, tl, wl):
    keep = (tb > tl[0] + MAX_LAG_S) & (tb < tl[-1] - MAX_LAG_S)
    tb, wb = tb[keep], wb[keep]
    best = None
    for lag in np.arange(-MAX_LAG_S, MAX_LAG_S, 0.002):
        wi = resample(tl, wl, tb + lag, 0.02)
        R, s = kabsch(wb, wi)
        rms = math.sqrt(np.mean(np.sum((wb - wi @ R.T) ** 2, axis=1)))
        if best is None or rms < best[0]:
            best = (rms, lag, R, s, wi, wb)
    return best


def fit_acc(tb, wb, ab, tl, al, lag):
    """Gravity in both accelerometers at the moments the robot is held still. The body IMU is a 50 Hz
    snapshot, so walking's foot impacts alias in it: only still moments compare. A mirror is allowed."""
    keep = (tb > tl[0] + MAX_LAG_S) & (tb < tl[-1] - MAX_LAG_S)
    tb, wb, ab = tb[keep], wb[keep], ab[keep]
    speed = np.linalg.norm(wb, axis=1)
    calm = smooth(tb, np.column_stack([speed] * 3), STILL_WINDOW_S)[:, 0]
    ab_s = resample(tb, ab, tb, STILL_WINDOW_S)
    al_s = resample(tl, al, tb + lag, STILL_WINDOW_S)
    g = np.linalg.norm(ab_s, axis=1)
    still = (calm < STILL_GYRO) & (np.abs(g - np.median(g)) < 0.3)
    if still.sum() < 50:
        return None
    a, b = ab_s[still], al_s[still]
    fits = []
    for proper in (True, False):
        bn = b / np.linalg.norm(b, axis=1, keepdims=True) * np.linalg.norm(a, axis=1, keepdims=True)
        R, _ = kabsch(a, bn, proper)
        err = np.degrees(np.arccos(np.clip(np.sum(a * (bn @ R.T), axis=1) / np.sum(a * a, axis=1), -1, 1)))
        fits.append((float(np.sqrt(np.mean(err ** 2))), R))
    d = a / np.linalg.norm(a, axis=1, keepdims=True)
    m = d.mean(axis=0) / np.linalg.norm(d.mean(axis=0))
    flat = d - np.outer(d @ m, m)
    spread = np.degrees(np.arcsin(np.clip(np.sqrt(np.linalg.eigvalsh(flat.T @ flat / len(d))[::-1][:2]), 0, 1)))
    return fits, spread, int(still.sum()), float(np.median(np.linalg.norm(b, axis=1))), float(np.median(g))


def rpy(R):
    return (math.atan2(R[2, 1], R[2, 2]), math.asin(-np.clip(R[2, 0], -1, 1)), math.atan2(R[1, 0], R[0, 0]))


def matrix(R, indent):
    return ("[" + f",\n{' ' * indent}".join(", ".join(f"{v:.4f}" for v in row) for row in R) + "]")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--base_frame", default="base")
    ap.add_argument("--lidar_frame", default="radar")
    args = ap.parse_args(remove_ros_args()[1:])

    rclpy.init()
    node = rclpy.create_node("lidar_imu_calib")
    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)
    data = {"body": [], "lidar": []}

    def record(key):
        def cb(m):
            w, a = m.angular_velocity, m.linear_acceleration
            data[key].append((Time.from_msg(m.header.stamp).nanoseconds * 1e-9, w.x, w.y, w.z, a.x, a.y, a.z))
        return cb

    node.create_subscription(Imu, "/sensors/imu", record("body"), qos_profile_sensor_data)
    node.create_subscription(Imu, "/lidar/imu", record("lidar"), qos_profile_sensor_data)
    print(f"recording {args.seconds:.0f} s: turn in place both ways and walk, then hold still a few seconds "
          "each: standing, sitting, lying, and tipped left, right, nose up and nose down...")
    start = node.get_clock().now()
    while rclpy.ok() and (node.get_clock().now() - start).nanoseconds * 1e-9 < args.seconds:
        rclpy.spin_once(node, timeout_sec=0.1)

    tf = None
    for _ in range(20):
        try:
            tf = tf_buffer.lookup_transform(args.base_frame, args.lidar_frame, Time()).transform
            break
        except TransformException:
            rclpy.spin_once(node, timeout_sec=0.5)
    node.destroy_node()
    rclpy.try_shutdown()

    if len(data["body"]) < 100 or len(data["lidar"]) < 500:
        print(f"too few samples: {len(data['body'])} /sensors/imu, {len(data['lidar'])} /lidar/imu")
        return
    if tf is None:
        print(f"no TF {args.base_frame} <- {args.lidar_frame}; is real_sensors running?")
        return
    mount = rotation(tf.rotation)
    body, lidar = (np.array(data[k]) for k in ("body", "lidar"))
    tb, wb, ab = body[:, 0], body[:, 1:4], body[:, 4:7]
    tl, wl, al = lidar[:, 0], lidar[:, 1:4], lidar[:, 4:7]

    rms, lag, R_gyro, s, wi, wbk = fit_gyro(tb, wb, tl, wl)
    scale = np.linalg.norm(wbk, axis=1).sum() / max(np.linalg.norm(wi, axis=1).sum(), 1e-9)
    print(f"samples: {len(body)} body, {len(lidar)} lidar IMU")
    print(f"gyro: lag {lag * 1000:+.0f} ms, RMS error {rms:.3f} rad/s, scale body/lidar {scale:.3f}, "
          f"motion per axis {np.round(s, 1).tolist()}")
    if s[1] < 0.05 * s[0]:
        print("  only one axis turned: walk too, or the rotation about it is a guess")
    print(f"  base <- lidar gyro rpy: [{', '.join(f'{v:.4f}' for v in rpy(R_gyro))}]")

    acc = fit_acc(tb, wb, ab, tl, al, lag)
    if acc is None:
        print("accel: too few still moments; hold each posture still for a few seconds")
        print(f"point_lio_go2.yaml:\n            extrinsic_R: {matrix(R_gyro.T @ mount, 26)}")
        return
    fits, spread, n_still, g_lidar, g_body = acc
    (err_p, R_p), (err_m, R_m) = fits
    print(f"accel: {n_still} still samples, gravity tilted {spread[0]:.1f} and {spread[1]:.1f} deg (RMS) on two axes")
    print(f"  |g| at rest: body {g_body:.2f}, lidar {g_lidar:.2f} m/s^2 (point_lio_go2.yaml acc_norm: {g_lidar:.2f})")
    print(f"  fit error: rotation {err_p:.2f} deg, mirrored {err_m:.2f} deg")
    if spread[1] < 3.0:
        print("  tilted about one axis only: also tip it sideways (front-back and left-right both needed)")
    R_acc = R_p if err_p <= err_m else R_m
    acc_to_gyro = R_gyro.T @ R_acc
    if np.linalg.det(acc_to_gyro) > 0:
        apart = math.degrees(math.acos(np.clip((np.trace(acc_to_gyro) - 1) / 2, -1, 1)))
        print(f"  accelerometer vs gyro axes: {apart:.1f} deg apart")
    else:
        print("  accelerometer is mirrored against the gyro")

    print(f"\nconfig.yaml real_lidar:\n  imu_acc_R: {matrix(acc_to_gyro, 14)}")
    print(f"point_lio_go2.yaml:\n            extrinsic_R: {matrix(R_gyro.T @ mount, 26)}")


if __name__ == "__main__":
    main()
