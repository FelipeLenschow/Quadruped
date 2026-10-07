"""Lidar IMU calibration: the L1's gyro and accelerometer axes on the body, against the body IMU.

Both IMUs sit rigidly on the body. The gyros measure the same rotation in different axes; the
accelerometers the same specific force, once the lidar's offset from the body IMU is accounted for.
The L1 reports its accelerometer in other axes than its gyro, so each gets its own fit.

Record while the robot turns both ways, walks forward, back and sideways, and sits and stands (posture
changes tilt gravity, which pins the accelerometer). Prints Point-LIO's extrinsic_R and config.yaml's
real_lidar.imu_acc_R, which maps the accelerometer into the gyro's axes.
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
ACC_WINDOW_S = 0.1


def kabsch(a, b, proper=True):
    """R minimising |a - R b|, and the singular values of the cross-covariance (how many axes were excited).
    proper=False also allows a mirror (det -1)."""
    u, s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt)) if proper else 1.0
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


def fit_acc(tb, wb, ab, tl, al, lag, r_lidar):
    """Body accelerometer moved to the lidar (a + alpha x r + w x (w x r)), against the lidar accelerometer."""
    keep = (tb > tl[0] + MAX_LAG_S) & (tb < tl[-1] - MAX_LAG_S)
    tb, wb, ab = tb[keep], wb[keep], ab[keep]
    w = resample(tb, wb, tb, ACC_WINDOW_S)
    alpha = np.gradient(w, tb, axis=0)
    at_lidar = resample(tb, ab, tb, ACC_WINDOW_S) + np.cross(alpha, r_lidar) + np.cross(w, np.cross(w, r_lidar))
    ai = resample(tl, al, tb + lag, ACC_WINDOW_S)
    R, s = kabsch(at_lidar, ai, proper=False)
    rms = math.sqrt(np.mean(np.sum((at_lidar - ai @ R.T) ** 2, axis=1)))
    scale = np.linalg.norm(at_lidar, axis=1).mean() / max(np.linalg.norm(ai, axis=1).mean(), 1e-9)
    return R, s, rms, scale, ab.mean(axis=0)


def rpy(R):
    return (math.atan2(R[2, 1], R[2, 2]), math.asin(-np.clip(R[2, 0], -1, 1)), math.atan2(R[1, 0], R[0, 0]))


def matrix(R, indent):
    return ("[" + f",\n{' ' * indent}".join(", ".join(f"{v:.4f}" for v in row) for row in R) + "]")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seconds", type=float, default=45.0)
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
    print(f"recording {args.seconds:.0f} s: turn in place both ways, walk forward, back and sideways, "
          "then sit and stand...")
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
    r_lidar = np.array([tf.translation.x, tf.translation.y, tf.translation.z])
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

    R_acc, s_acc, rms_acc, scale_acc, body_mean = fit_acc(tb, wb, ab, tl, al, lag, r_lidar)
    print(f"accel: RMS error {rms_acc:.2f} m/s^2, scale body/lidar {scale_acc:.3f}, "
          f"det {np.linalg.det(R_acc):+.0f}, spread per axis {np.round(s_acc, 0).tolist()}")
    print(f"  body accelerometer mean {np.round(body_mean, 2).tolist()} (+z up expected)")
    if s_acc[1] < 0.01 * s_acc[0]:
        print("  gravity hardly tilted: sit and stand during the recording")
    acc_to_gyro = R_gyro.T @ R_acc
    tilt = math.degrees(math.acos(np.clip((np.trace(acc_to_gyro) - 1) / 2, -1, 1))) \
        if np.linalg.det(acc_to_gyro) > 0 else float("nan")
    print(f"  accelerometer vs gyro axes: {tilt:.1f} deg apart" if tilt == tilt else
          "  accelerometer is mirrored against the gyro")

    print(f"\nconfig.yaml real_lidar:\n  imu_acc_R: {matrix(acc_to_gyro, 14)}")
    print(f"point_lio_go2.yaml:\n            extrinsic_R: {matrix(R_gyro.T @ mount, 26)}")


if __name__ == "__main__":
    main()
