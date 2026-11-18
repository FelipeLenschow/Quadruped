#!/usr/bin/env python3
"""Read-only check of /lidar/fan_ranges on flat ground, before a lidar policy walks.

Listens only (publishes nothing). Records the fan for some seconds with the robot standing on a flat floor and:
  1. the rate, the size against the policy's fan, NaN and out-of-range values;
  2. the floor height under the fan fitted from the front rays (gravity from /sensors/imu), and each ray against
     the range to that flat floor, per ring and per direction;
  3. the policy run offline (not in the controller) on the robot's own joint and IMU state with a zero command,
     once with the real ranges and once with the ideal flat-floor ranges: how much the joint targets differ.

    python3 fan_flat_check.py --checkpoint <run>/model_<n>.qpol [--seconds 5]
"""
import argparse
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32MultiArray

import quadruped_control as qc

TRAIN_FAN_HEIGHT = 0.273  # lidar_fan.prior_height: the fan's height above the floor at training's stand


class Listener(Node):
    def __init__(self):
        super().__init__("fan_flat_check")
        self.scans, self.stamps = [], []
        self.imu = self.joints = self.lin_vel = None
        self.create_subscription(Float32MultiArray, "/lidar/fan_ranges", self._scan, 10)
        self.create_subscription(Imu, "/sensors/imu", lambda m: setattr(self, "imu", m), 10)
        self.create_subscription(JointState, "/sensors/joint_states", lambda m: setattr(self, "joints", m), 10)
        self.create_subscription(Vector3, "/estimator/base_lin_vel", lambda m: setattr(self, "lin_vel", m), 10)

    def _scan(self, m):
        self.scans.append(np.asarray(m.data, dtype=np.float64))
        self.stamps.append(time.monotonic())


def quat_wxyz(imu):
    o = imu.orientation
    return np.array([o.w, o.x, o.y, o.z])


def up_in_base(q):
    """The world's up vector in the base frame, from the base quaternion (w, x, y, z)."""
    w, x, y, z = q / np.linalg.norm(q)
    # Third row of R (base -> world) is the world z axis in base coordinates.
    return np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])


def flat_ranges(dirs, pos, up, fan_height, max_range):
    """Ranges from the fan origin to a flat floor fan_height below it (along `up`), max_range for rays not going down."""
    down = -(dirs @ up)
    r = np.where(down > 1e-3, fan_height / np.maximum(down, 1e-3), max_range)
    return np.minimum(r, max_range)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="the lidar policy's .qpol")
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args(remove_ros_args()[1:])

    spec = qc.PolicySpec.from_file(args.checkpoint)
    fan = spec.lidar
    dep, az = np.asarray(fan.depressions), np.asarray(fan.azimuths)
    d, a = np.meshgrid(np.radians(dep), np.radians(az), indexing="ij")
    dirs = np.stack([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), -np.sin(d)], -1).reshape(-1, 3)
    ring = np.repeat(np.arange(len(dep)), len(az))
    azr = np.tile(az, len(dep))
    pos = np.asarray(fan.pos)

    rclpy.init()
    node = Listener()
    end = time.monotonic() + args.seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
    scans, imu, joints, lin_vel = node.scans, node.imu, node.joints, node.lin_vel
    node.destroy_node()
    rclpy.shutdown()

    print(f"== 1. Topic ({spec.name}: {len(dep)} x {len(az)} = {fan.rays} rays, max {fan.max_range} m)")
    if not scans:
        print("FAIL: no /lidar/fan_ranges received.")
        return 1
    rate = (len(scans) - 1) / (node.stamps[-1] - node.stamps[0]) if len(scans) > 1 else 0.0
    gaps = np.diff(node.stamps) if len(scans) > 1 else np.array([0.0])
    sizes = {len(s) for s in scans}
    print(f"messages {len(scans)}, rate {rate:.1f} Hz, largest gap {gaps.max() * 1000:.0f} ms, sizes {sorted(sizes)}")
    if sizes != {fan.rays}:
        print(f"FAIL: size differs from the policy's {fan.rays} rays (the controller holds the default pose).")
        return 1
    S = np.stack(scans)
    bad = ~np.isfinite(S) | (S < 0.0) | (S > fan.max_range + 1e-3)
    print(f"NaN / negative / over {fan.max_range} m: {int(bad.sum())} values")
    r = np.nanmedian(np.where(bad, np.nan, S), axis=0)
    jitter = np.nanstd(np.where(bad, np.nan, S), axis=0)
    print(f"jitter over time (std per ray): median {np.median(jitter) * 100:.1f} cm, 95% {np.percentile(jitter, 95) * 100:.1f} cm")

    if imu is None:
        print("FAIL: no /sensors/imu.")
        return 1
    q = quat_wxyz(imu)
    up = up_in_base(q)
    tilt = np.degrees(np.arccos(np.clip(up[2], -1, 1)))
    down = -(dirs @ up)
    print(f"\n== 2. Against a flat floor (base tilt {tilt:.1f} deg)")
    front = (np.abs(azr) <= 60) & (r < fan.max_range - 0.01) & (down > 0.05)
    h_rays = r * down + 0.0  # fan height above the floor seen by each ray
    h = float(np.median(h_rays[front]))
    print(f"floor under the fan from the front rays: {h:.3f} m (training stand {TRAIN_FAN_HEIGHT:.3f} m, "
          f"{(h - TRAIN_FAN_HEIGHT) * 100:+.1f} cm); spread of the front rays {np.std(h_rays[front]) * 100:.1f} cm")
    exp = flat_ranges(dirs, pos, up, h, fan.max_range)
    err = r - exp
    print(f"all rays: median |error| {np.median(np.abs(err)) * 100:.1f} cm, 90% {np.percentile(np.abs(err), 90) * 100:.1f} cm, "
          f"> 5 cm {int((np.abs(err) > 0.05).sum())}, > 20 cm {int((np.abs(err) > 0.20).sum())} of {fan.rays}")
    print(f"rays at max range: real {int((r >= fan.max_range - 0.01).sum())}, flat floor {int((exp >= fan.max_range - 0.01).sum())}")
    print("\nper ring (depression deg): median error / median |error| cm, rays > 20 cm")
    for i, dv in enumerate(dep):
        m = ring == i
        print(f"  ring {i:2d} {dv:5.1f}: {np.median(err[m]) * 100:+6.1f} / {np.median(np.abs(err[m])) * 100:5.1f}, "
              f"{int((np.abs(err[m]) > 0.20).sum()):2d}")
    print("\nper direction (azimuth deg, 0 = ahead, + = left): median error / median |error| cm, rays > 20 cm")
    sectors = [("ahead", -30, 30), ("front left", 30, 90), ("rear left", 90, 150), ("behind", 150, 180),
               ("rear right", -150, -90), ("front right", -90, -30)]
    wrapped = (azr + 180.0) % 360.0 - 180.0
    for name, lo, hi in sectors:
        m = (np.abs(wrapped) >= 150) if name == "behind" else (wrapped >= lo) & (wrapped < hi)
        print(f"  {name:<11} {np.median(err[m]) * 100:+6.1f} / {np.median(np.abs(err[m])) * 100:5.1f}, "
              f"{int((np.abs(err[m]) > 0.20).sum()):3d}")
    worst = np.argsort(-np.abs(err))[:8]
    print("\nworst rays (ring, azimuth deg, real m, flat m):")
    for k in worst:
        print(f"  ring {ring[k]:2d} az {azr[k]:+6.0f}: {r[k]:.3f} vs {exp[k]:.3f} ({err[k] * 100:+.0f} cm)")

    # The floor plane the lidar sees (robust fit to the ray end points), against the IMU's level: a difference is a
    # lidar mount angle error or a floor that is not level. Points well above the plane are objects.
    pts = pos + r[:, None] * dirs
    keep = (r < fan.max_range - 0.01) & (down > 0.05)
    for _ in range(10):
        c = pts[keep].mean(0)
        n = np.linalg.svd(pts[keep] - c)[2][-1]
        n = n if n @ up > 0 else -n
        res = (pts - c) @ n
        keep = (r < fan.max_range - 0.01) & (np.abs(res) < 0.03)
    off = np.degrees(np.arccos(np.clip(n @ up, -1, 1)))
    # The plane's tilt from the IMU's level, as pitch (nose) and roll (left side) in the base frame.
    side = np.cross(up, n)
    print(f"\nfloor plane from the lidar: {off:.2f} deg off the IMU's level "
          f"(about x {np.degrees(side[0]):+.2f}, about y {np.degrees(side[1]):+.2f} deg), "
          f"fan {float((pos - c) @ n):.3f} m above it, {int(keep.sum())} rays on it within 3 cm")
    lidar_tilt = np.degrees(np.arccos(np.clip(n[2], -1, 1)))
    print(f"base tilt to that floor, from the lidar: {lidar_tilt:.2f} deg (normal in base {np.round(n, 4).tolist()}); "
          f"base tilt to gravity, from the IMU: {tilt:.2f} deg (up in base {np.round(up, 4).tolist()})")
    above = res > 0.05
    print(f"rays ending more than 5 cm above that plane (objects): {int(above.sum())}")
    if above.any():
        for k in np.argsort(-res)[:6]:
            if res[k] <= 0.05:
                break
            print(f"  ring {ring[k]:2d} az {azr[k]:+6.0f}: {r[k]:.2f} m out, {res[k] * 100:.0f} cm above the floor")
    print(f"after the plane fit: median |error| {np.median(np.abs(res[keep | ~above])) * 100:.1f} cm, "
          f"rays more than 5 cm below the plane {int((res < -0.05).sum())}")

    print("\n== 3. The policy offline: real ranges against the ideal flat floor (zero command)")
    if joints is None:
        print("skipped: no /sensors/joint_states.")
        return 0
    qj = np.asarray(joints.position[:12])
    dq = np.asarray(joints.velocity[:12]) if len(joints.velocity) >= 12 else np.zeros(12)
    gyro = np.array([imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z])
    v = np.array([lin_vel.x, lin_vel.y, lin_vel.z]) if lin_vel else np.zeros(3)
    ideal = flat_ranges(dirs, pos, up, h, fan.max_range)
    train = flat_ranges(dirs, pos, up, TRAIN_FAN_HEIGHT, fan.max_range)
    out = {}
    for label, scan in (("real", r), ("flat, same height", ideal), ("flat, training height", train)):
        runner = qc.PolicyRunner(spec)
        for _ in range(100):  # fill the history with the same still state
            act = None
            for n_cmd in (3, 4, 5, 6):
                try:
                    act = np.asarray(runner.infer(q, v, gyro, qj, dq, np.zeros(n_cmd), 0.02, scan.astype(np.float32)))
                    break
                except Exception as e:  # the command size is the policy's
                    last = e
            if act is None:
                print(f"skipped: the runner refused the inputs ({last}).")
                return 0
        out[label] = act
    scale = np.asarray(spec.action_scale, dtype=np.float64)
    out = {k: o * scale for k, o in out.items()}  # policy output to radians of joint target
    base = out["flat, same height"]
    for label in ("real", "flat, training height"):
        diff = np.abs(out[label] - base)
        print(f"{label:<22} vs flat, same height: joint targets differ by mean {diff.mean():.3f}, max {diff.max():.3f} rad")
    print("(references, narrow flight in MuJoCo: 0.004 rad for the training noise, 0.079 rad for the L1's hidden rays)")
    print("target offsets (rad) with the real ranges:", np.round(out["real"], 3).tolist())
    print("measured joints now:                     ", np.round(qj, 3).tolist())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
