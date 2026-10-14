#!/usr/bin/env python3
"""C++ kinematics and state estimator against the Python ones in quadruped_core, step by step."""
import argparse
import sys
import time

import numpy as np

import quadruped_control as qc
from quadruped_core import paths
from quadruped_core.config_loader import load_config
from quadruped_core.telemetry.estimator import StateEstimator as PyEstimator, projected_gravity_b
from quadruped_core.telemetry.kinematics import Go2Kinematics


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def scenario(steps, rng):
    q_home = np.array([0.0] * 4 + [0.9] * 4 + [-1.8] * 4)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    contact = np.ones(4)
    pattern_left = 0
    for t in range(steps):
        if pattern_left == 0:
            kind = rng.choice(["trot", "stand", "flight", "random"], p=[0.4, 0.25, 0.15, 0.2])
            pattern_left = int(rng.integers(5, 60))
        pattern_left -= 1
        if kind == "trot":
            contact = np.array([1, 0, 0, 1.0]) if (t // 10) % 2 else np.array([0, 1, 1, 0.0])
        elif kind == "stand":
            contact = np.ones(4)
        elif kind == "flight":
            contact = np.zeros(4)
        else:
            contact = (rng.random(4) > 0.5).astype(float)

        tilt = 0.3 if rng.random() < 0.01 else 0.02
        axis = rng.normal(size=3)
        angle = tilt * rng.normal()
        dq_rot = np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis / np.linalg.norm(axis)])
        quat = quat_mul(quat, dq_rot)
        quat /= np.linalg.norm(quat)

        accel = -projected_gravity_b(quat) + rng.normal(scale=1.0, size=3)
        joint_pos = q_home + rng.normal(scale=0.3, size=12)
        joint_vel = rng.normal(scale=2.0, size=12)
        if rng.random() < 0.005:
            joint_vel *= 40.0
        gyro = rng.normal(scale=1.0, size=3)

        no_joints = rng.random() < 0.02
        yield dict(
            quat_wxyz=quat.copy(),
            accel_body=accel,
            feet_contact=contact.copy(),
            joint_pos=None if no_joints else joint_pos,
            joint_vel=None if no_joints else joint_vel,
            gyro_body=None if rng.random() < 0.02 else gyro,
        )


def check_kinematics(py_kin, cpp_kin, rng, n):
    worst = 0.0
    for _ in range(n):
        leg = int(rng.integers(4))
        q = rng.uniform(-3.0, 3.0, size=3)
        for name in ("foot_position_body", "foot_jacobian_body", "leg_chain_body"):
            a = getattr(py_kin, name)(leg, q)
            b = getattr(cpp_kin, name)(leg, q)
            worst = max(worst, float(np.max(np.abs(a - b))))
    return worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1e-9)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    geometry = qc.LegGeometry.from_yaml(paths.description("config", "go2_kinematics.yaml"))
    cpp_kin = qc.Kinematics(geometry)
    kin_err = check_kinematics(Go2Kinematics(), cpp_kin, rng, 20000)
    print(f"kinematics   max |py - cpp| = {kin_err:.2e}")

    est_cfg = load_config().get("state_estimator", {}) or {}
    kal = est_cfg.get("kalman", {}) or {}
    params = dict(
        dt=float(est_cfg.get("dt", 0.02)),
        q_vel=float(kal.get("process_noise_vel", 0.01)),
        q_bias=float(kal.get("process_noise_bias", 1e-3)),
        r_meas=float(kal.get("measurement_noise", 0.05)),
    )
    py_est = PyEstimator(**params)
    cpp_est = qc.StateEstimator(cpp_kin, qc.EstimatorParams(**params))

    inputs = list(scenario(args.steps, rng))
    worst = dict(velocity=0.0, bias=0.0, covariance=0.0)
    guard_trips = relax_steps = 0
    for t, kw in enumerate(inputs):
        v_py = py_est.update(**kw)
        v_cpp = cpp_est.update(**kw)
        worst["velocity"] = max(worst["velocity"], float(np.max(np.abs(v_py - v_cpp))))
        worst["bias"] = max(worst["bias"], float(np.max(np.abs(py_est.accel_bias - cpp_est.accel_bias))))
        p_py = py_est.covariance
        worst["covariance"] = max(
            worst["covariance"],
            float(np.max(np.abs(p_py - cpp_est.covariance) / np.maximum(1.0, np.abs(p_py)))))
        guard_trips += int(np.all(v_py == 0.0) and np.all(p_py[:3, :3] == np.eye(3)))
        relax_steps += int(py_est._no_contact_time > py_est.no_contact_relax_time)
        if max(worst.values()) > args.tol:
            print(f"MISMATCH at step {t}: {worst}")
            return 1
    print(f"estimator    {args.steps} steps, {relax_steps} relaxing, {guard_trips} divergence resets")
    for k, v in worst.items():
        print(f"             max |py - cpp| {k:<10} = {v:.2e}")

    n = min(5000, len(inputs))
    for name, est in (("python", PyEstimator(**params)), ("c++", qc.StateEstimator(cpp_kin, qc.EstimatorParams(**params)))):
        start = time.perf_counter()
        for kw in inputs[:n]:
            est.update(**kw)
        print(f"timing       {name:<6} {1e6 * (time.perf_counter() - start) / n:8.1f} us/update")

    ok = kin_err <= args.tol
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
