#!/usr/bin/env python3
"""C++ kinematics and state estimator against the recorded Python ones (reference/estimator.npz), step by step."""
import argparse
import os
import sys
import time

import numpy as np

import quadruped_control as qc
from quadruped_common.kinematics import projected_gravity_b

import reference


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


KINEMATICS = ("foot_position_body", "foot_jacobian_body", "leg_chain_body")


def kinematics_inputs(rng, n):
    return [(int(rng.integers(4)), rng.uniform(-3.0, 3.0, size=3)) for _ in range(n)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default=reference.path("estimator.npz"))
    parser.add_argument("--tol", type=float, default=1e-9)
    args = parser.parse_args()
    meta, ref = reference.load(args.reference)
    rng = np.random.default_rng(meta["seed"])

    with reference.frozen(meta["files"]) as tmp:
        geometry = qc.LegGeometry.from_yaml(os.path.join(tmp, "go2_kinematics.yaml"))
    kin = qc.Kinematics(geometry)
    kin_err = 0.0
    for i, (leg, q) in enumerate(kinematics_inputs(rng, meta["kinematics_samples"])):
        for name in KINEMATICS:
            kin_err = max(kin_err, float(np.max(np.abs(ref[name][i] - getattr(kin, name)(leg, q)))))
    print(f"kinematics   {meta['kinematics_samples']} samples, max |py - cpp| = {kin_err:.2e}")

    inputs = list(scenario(meta["steps"], rng))
    if reference.Hash().add(inputs).hexdigest() != meta["inputs"]:
        print("FAIL: the generated inputs differ from the recording's (scenario() or numpy's generator changed)")
        return 1
    est = qc.StateEstimator(kin, qc.EstimatorParams(**meta["params"]))
    every = meta["covariance_every"]
    worst = dict(velocity=0.0, bias=0.0, covariance=0.0)
    for t, kw in enumerate(inputs):
        v = est.update(**kw)
        worst["velocity"] = max(worst["velocity"], float(np.max(np.abs(ref["velocity"][t] - v))))
        worst["bias"] = max(worst["bias"], float(np.max(np.abs(ref["bias"][t] - est.accel_bias))))
        if t % every == 0:
            p = ref["covariance"][t // every]
            worst["covariance"] = max(
                worst["covariance"], float(np.max(np.abs(p - est.covariance) / np.maximum(1.0, np.abs(p)))))
        if max(worst.values()) > args.tol:
            print(f"MISMATCH at step {t}: {worst}")
            return 1
    print(f"estimator    {meta['steps']} steps, {meta['relaxing']} relaxing, {meta['resets']} divergence resets")
    for k, v in worst.items():
        print(f"             max |py - cpp| {k:<10} = {v:.2e}")

    est = qc.StateEstimator(kin, qc.EstimatorParams(**meta["params"]))
    n = min(5000, len(inputs))
    start = time.perf_counter()
    for kw in inputs[:n]:
        est.update(**kw)
    print(f"timing       c++ {1e6 * (time.perf_counter() - start) / n:6.1f} us/update (python was {meta['python_us']:.0f})")

    ok = kin_err <= args.tol
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
