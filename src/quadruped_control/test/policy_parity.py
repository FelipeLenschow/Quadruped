#!/usr/bin/env python3
"""C++ PolicyRunner against quadruped_core's, step by step, for every policy in policies.json."""
import argparse
import os
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np

import quadruped_control as qc
from quadruped_control.export import load_runner, runner_to_qpol, write_qpol
from quadruped_core import paths
from quadruped_core.config_loader import load_policies
from quadruped_core.controller.policy_runner import WTW_GAITS


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def events(steps, rng, qpos, wtw):
    """Per step: state, commands, and what to do before stepping (reset, gait change)."""
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    cmd = np.zeros(4)
    left = 0
    for t in range(steps):
        action = None
        if left == 0:
            left = int(rng.integers(10, 150))
            kind = rng.choice(["stand", "walk", "turn", "tiny"], p=[0.3, 0.4, 0.2, 0.1])
            if kind == "stand":
                cmd = np.zeros(4)
            elif kind == "walk":
                cmd = np.array([rng.uniform(-1, 2), rng.uniform(-0.6, 0.6), rng.uniform(-1, 1), rng.uniform(-0.1, 0.1)])
            elif kind == "turn":
                cmd = np.array([0.0, 0.0, rng.uniform(-2, 2), 0.0])
            else:
                cmd = rng.normal(scale=0.01, size=4)
            if rng.random() < 0.1:
                action = ("reset",)
            elif wtw and rng.random() < 0.3:
                if rng.random() < 0.5:
                    action = ("gait", str(rng.choice(list(WTW_GAITS))),
                              {"frequency": float(rng.uniform(0.0, 5.0)), "duty": float(rng.uniform(0.2, 0.9))})
                else:
                    action = ("vector", rng.uniform(-1.0, 5.0, size=int(rng.integers(3, 10))).astype(np.float32))
        left -= 1
        axis = rng.normal(size=3)
        angle = 0.02 * rng.normal()
        quat = quat_mul(quat, np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis / np.linalg.norm(axis)]))
        quat /= np.linalg.norm(quat)
        state = dict(
            quat=quat.copy(),
            lin_vel=cmd[:3] * 0.8 + rng.normal(scale=0.1, size=3),
            gyro=rng.normal(scale=0.5, size=3),
            q=qpos + rng.normal(scale=0.2, size=12),
            dq=rng.normal(scale=2.0, size=12),
        )
        yield action, state, cmd.copy()


def rel_error(a, b):
    """Max |a - b| relative to |a| past 1 (torch and Eigen round float32 matmuls differently); equal non-finite values match."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    with np.errstate(invalid="ignore"):
        err = np.where(same, 0.0, np.abs(a - b) / np.maximum(1.0, np.abs(a)))
    return float(np.nan_to_num(err, nan=np.inf).max())


def py_state(s):
    return SimpleNamespace(
        imu=SimpleNamespace(quaternion=s["quat"].tolist(), gyroscope=s["gyro"].tolist()),
        base_lin_vel=s["lin_vel"].tolist(),
        motorState=[SimpleNamespace(q=float(s["q"][i]), dq=float(s["dq"][i])) for i in range(12)],
    )


def run(name, path, steps, seed, obs_tol, act_tol):
    py = load_runner(path)
    with tempfile.TemporaryDirectory() as tmp:
        qpol = os.path.join(tmp, f"{name}.qpol")
        write_qpol(qpol, *runner_to_qpol(py, name=name, source=path))
        cpp = qc.PolicyRunner.from_file(qpol)

    seen = {}
    get_action = py.get_action
    py.get_action = lambda obs: (seen.__setitem__("obs", obs), get_action(obs))[1]

    rng = np.random.default_rng(seed)
    worst_obs = worst_act = 0.0
    runaway = 0
    t_py = t_cpp = 0.0
    wtw = py._obs_layout == "wtw"
    for t, (action, s, cmd) in enumerate(events(steps, rng, py.desired_qpos.astype(np.float64), wtw)):
        if action and action[0] == "reset":
            py.reset_history()
            cpp.reset_history()
        elif action and action[0] == "gait":
            py.set_gait(gait=action[1], **action[2])
            cpp.set_gait(action[1], **action[2])
        elif action and action[0] == "vector":
            py.set_gait_vector(action[1])
            cpp.set_gait_vector(action[1])
        if wtw:
            assert np.array_equal(py.gait_command, cpp.gait_command), (t, py.gait_command, cpp.gait_command)

        state = py_state(s)
        start = time.perf_counter()
        a_py, _ = py.infer(state, cmd, py.desired_qpos, list(range(12)), dt=0.02)
        t_py += time.perf_counter() - start
        start = time.perf_counter()
        a_cpp = cpp.infer(s["quat"], s["lin_vel"], s["gyro"], s["q"], s["dq"], cmd, 0.02)
        t_cpp += time.perf_counter() - start

        worst_obs = max(worst_obs, rel_error(seen["obs"], cpp.last_obs))
        worst_act = max(worst_act, rel_error(a_py, a_cpp))
        # Both continue from the same actions, so float32 rounding cannot compound through last_actions.
        py.last_actions[:] = a_cpp
        # Random states drive the action feedback far past any real target (to inf); restart from there.
        if not np.all(np.abs(a_py) < 100.0):
            py.reset_history()
            cpp.reset_history()
            runaway += 1
        if worst_obs > obs_tol or worst_act > act_tol:
            i = int(np.nanargmax(np.abs(seen["obs"] - cpp.last_obs) / np.maximum(1.0, np.abs(seen["obs"]))))
            print(f"{name:<8} MISMATCH at step {t}: obs {worst_obs:.2e} (index {i}: py {seen['obs'][i]} cpp {cpp.last_obs[i]}), "
                  f"actions {worst_act:.2e}")
            return False
    print(f"{name:<8} {py._obs_layout:<14} {steps} steps  max |py - cpp| obs {worst_obs:.1e}  actions {worst_act:.1e}  "
          f"python {1e6 * t_py / steps:6.0f} us  c++ {1e6 * t_cpp / steps:5.0f} us  ({runaway} runaway resets)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("policies", nargs="*", help="policies.json names (default: all)")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obs-tol", type=float, default=1e-4)
    parser.add_argument("--act-tol", type=float, default=2e-4)
    args = parser.parse_args()

    catalog = load_policies().get("policies") or {}
    ok = bool(catalog)
    for name in args.policies or list(catalog):
        path = catalog[name]["path"]
        path = path if os.path.isabs(path) else paths.repo(path)
        if not os.path.exists(path):
            print(f"{name:<8} skipped, no checkpoint at {path}")
            continue
        ok = run(name, path, args.steps, args.seed, args.obs_tol, args.act_tol) and ok
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
