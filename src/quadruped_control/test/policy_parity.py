#!/usr/bin/env python3
"""C++ PolicyRunner against the recorded Python one (reference/policy.npz), step by step, for every recorded policy."""
import argparse
import os
import sys
import tempfile
import time

import numpy as np

import quadruped_control as qc
from quadruped_control.export import WTW_GAITS, export_checkpoint
from quadruped_common.config_loader import load_policies

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


def obs_index(obs_dim, single):
    """The observation entries recorded: all of them, or the newest and oldest frame of a long history."""
    if obs_dim <= 2 * single:
        return np.arange(obs_dim)
    return np.concatenate([np.arange(single), np.arange(obs_dim - single, obs_dim)])


def apply(runner, action):
    if action and action[0] == "reset":
        runner.reset_history()
    elif action and action[0] == "gait":
        runner.set_gait(action[1], **action[2])
    elif action and action[0] == "vector":
        runner.set_gait_vector(action[1])


def run(name, meta, ref, live_catalog, obs_tol, act_tol):
    path = reference.find_checkpoint(meta["path"], meta["sha1"], live_catalog)
    if path is None:
        print(f"{name:<8} skipped, the recorded checkpoint ({meta['path']}) is gone")
        return True
    with tempfile.TemporaryDirectory() as tmp:
        qpol = os.path.join(tmp, f"{name}.qpol")
        export_checkpoint(path, qpol, name=name)
        cpp = qc.PolicyRunner.from_file(qpol)

    obs, actions, forced = ref[f"{name}/obs"], ref[f"{name}/actions"], ref[f"{name}/forced"]
    idx = obs_index(cpp.spec.obs_dim, cpp.spec.obs_dim_single)
    wtw = meta["layout"] == "wtw"
    rng = np.random.default_rng(meta["seed"])
    inputs = list(events(meta["steps"], rng, np.asarray(meta["qpos"]), wtw))
    if reference.Hash().add(inputs).hexdigest() != meta["inputs"]:
        print(f"{name:<8} FAIL: the generated inputs differ from the recording's (events() or numpy's generator changed)")
        return False

    worst_obs = worst_act = 0.0
    runaway = 0
    t_cpp = 0.0
    for t, (action, s, cmd) in enumerate(inputs):
        apply(cpp, action)
        if wtw and not np.array_equal(ref[f"{name}/gait"][t], cpp.gait_command):
            print(f"{name:<8} MISMATCH at step {t}: gait command py {ref[f'{name}/gait'][t]} cpp {cpp.gait_command}")
            return False
        start = time.perf_counter()
        a_cpp = cpp.infer(s["quat"], s["lin_vel"], s["gyro"], s["q"], s["dq"], cmd, 0.02)
        t_cpp += time.perf_counter() - start

        worst_obs = max(worst_obs, rel_error(obs[t], cpp.last_obs[idx]))
        worst_act = max(worst_act, rel_error(actions[t], a_cpp))
        # Continue from the actions the recording continued from, so float32 rounding cannot compound.
        cpp.last_actions = forced[t]
        # Random states drive the action feedback far past any real target (to inf); restart from there.
        if not np.all(np.abs(actions[t]) < 100.0):
            cpp.reset_history()
            runaway += 1
        if worst_obs > obs_tol or worst_act > act_tol:
            err = np.abs(obs[t] - cpp.last_obs[idx]) / np.maximum(1.0, np.abs(obs[t]))
            i = int(np.nanargmax(err))
            print(f"{name:<8} MISMATCH at step {t}: obs {worst_obs:.2e} (index {idx[i]}: py {obs[t][i]} cpp "
                  f"{cpp.last_obs[idx[i]]}), actions {worst_act:.2e}")
            return False
    steps = meta["steps"]
    print(f"{name:<8} {meta['layout']:<14} {steps} steps  max |py - cpp| obs {worst_obs:.1e}  actions {worst_act:.1e}  "
          f"c++ {1e6 * t_cpp / steps:5.0f} us (python was {meta['python_us']:.0f})  ({runaway} runaway resets)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("policies", nargs="*", help="recorded policy names (default: all)")
    parser.add_argument("--reference", default=reference.path("policy.npz"))
    parser.add_argument("--obs-tol", type=float, default=1e-4)
    parser.add_argument("--act-tol", type=float, default=2e-4)
    args = parser.parse_args()

    meta, ref = reference.load(args.reference)
    live = load_policies().get("policies") or {}
    ok = True
    for name in args.policies or list(meta["policies"]):
        if name not in meta["policies"]:
            print(f"{name:<8} FAIL: not in the recording ({', '.join(meta['policies'])})")
            ok = False
            continue
        ok = run(name, meta["policies"][name], ref, live, args.obs_tol, args.act_tol) and ok
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
