"""Write a checkpoint as a .qpol file for the C++ PolicyRunner.

quadruped_core's PolicyRunner stays the one place that reads checkpoints (skrl and rsl_rl
archives, params/env.yaml, layout detection, the QUADRUPED_* overrides); this takes what it
decided and the actor's weights. Format: b"QPOL", uint32 version, uint32 header length, a YAML
header, then every tensor the header lists as little-endian float32, row-major, in order.
"""
import contextlib
import io
import struct

import numpy as np
import torch
import yaml

VERSION = 1
CLOCK_KEYS = (
    "foot_offsets", "fast_offsets", "fast_swing_times", "blend_speed", "fixed_duty", "reset_to_command",
    "reset_speed", "speed_source", "stride_min", "stride_gain", "swing_time", "min_frequency",
    "max_frequency", "min_duty", "max_duty", "filter_time", "foot_x", "foot_y", "yaw_radius",
    "stand_threshold", "integrate_phase", "sync_gain",
)


def _plain(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    return v


def runner_to_qpol(runner, name="", source=""):
    """(header dict, [(tensor name, float32 array)]) for a loaded quadruped_core PolicyRunner."""
    from quadruped_core.controller.policy_runner import WTW_GAITS

    if runner.is_jit:
        raise ValueError(f"{source or name}: TorchScript policies are not supported; export the training checkpoint.")
    linears = [m for m in runner.policy.net_container if isinstance(m, torch.nn.Linear)] + [runner.policy.policy_layer]
    tensors = [
        ("obs_mean", runner.scaler.running_mean),
        ("obs_var", runner.scaler.running_variance),
    ]
    for i, layer in enumerate(linears):
        tensors += [(f"layer{i}.weight", layer.weight), (f"layer{i}.bias", layer.bias)]
    tensors = [(n, t.detach().cpu().numpy().astype("<f4")) for n, t in tensors]

    meta = {
        "name": name,
        "source": source,
        "layout": runner._obs_layout,
        "obs_dim": int(runner.obs_dim),
        "obs_dim_single": int(runner._obs_dim_single),
        "history_len": int(runner._obs_history_len),
        "action_scale": float(getattr(runner, "action_scale", 0.25)),
        "desired_qpos": _plain(runner.desired_qpos),
        "num_layers": len(linears),
    }
    if runner._obs_layout == "wtw":
        meta["wtw"] = {
            "blocks": _plain(runner._wtw_blocks),
            "gait_command": _plain(runner.gait_command),
            "gaits": {k: list(v) for k, v in WTW_GAITS.items()},
            "trained": list(runner._gait_trained),
            "limits": {int(k): _plain(v) for k, v in runner._gait_limits.items()},
            "stance": float(runner._gait_stance),
            "max_stride": None if runner._max_stride is None else float(runner._max_stride),
            "yaw_radius": float(runner._yaw_radius),
        }
    if runner._obs_layout == "unitree_clock":
        clock = {k: _plain(runner._clock_cfg[k]) for k in CLOCK_KEYS if runner._clock_cfg.get(k) is not None}
        clock["max_foot_speed"] = float(runner._max_foot_speed)
        meta["clock"] = clock
    # For policies.json entries with commands "trick": the switches and the clock (PolicyManager.load_policy).
    trick = ((runner._run_params(source, "env.yaml") or {}).get("commands") or {}).get("trick") or {} if source else {}
    meta["trick"] = {"clock_hz": float(trick.get("clock_hz") or 0.0), "switches": int(trick.get("switches") or 1)}
    meta["tensors"] = [{"name": n, "shape": list(a.shape)} for n, a in tensors]
    return meta, tensors


def write_qpol(path, meta, tensors):
    header = yaml.safe_dump(meta, sort_keys=False).encode()
    with open(path, "wb") as f:
        f.write(b"QPOL" + struct.pack("<II", VERSION, len(header)) + header)
        for _, a in tensors:
            f.write(np.ascontiguousarray(a, dtype="<f4").tobytes())


def load_runner(checkpoint, quiet=True):
    """quadruped_core's PolicyRunner for a checkpoint, as PolicyManager builds it for a switched-to policy."""
    from quadruped_core.controller.policy_runner import PolicyRunner

    out = io.StringIO()
    with contextlib.redirect_stdout(out) if quiet else contextlib.nullcontext():
        return PolicyRunner(checkpoint, robot_type="go2", obs_dim_from_env=False, verbose=False)


def export_checkpoint(checkpoint, path, name=""):
    runner = load_runner(checkpoint)
    write_qpol(path, *runner_to_qpol(runner, name=name, source=checkpoint))
    return runner
