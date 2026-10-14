"""Training checkpoints (skrl or rsl_rl) as .qpol files for the C++ PolicyRunner.

What the archive does not say (observation layout, history, gait and clock settings) comes from the
run's params/env.yaml and params/agent.yaml. Format: b"QPOL", uint32 version, uint32 header length, a
YAML header, then every tensor the header lists as little-endian float32, row-major, in order.
"""
import os
import struct
import zipfile

import numpy as np
import torch
import yaml

from quadruped_common.robot_defaults import DEFAULT_STANCE_QPOS

VERSION = 1
ACTION_SCALE = 0.25
SINGLE_STEP = 49  # our own policies' observation width per step

# Walk These Ways gaits: phase / offset / bound, feet FL, FR, RL, RR (walk_these_ways mdp/commands/gait_command.py).
WTW_GAITS = {
    "trot": (0.5, 0.0, 0.0),
    "pace": (0.0, 0.0, 0.5),
    "bound": (0.0, 0.5, 0.0),
    "pronk": (0.0, 0.0, 0.0),
    "walk": (0.0, 0.75, 0.5),
}
# WTW per-step blocks ahead of the gait command (8 or 9 wide) and the 4-wide clock.
WTW_FIXED_BLOCKS = (3, 3, 3, 3, 12, 12, 12)
WTW_CLOCK = 4
CLOCK_KEYS = (
    "foot_offsets", "fast_offsets", "fast_swing_times", "blend_speed", "fixed_duty", "reset_to_command",
    "reset_speed", "speed_source", "stride_min", "stride_gain", "swing_time", "min_frequency",
    "max_frequency", "min_duty", "max_duty", "filter_time", "foot_x", "foot_y", "yaw_radius",
    "stand_threshold", "integrate_phase", "sync_gain",
)


def _warn(msg):
    print(f"[export] WARNING: {msg}", flush=True)


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


class _TolerantLoader(yaml.SafeLoader):
    """Isaac Lab tags its dumps with !!python/...; keep tuples and mappings, drop the rest."""


_TolerantLoader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: loader.construct_sequence(node))
_TolerantLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/",
    lambda loader, suffix, node: loader.construct_mapping(node, deep=True) if isinstance(node, yaml.MappingNode) else None)


def run_params(checkpoint, name="agent.yaml"):
    """The run's params/<name>: one level up for skrl (<run>/checkpoints/x.pt), beside it for rsl_rl. None if absent."""
    here = os.path.dirname(os.path.abspath(checkpoint))
    for base in (os.path.dirname(here), here):
        path = os.path.join(base, "params", name)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return yaml.load(f, Loader=_TolerantLoader) or {}
            except Exception as e:
                _warn(f"could not read {path} ({e}).")
                return None
    return None


def _is_torchscript(path):
    """TorchScript archives carry a code/ directory; state dicts do not (rsl_rl exports policy.pt as TorchScript)."""
    if path.endswith(".jit"):
        return True
    try:
        with zipfile.ZipFile(path) as z:
            return any("/code/" in n for n in z.namelist())
    except (zipfile.BadZipFile, OSError):
        return False


def _rsl_rl_actor(data, checkpoint):
    # rsl_rl >= 4 saves actor_state_dict with mlp.* / obs_normalizer.* keys.
    if "actor_state_dict" in data and "model_state_dict" not in data:
        state = {}
        for k, v in data["actor_state_dict"].items():
            if k.startswith("mlp."):
                state["actor." + k[4:]] = v
            elif k.startswith("obs_normalizer."):
                state["actor_obs_normalizer." + k[len("obs_normalizer."):].lstrip("_")] = v
    else:
        state = data["model_state_dict"]
    index = sorted(int(k.split(".")[1]) for k in state if k.startswith("actor.") and k.endswith(".weight"))
    if not index:
        raise ValueError(f"{checkpoint}: an rsl_rl checkpoint with no 'actor.*' weights.")
    linears = [(state[f"actor.{i}.weight"], state[f"actor.{i}.bias"]) for i in index]
    activation = ((run_params(checkpoint) or {}).get("policy") or {}).get("activation")
    if activation and str(activation).lower() != "elu":
        _warn(f"{checkpoint}: trained with '{activation}', but the C++ MLP is ELU; it will not behave.")
    # The normalizer: inside the model (rsl_rl >= 2.3) or beside it; the critic's has another width.
    norm = data.get("obs_norm_state_dict") or {
        k.split("actor_obs_normalizer.")[-1]: v for k, v in state.items() if k.startswith("actor_obs_normalizer.")}
    if norm and "mean" in norm:
        return linears, norm["mean"], norm["var"]
    return linears, None, None  # empirical_normalization: false trains on raw observations


def _skrl_actor(data, checkpoint):
    policy = {k.split("_model.")[-1]: v for k, v in (data.get("policy") or {}).items() if "net" in k or "policy" in k}
    hidden = sorted(int(k.split(".")[1]) for k in policy if k.startswith("net_container.") and k.endswith(".weight"))
    linears = [(policy[f"net_container.{i}.weight"], policy[f"net_container.{i}.bias"]) for i in hidden]
    # models.separate: False has a policy_layer head; True ends net_container with the head instead.
    if "policy_layer.weight" in policy:
        linears.append((policy["policy_layer.weight"], policy["policy_layer.bias"]))
    if not linears:
        raise ValueError(f"{checkpoint}: no policy weights (keys: {sorted(data)}).")
    # skrl renamed state_preprocessor to observation_preprocessor in 2.1.0.
    scaler = data.get("observation_preprocessor") or data.get("state_preprocessor") or data.get("running_standard_scaler")
    if scaler:
        scaler = {k.split("_model.")[-1]: v for k, v in scaler.items()}
        return linears, scaler["running_mean"], scaler["running_variance"]
    agent = (run_params(checkpoint) or {}).get("agent")
    off = isinstance(agent, dict) and any(agent.get(k, 0) is None for k in ("observation_preprocessor", "state_preprocessor"))
    if not off:
        _warn(f"{checkpoint}: no observation scaler, and params/agent.yaml does not say it trained without one. "
              "The policy will see unnormalized observations and almost certainly not stand.")
    return linears, None, None


def _layout(checkpoint, obs_dim, env):
    terms = {k: v for k, v in (((env.get("observations") or {}).get("policy")) or {}).items()
             if isinstance(v, dict) and "func" in v}
    if not terms:  # a direct-workflow task (Walk), or no params/ to read
        layout = "unitree" if obs_dim == 45 else "isaac"
        if not env:
            _warn(f"{checkpoint}: no params/env.yaml beside it; guessing observation layout '{layout}' from its width.")
        return layout
    if "gait_command" in terms:
        return "wtw"
    if "clock" in terms:
        return "unitree_clock"
    if any(v.get("scale") for v in terms.values()):
        return "unitree_vel" if "base_lin_vel" in terms else "unitree"
    return "isaac"


def _wtw(checkpoint, obs_dim, env):
    gait = ((env.get("commands") or {}).get("gait")) or {}
    history = int((((env.get("observations") or {}).get("policy")) or {}).get("history_length") or 1)
    single = obs_dim // history
    width = single - sum(WTW_FIXED_BLOCKS) - WTW_CLOCK
    if history * single != obs_dim or width not in (8, 9):
        raise ValueError(f"{checkpoint}: WTW input {obs_dim} does not split into {history} steps with an 8- or 9-wide gait command.")
    nominal = [0.0, float(gait.get("nominal_frequency", 3.0)), *WTW_GAITS["trot"],
               float(gait.get("nominal_swing_height", 0.08)), 0.0, float(gait.get("nominal_stance_width", 0.34)),
               float(gait.get("nominal_duty", 0.5))]
    ranges = gait.get("ranges") or {}
    columns = ((0, "body_height"), (1, "frequency"), (5, "swing_height"), (6, "body_pitch"), (7, "stance_width"), (8, "duty"))
    max_stride = gait.get("max_stride")
    return single, {
        "blocks": list(WTW_FIXED_BLOCKS + (width, WTW_CLOCK)),
        "gait_command": np.array(nominal[:width], dtype=np.float32).tolist(),
        "gaits": {k: list(v) for k, v in WTW_GAITS.items()},
        "trained": [g for g in (gait.get("gait_names") or ("trot", "pace", "bound", "pronk")) if g in WTW_GAITS],
        "limits": {c: [float(x) for x in ranges[n]] for c, n in columns if n in ranges and c < width},
        "stance": float(gait.get("stance_duration", 0.5)),
        "max_stride": None if max_stride is None else float(max_stride),
        "yaw_radius": float(gait.get("yaw_radius", 0.3)),
    }


def _clock(checkpoint, env):
    commands = env.get("commands") or {}
    cfg = commands.get("clock") or {}
    if not cfg:
        raise ValueError(f"{checkpoint}: clock policy without commands.clock in params/env.yaml.")
    clock = {k: _plain(cfg[k]) for k in CLOCK_KEYS if cfg.get(k) is not None}
    clock["max_foot_speed"] = float((commands.get("base_velocity") or {}).get("max_foot_speed") or 0.0)
    return clock


def read_checkpoint(checkpoint, name=""):
    """(header dict, [(tensor name, float32 array)]) of a training checkpoint."""
    if _is_torchscript(checkpoint):
        raise ValueError(f"{checkpoint}: TorchScript policies are not supported; export the training checkpoint.")
    data = torch.load(checkpoint, map_location="cpu")
    rsl_rl = ("model_state_dict" in data or "actor_state_dict" in data) and "policy" not in data
    linears, mean, var = (_rsl_rl_actor if rsl_rl else _skrl_actor)(data, checkpoint)
    linears = [(w.detach().cpu().numpy(), b.detach().cpu().numpy()) for w, b in linears]
    for (w0, _), (w1, _) in zip(linears, linears[1:]):
        if w1.shape[1] != w0.shape[0]:
            raise ValueError(f"{checkpoint}: layer widths do not chain ({w0.shape} then {w1.shape}).")
    if linears[-1][0].shape[0] != 12:
        raise ValueError(f"{checkpoint}: {linears[-1][0].shape[0]} actions, not 12.")
    obs_dim = int(linears[0][0].shape[1])
    mean = np.zeros(obs_dim) if mean is None else torch.as_tensor(mean).flatten().cpu().numpy()
    var = np.ones(obs_dim) if var is None else torch.as_tensor(var).flatten().cpu().numpy()

    env = run_params(checkpoint, "env.yaml") or {}
    layout = _layout(checkpoint, obs_dim, env)
    wtw = clock = None
    if layout == "wtw":
        single, wtw = _wtw(checkpoint, obs_dim, env)
    elif layout == "unitree_clock":
        single, clock = obs_dim, _clock(checkpoint, env)
    elif obs_dim % SINGLE_STEP and 45 <= obs_dim <= 60:
        single = obs_dim  # one step of another width, no history (the stock Isaac Lab task's 48)
    else:
        single = SINGLE_STEP
        if obs_dim % single:
            _warn(f"{checkpoint}: input {obs_dim} is not a multiple of the {single}-wide step.")

    tensors = [("obs_mean", mean), ("obs_var", var)]
    for i, (w, b) in enumerate(linears):
        tensors += [(f"layer{i}.weight", w), (f"layer{i}.bias", b)]
    tensors = [(n, np.asarray(a).astype("<f4")) for n, a in tensors]

    meta = {
        "name": name,
        "source": checkpoint,
        "layout": layout,
        "obs_dim": obs_dim,
        "obs_dim_single": int(single),
        "history_len": max(1, obs_dim // single),
        "action_scale": ACTION_SCALE,
        "desired_qpos": _plain(DEFAULT_STANCE_QPOS),
        "num_layers": len(linears),
    }
    if wtw:
        meta["wtw"] = wtw
    if clock:
        meta["clock"] = clock
    # policies.json entries with commands "trick": how many switches, and the clock.
    trick = (env.get("commands") or {}).get("trick") or {}
    meta["trick"] = {"clock_hz": float(trick.get("clock_hz") or 0.0), "switches": int(trick.get("switches") or 1)}
    meta["tensors"] = [{"name": n, "shape": list(a.shape)} for n, a in tensors]
    return meta, tensors


def write_qpol(path, meta, tensors):
    header = yaml.safe_dump(meta, sort_keys=False).encode()
    with open(path, "wb") as f:
        f.write(b"QPOL" + struct.pack("<II", VERSION, len(header)) + header)
        for _, a in tensors:
            f.write(np.ascontiguousarray(a, dtype="<f4").tobytes())


def export_checkpoint(checkpoint, path, name=""):
    meta, tensors = read_checkpoint(checkpoint, name=name)
    write_qpol(path, meta, tensors)
    return meta
