"""Swing/stance statistics from a per-foot contact log. Numpy only, so both the MuJoCo sweep
(ROS 2 interpreter) and the Isaac sweep (Isaac venv) can import it."""

import numpy as np

FEET = ("FL", "FR", "RL", "RR")


def _runs(x):
    """(start, end, value) for every run of equal values in a 1-D bool array; end is exclusive."""
    edges = np.flatnonzero(np.diff(x.astype(np.int8))) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [len(x)]))
    return [(int(s), int(e), bool(x[s])) for s, e in zip(starts, ends)]


def debounce(contact, dt, min_air_s=0.05):
    """Fill airborne gaps shorter than min_air_s: that is contact chatter, not a swing."""
    out = contact.copy()
    min_n = max(1, int(round(min_air_s / dt)))
    for f in range(out.shape[1]):
        for s, e, v in _runs(out[:, f]):
            if not v and e - s < min_n and s > 0 and e < len(out):
                out[s:e, f] = True
    return out


def analyze(contact, foot_xy, dt, min_air_s=0.05, timeline_s=2.0, timeline_dt=0.01):
    """contact: (T, 4) bool, foot_xy: (T, 4, 2) world xy. Phases cut off by the window edges are
    dropped, so every swing and stance counted here was seen from start to end."""
    contact = debounce(np.asarray(contact, dtype=bool), dt, min_air_s)
    foot_xy = np.asarray(foot_xy, dtype=float)
    out = {k: [] for k in ("swing_time_s", "swing_time_std_s", "stance_time_s", "stance_time_std_s",
                           "duty_factor", "stride_period_s", "stride_length_m", "n_strides")}
    for f in range(contact.shape[1]):
        runs = _runs(contact[:, f])
        inner = runs[1:-1]
        swings = [(e - s) * dt for s, e, v in inner if not v]
        stances = [(e - s) * dt for s, e, v in inner if v]
        touchdowns = [s for s, e, v in runs[1:] if v]
        periods = np.diff(touchdowns) * dt if len(touchdowns) > 1 else []
        lengths = [np.linalg.norm(foot_xy[b, f] - foot_xy[a, f]) for a, b in zip(touchdowns, touchdowns[1:])]
        out["swing_time_s"].append(_mean(swings))
        out["swing_time_std_s"].append(_std(swings))
        out["stance_time_s"].append(_mean(stances))
        out["stance_time_std_s"].append(_std(stances))
        out["duty_factor"].append(float(contact[:, f].mean()) if len(contact) else None)
        out["stride_period_s"].append(_mean(periods))
        out["stride_length_m"].append(_mean(lengths))
        out["n_strides"].append(len(periods))
    n_down = contact.sum(axis=1)
    out["feet_in_contact_frac"] = [float((n_down == k).mean()) if len(contact) else 0.0 for k in range(5)]
    step = max(1, int(round(timeline_dt / dt)))
    tail = contact[-int(round(timeline_s / dt)):][::step]
    out["gait_timeline"] = {
        "dt": step * dt,
        "contact": ["".join("1" if c else "0" for c in tail[:, f]) for f in range(contact.shape[1])],
    }
    return out


def average(reports):
    """Mean of several analyze() outputs (e.g. the envs of one Isaac sweep point), ignoring None."""
    if not reports:
        return {}
    out = {}
    for k, v in reports[0].items():
        if k == "gait_timeline":
            out[k] = v
        elif isinstance(v, list):
            cols = zip(*(r[k] for r in reports))
            out[k] = [_mean([x for x in col if x is not None]) for col in cols]
    out["n_strides"] = [int(sum(r["n_strides"][f] for r in reports)) for f in range(len(reports[0]["n_strides"]))]
    return out


def rounded(m, n=3):
    r = lambda x: x if x is None or isinstance(x, int) else round(float(x), n)
    return {k: ([r(x) for x in v] if isinstance(v, list) else v)
            for k, v in m.items()}


def _mean(x):
    return float(np.mean(x)) if len(x) else None


def _std(x):
    return float(np.std(x)) if len(x) > 1 else None
