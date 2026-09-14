"""Per-foot foot contact from raw FSR counts.

real_driver.py gates every foot on one global threshold (10 counts over that foot's stored
offset). Measured against this robot's sensors that threshold is wrong for the rear feet: at
every forward speed the rear plateaus sit at 0.30-0.44 of the front ones for the same share of
body weight, so the single threshold falls inside the rear stance band. Standing, the rear feet
never cross it; walking, they cross it only near the load peak. The duty factors that come out
are impossible -- 0.86/0.89 front against 0.19/0.09 rear at 0.15 m/s, where any gait needs
about 2.0 feet down at once -- and every swing time, step frequency and phase offset derived
from them inherits the error.

Per foot, per window: read the swing floor and the stance plateau off that foot's own
percentiles, run a Schmitt trigger between them, and place each touchdown by linear
interpolation of the threshold crossing, so an event is not quantised to the 31 ms telemetry
sample. A foot whose load never separates into two levels is reported unusable rather than
guessed at.

Chatter is rejected against the stride period, and only on the touchdown list. The weak rear
sensors dip back under the falling threshold inside a single stance, splitting it in two: at a
0.15 m/s command that put RR at 28 touchdowns against 16-18 for the other three, its median
stride at 0.73 s against their 1.3-1.4 s. The split gaps are about 0.24 s, which is longer than
the real swings the front-right foot takes at that speed (0.14-0.20 s), so no rule on gap length
alone can separate them. What does separate them is that a gait is periodic and all four legs
share one stride: two touchdowns inside half a stride cannot both start one, whatever the gap
between them looks like. The period is measured once across the feet, and each foot's touchdowns
are then thinned against it. Over the walking segments that takes the spread between the four
legs' step frequencies -- which a periodic gait forces to 1.0 -- from 1.21 unthinned to 1.10,
against 1.30 for the driver's own contact flags. Duty factor and swing time keep reading the
Schmitt trigger directly, so thinning cannot invent or erase stance.
"""
import numpy as np

FLOOR_COUNTS = 5.0      # a foot held below this is carrying nothing
MIN_RANGE = 10.0        # p5..p95 spread under this never separates stance from swing
RISE_FRAC, FALL_FRAC = 0.45, 0.25
MIN_PHASE_S = 0.06      # a stance or swing shorter than one telemetry sample is not real
STRIDE_FRAC = 0.5       # two touchdowns inside half a stride cannot both start one
AMBIGUOUS_MAX = 0.30    # usable only if fewer than this share of samples sit between thresholds


def _crossing(t0, t1, v0, v1, level):
    return t1 if v1 == v0 else t0 + (level - v0) / (v1 - v0) * (t1 - t0)


def _stance_intervals(t, load, rise, fall):
    """Raw (touchdown, liftoff) pairs from the Schmitt trigger, times interpolated.

    A stance still open at the end of the window gets liftoff None.
    """
    spans, open_td = [], (t[0] if load[0] > rise else None)
    state = bool(load[0] > rise)
    for k in range(1, len(t)):
        if not state and load[k] > rise:
            state = True
            open_td = _crossing(t[k - 1], t[k], load[k - 1], load[k], rise)
        elif state and load[k] < fall:
            state = False
            spans.append((open_td, _crossing(t[k - 1], t[k], load[k - 1], load[k], fall)))
            open_td = None
    if state:
        spans.append((open_td, None))
    return spans


def _clean(spans, t_end):
    """Drop stances and swings too short to be either, at one telemetry sample."""
    merged = []
    for td, lift in spans:
        if merged and merged[-1][1] is not None and td - merged[-1][1] < MIN_PHASE_S:
            merged[-1] = (merged[-1][0], lift)
        else:
            merged.append((td, lift))
    return [(td, lift) for td, lift in merged
            if (t_end if lift is None else lift) - td >= MIN_PHASE_S]


def _detect(t, load):
    floor, top = np.percentile(load, 5), np.percentile(load, 95)
    if top - floor < MIN_RANGE:
        return None, floor, top
    rise, fall = floor + RISE_FRAC * (top - floor), floor + FALL_FRAC * (top - floor)
    return _clean(_stance_intervals(t, load, rise, fall), t[-1]), floor, top


def _thin(touchdown, min_interval):
    """One touchdown per stride: the first of any run that falls inside one period."""
    kept = []
    for td in touchdown:
        if not kept or td - kept[-1] >= min_interval:
            kept.append(td)
    return kept


def _held(t, load):
    steady = float(np.median(load)) > FLOOR_COUNTS
    return {"contact": np.full(len(t), steady), "touchdown": [], "liftoff": [], "usable": True,
            "note": "held in stance" if steady else "held off the ground",
            "plateau": float(np.median(load)) if steady else 0.0}


def _package(t, load, spans, floor, top, min_interval):
    rise, fall = floor + RISE_FRAC * (top - floor), floor + FALL_FRAC * (top - floor)
    contact = np.zeros(len(t), dtype=bool)
    for td, lift in spans:
        contact |= (t >= td) & (t <= (t[-1] if lift is None else lift))
    ambiguous = float(np.mean((load > fall) & (load < rise)))
    # A stance open when the window began has no touchdown of its own to report.
    touchdown = _thin([td for td, _ in spans if td > t[0]], min_interval)
    liftoff = [lift for _, lift in spans if lift is not None]
    return {"contact": contact, "touchdown": touchdown, "liftoff": liftoff,
            "usable": ambiguous <= AMBIGUOUS_MAX, "note": f"ambiguous {ambiguous:.0%}",
            "plateau": float(np.percentile(load[contact], 95)) if contact.any() else 0.0}


def stride_period(seeds):
    """One stride period for the window, from the per-foot medians.

    The median of the four is used rather than the pooled intervals, so a single foot that
    double-counts its stances cannot drag the estimate down with it.
    """
    per_foot = [float(np.median(np.diff([td for td, _ in s]))) for s in seeds
                if s is not None and len(s) >= 3]
    return float(np.median(per_foot)) if per_foot else None


def segment_contacts(t, load):
    """Contact state and event times for each column of a (samples, 4) load array."""
    if len(t) < 3:
        empty = {"contact": np.zeros(len(t), dtype=bool), "touchdown": [], "liftoff": [],
                 "usable": False, "note": "too few samples", "plateau": 0.0}
        return [dict(empty) for _ in range(load.shape[1])]

    detected = [_detect(t, load[:, f]) for f in range(load.shape[1])]
    period = stride_period([spans for spans, _, _ in detected])
    min_interval = 0.0 if period is None else STRIDE_FRAC * period
    out = []
    for f, (spans, floor, top) in enumerate(detected):
        out.append(_held(t, load[:, f]) if spans is None
                   else _package(t, load[:, f], spans, floor, top, min_interval))
    return out


def relative_phase(touchdown_a, touchdown_b):
    """Phase of each B touchdown within A's current stride, unfolded to [0, 1)."""
    a = np.asarray(touchdown_a, dtype=float)
    out = []
    for tb in touchdown_b:
        prior = a[a < tb]
        if len(prior) < 2:
            continue
        stride = prior[-1] - prior[-2]
        if stride <= 0:
            continue
        out.append(((tb - prior[-1]) % stride) / stride)
    return np.asarray(out)


def phase_offset(touchdown_a, touchdown_b, target):
    """How far pair (A, B) sits from `target` phase, as a share of a stride in [0, 0.5].

    Averaged on the circle, so strides either side of the wrap do not cancel to the middle.
    `lock` is the resultant length: 1.0 is a pair that repeats the same offset every stride,
    and near 0 means there is no repeatable phase relation to report.
    """
    p = relative_phase(touchdown_a, touchdown_b)
    if not len(p):
        return None
    resultant = np.mean(np.exp(2j * np.pi * p))
    mean = float(np.angle(resultant) / (2 * np.pi) % 1.0)
    delta = abs(mean - target) % 1.0
    return {"offset": min(delta, 1.0 - delta), "mean_phase": mean,
            "lock": float(abs(resultant)), "n": int(len(p))}
