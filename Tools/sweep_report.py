"""
Eval sweep report from real-robot recordings.

Operator/sweep_teleop.py paces the Mujoco/eval_mujoco.py speed table by hand on the robot and marks
every segment on /sweep/state; the launcher records that into an MCAP session next to the
telemetry. This reads one or more of those sessions back, cuts out the walking window of every
completed segment, and writes real_eval_report_<checkpoint>.json in the same shape as
mujoco_eval_report_<checkpoint>.json, next to the checkpoint, so Tools/eval_viewer.py lists the
robot beside the simulators.

    python Tools/sweep_report.py Mcap/Recordings/run_teleop_sweep_20260912_101500 [more sessions]

Several sessions can go in at once, and an existing report is merged into rather than replaced
(--fresh replaces it). Each (axis, speed) keeps its most recent completed, undiscarded segment, so a
sweep split over battery changes, or a speed run again, ends up as one report.

What the robot can give, compared with the MuJoCo sweep:
  * Timing uses the robot's own header stamps (~39 Hz telemetry), never the laptop's receive time -
    over WiFi those arrive in bursts. /estimator/feet_contact has no stamp, but it goes out in the
    same tick as /sensors/joint_states, one each, so each contact message takes the stamp of the
    joint_states message with the same index.
  * actual_speed and position_error come from the state estimator (/odom/state_estimator); there is
    no ground truth. Yaw rate is the IMU gyro, as in MuJoCo.
  * foot_lift_height_cm is forward kinematics on the joint angles, rotated by the IMU attitude and
    measured from where that foot sits during stance. Base bob leaks into it.
  * Swing time, step frequency and phase offsets come from 25 ms contact samples, not 1 ms.
  * grf_stance_N, grf_peak_stance_N and foot_landing_vel_ms are null. The FSRs read raw counts, not
    newtons - they are reported as fsr_stance_counts / fsr_peak_stance_counts instead - and a 25 ms
    sample cannot catch the last instant before touchdown.
"""

import os
import sys
import json
import glob
import time
import argparse
import importlib.util

import numpy as np
from mcap_ros2.reader import read_ros2_messages

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contact_filter

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# By path: importing the Telemetry package runs its __init__, which pulls the estimator in too.
# kinematics.py itself needs only numpy and yaml.
_spec = importlib.util.spec_from_file_location(
    "go2_kinematics", os.path.join(REPO_DIR, "Telemetry", "kinematics.py"))
_kin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_kin)

# Isaac order, the one Telemetry/kinematics.py indexes as [leg, leg + 4, leg + 8].
JOINT_ORDER = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]
TOPICS = [
    "/sweep/state",
    "/sensors/joint_states",
    "/sensors/imu",
    "/odom/state_estimator",
    "/estimator/feet_contact",
    "/sensors/foot_force",
    "/sensors/foot_force_calibration",
    "/safety/reset",
    "/diagnostics/loop_stats",
]
FEET = ["FL", "FR", "RL", "RR"]
# Trot targets: diagonals land together, every other pair half a stride apart.
LEG_PAIRS = [("front_FL_FR", 0, 1, 0.5), ("rear_RL_RR", 2, 3, 0.5),
             ("lat_L_FL_RL", 0, 2, 0.5), ("lat_R_FR_RR", 1, 3, 0.5),
             ("diag_FL_RR", 0, 3, 0.0), ("diag_FR_RL", 1, 2, 0.0)]
AXIS_ORDER = {"x": 0, "y": 1, "yaw": 2}
UNITS = {"x": "m/s", "y": "m/s", "yaw": "rad/s"}


def mcap_files(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.mcap")))
        if not files:
            raise SystemExit(f"[sweep_report] no .mcap file in {path}")
        return files
    return [path]


def _stamp_s(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


class Recording:
    """One recorded session, every time on the robot's clock (seconds)."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        markers = []
        js_t, js_log, js_q = [], [], []
        imu_t, imu_quat, imu_gyro = [], [], []
        odom_t, odom_v = [], []
        contact, contact_log = [], []
        fsr, fsr_log = [], []
        calibration, resets_log = [], []
        loop_stats_t, loop_stats = [], []
        order = None

        for f in mcap_files(path):
            for m in read_ros2_messages(f, topics=TOPICS):
                topic, msg, log = m.channel.topic, m.ros_msg, m.log_time_ns * 1e-9
                if topic == "/sensors/joint_states":
                    if order is None:
                        names = list(msg.name)
                        order = ([names.index(n) for n in JOINT_ORDER]
                                 if all(n in names for n in JOINT_ORDER) else list(range(12)))
                    js_t.append(_stamp_s(msg))
                    js_log.append(log)
                    js_q.append([msg.position[i] for i in order])
                elif topic == "/sensors/imu":
                    o, g = msg.orientation, msg.angular_velocity
                    imu_t.append(_stamp_s(msg))
                    imu_quat.append([o.w, o.x, o.y, o.z])
                    imu_gyro.append([g.x, g.y, g.z])
                elif topic == "/odom/state_estimator":
                    v = msg.twist.twist.linear
                    odom_t.append(_stamp_s(msg))
                    odom_v.append([v.x, v.y, v.z])
                elif topic == "/estimator/feet_contact":
                    contact.append(list(msg.data)[:4])
                    contact_log.append(log)
                elif topic == "/sensors/foot_force":
                    fsr.append(list(msg.data)[:4])
                    fsr_log.append(log)
                elif topic == "/sensors/foot_force_calibration":
                    calibration.append((log, list(msg.data)))
                elif topic == "/safety/reset":
                    if msg.data:
                        resets_log.append(log)
                elif topic == "/diagnostics/loop_stats":
                    loop_stats_t.append(log)
                    loop_stats.append(list(msg.data))
                elif topic == "/sweep/state":
                    try:
                        markers.append((log, json.loads(msg.data)))
                    except ValueError:
                        pass

        if not markers:
            raise SystemExit(f"[sweep_report] {path}: no /sweep/state messages - this session was "
                             f"not recorded from the Eval Sweep teleop.")
        if not js_t:
            raise SystemExit(f"[sweep_report] {path}: no /sensors/joint_states - no telemetry "
                             f"reached the recorder.")

        js_t, js_log = np.asarray(js_t), np.asarray(js_log)
        # Receive time = robot stamp + clock offset + network delay. The delay is never negative and
        # sits near its floor for a good share of messages, so a low percentile of the difference is
        # the offset. It moves everything the robot did not stamp onto the robot's clock.
        self.clock_offset = float(np.percentile(js_log - js_t, 5))

        contact = np.asarray(contact, dtype=np.float64).reshape(-1, 4)
        if len(contact) == len(js_t):
            contact_t = js_t.copy()
        else:
            print(f"[sweep_report] {os.path.basename(path)}: {len(contact)} contact messages against "
                  f"{len(js_t)} joint_states - timing contacts by receive time instead.")
            contact_t = np.asarray(contact_log) - self.clock_offset

        js_sort = np.argsort(js_t, kind="stable")
        self.js_t, self.js_q = js_t[js_sort], np.asarray(js_q, dtype=np.float64)[js_sort]
        c_sort = np.argsort(contact_t, kind="stable")
        self.contact_t, self.contact = contact_t[c_sort], contact[c_sort]
        self.imu_t, self.imu_quat, self.imu_gyro = self._sorted(imu_t, imu_quat, imu_gyro)
        self.odom_t, self.odom_v = self._sorted(odom_t, odom_v)
        self.fsr_t, self.fsr = self._sorted(np.asarray(fsr_log) - self.clock_offset, fsr)
        self.calibration = [(t - self.clock_offset, d) for t, d in calibration]
        self.resets_t = np.asarray(resets_log) - self.clock_offset
        lt = np.asarray(loop_stats_t, dtype=np.float64) - self.clock_offset
        ls = np.asarray(loop_stats, dtype=np.float64).reshape(len(lt), -1) if len(lt) \
            else np.zeros((0, 3))
        l_sort = np.argsort(lt, kind="stable")
        self.loop_stats_t, self.loop_stats = lt[l_sort], ls[l_sort]

        # Markers carry "t", the operator machine's clock when they were sent. Their receive time
        # cannot stand in for it: /sweep/state is transient-local, so a recorder that joined late was
        # handed the backlog all at once. The smallest receive-minus-sent is a marker that arrived
        # live, and that places the operator's clock on the recorder's.
        marker_offset = min(log - ev["t"] for log, ev in markers if "t" in ev)
        self.events = sorted(
            ({**ev, "t_robot": ev["t"] + marker_offset - self.clock_offset}
             for _, ev in markers if "t" in ev),
            key=lambda e: e["t"])

    @staticmethod
    def _sorted(t, *columns):
        t = np.asarray(t, dtype=np.float64)
        if not len(t):
            # A topic the recording never saw. reshape(0, -1) cannot infer a width, so give it one;
            # every window over an empty array is empty whatever its width.
            return (t,) + tuple(np.zeros((0, 4)) for _ in columns)
        idx = np.argsort(t, kind="stable")
        return (t[idx],) + tuple(np.asarray(c, dtype=np.float64).reshape(len(t), -1)[idx]
                                 for c in columns)

    def completed_segments(self):
        """Segments marked complete - walked their full time, or ended by the operator releasing LB -
        and not discarded afterwards. Either way the window ends at segment_end."""
        segs = {}
        for ev in self.events:
            key = (ev.get("session"), ev.get("segment"))
            kind = ev.get("event")
            if kind == "segment_start":
                segs[key] = {k: ev.get(k) for k in ("session", "segment", "axis", "speed",
                                                     "checkpoint", "robot", "walk_s", "walk_m",
                                                     "ramp_s", "warmup_s")}
                segs[key].update(t_start=ev["t_robot"], t_wall=ev["t"])
            elif key not in segs:
                continue
            elif kind == "walk_start":
                segs[key]["t_walk"] = ev["t_robot"]
            elif kind == "segment_end":
                segs[key].update(t_end=ev["t_robot"], t_wall=ev["t"], status=ev.get("status"))
            elif kind == "discard":
                segs[key]["discarded"] = True
        return [dict(s, recording=self) for s in segs.values()
                if s.get("status") == "complete" and "t_walk" in s and "t_end" in s
                and not s.get("discarded")]


def _window(t, lo, hi):
    return slice(int(np.searchsorted(t, lo, "left")), int(np.searchsorted(t, hi, "right")))


def _nearest(t_src, t_query):
    i = np.clip(np.searchsorted(t_src, t_query), 1, len(t_src) - 1)
    left_closer = (t_query - t_src[i - 1]) < (t_src[i] - t_query)
    return np.where(left_closer, i - 1, i)


def _rotation(quat):
    w, x, y, z = quat
    return np.array([
        [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x ** 2 - 2 * y ** 2],
    ])


def measure(seg, settle_s=0.0):
    """One results entry, same keys as eval_mujoco.py. None if the window holds no telemetry."""
    rec = seg["recording"]
    axis, speed = seg["axis"], float(seg["speed"])
    ramp_s = float(seg.get("ramp_s") or 0.0)
    t_walk = seg["t_walk"]
    t0, t1 = t_walk + settle_s, seg["t_end"]
    duration = max(t1 - t0, 1e-3)

    def commanded(t):
        if ramp_s <= 0:
            return np.full_like(t, speed)
        return speed * np.clip((t - t_walk) / ramp_s, 0.0, 1.0)

    # --- Velocity tracking and drift ---
    so, si = _window(rec.odom_t, t0, t1), _window(rec.imu_t, t0, t1)
    to, v = rec.odom_t[so], rec.odom_v[so]
    ti, gyro = rec.imu_t[si], rec.imu_gyro[si]
    if len(to) < 2 or len(ti) < 2:
        return None
    dt_o, dt_i = np.diff(to), np.diff(ti)
    cx = commanded(to[:-1]) if axis == "x" else 0.0
    cy = commanded(to[:-1]) if axis == "y" else 0.0
    cyaw = commanded(ti[:-1]) if axis == "yaw" else 0.0
    err_x = float(np.sum((v[:-1, 0] - cx) * dt_o))
    err_y = float(np.sum((v[:-1, 1] - cy) * dt_o))
    err_yaw = float(np.sum((gyro[:-1, 2] - cyaw) * dt_i))
    if axis == "yaw":
        actual = float(np.mean(gyro[:, 2]))
    else:
        actual = float(np.mean(v[:, 0 if axis == "x" else 1]))

    # --- Gait timing, per foot, from that foot's own FSR counts ---
    # real_driver.py gates every foot on one global threshold; measured against this robot's
    # sensors that threshold falls inside the rear feet's stance band, so their stance is cut to
    # pieces and every swing time, step frequency and phase offset taken from it inherits the
    # error. contact_filter reads each foot's own swing floor and stance plateau instead.
    sf = _window(rec.fsr_t, t0, t1)
    cal = [d for t, d in rec.calibration if t <= t1 and len(d) >= 5]
    detected, contact_t, contact_source = None, None, "driver"
    if sf.stop - sf.start >= 3 and cal:
        offsets = np.asarray(cal[-1][1:5])
        contact_t = rec.fsr_t[sf]
        detected = contact_filter.segment_contacts(contact_t, rec.fsr[sf] - offsets)
        contact_source = "fsr_counts"
    if detected is None:
        sc = _window(rec.contact_t, t0, t1)
        contact_t = rec.contact_t[sc]
        flags = rec.contact[sc] > 0.5
        detected = [{"contact": flags[:, f], "touchdown": list(contact_t[1:][
                        flags[1:, f] & ~flags[:-1, f]]), "liftoff": [], "usable": True,
                     "note": "driver contact flag", "plateau": 0.0} for f in range(4)]

    stance_mask = np.column_stack([d["contact"] for d in detected]) if len(contact_t) else \
        np.zeros((0, 4), dtype=bool)
    duty = [float(np.mean(d["contact"])) if len(contact_t) else 0.0 for d in detected]
    step_freqs = [len(d["touchdown"]) / duration if duration > 0 else 0.0 for d in detected]

    swings = []
    for d in detected:
        flags = d["contact"]
        if len(flags) < 3:
            continue
        edges = np.flatnonzero(np.diff(flags.astype(np.int8)))
        # A swing is a liftoff followed by a touchdown, both inside the window.
        for a, b in zip(edges[:-1], edges[1:]):
            if not flags[a + 1]:
                swings.append(contact_t[b] - contact_t[a])
    swings = [w for w in swings if w > contact_filter.MIN_PHASE_S]
    avg_swing = float(np.mean(swings)) if swings else 0.0

    phase_pairs = {}
    for name, a, b, target in LEG_PAIRS:
        # percent is the folded phase difference, the same [0, 50]% convention eval_mujoco.py
        # reports; from_trot is how far that pair sits from its place in a trot.
        got = contact_filter.phase_offset(detected[a]["touchdown"], detected[b]["touchdown"], 0.0)
        if got is None:
            continue
        aim = contact_filter.phase_offset(detected[a]["touchdown"], detected[b]["touchdown"], target)
        phase_pairs[name] = {
            "percent": round(got["offset"] * 100.0, 2),
            "from_trot": round(aim["offset"] * 100.0, 2),
            "mean_phase": round(got["mean_phase"], 3),
            "lock": round(got["lock"], 3),
            "n": got["n"],
            "usable": bool(detected[a]["usable"] and detected[b]["usable"]),
        }

    quality = [{"foot": FEET[f], "usable": bool(d["usable"]), "note": d["note"],
                "stance_plateau_counts": round(d["plateau"], 1),
                "touchdowns": len(d["touchdown"])} for f, d in enumerate(detected)]

    def pair_percent(name):
        return phase_pairs[name]["percent"] if name in phase_pairs else 0.0

    # --- Foot lift: FK in the gravity-aligned frame, against that foot's stance level ---
    sj = _window(rec.js_t, t0, t1)
    tj, q = rec.js_t[sj], rec.js_q[sj]
    lift = [0.0] * 4
    if len(tj) and len(rec.imu_t) >= 2 and len(rec.contact_t) >= 2:
        quat = rec.imu_quat[_nearest(rec.imu_t, tj)]
        in_contact = (stance_mask[_nearest(contact_t, tj)] if len(contact_t)
                      else np.zeros((len(tj), 4), dtype=bool))
        z = np.zeros((len(tj), 4))
        for k in range(len(tj)):
            R = _rotation(quat[k])
            for f in range(4):
                z[k, f] = (R @ _kin.foot_position_body(f, q[k, [f, f + 4, f + 8]]))[2]
        for f in range(4):
            stance, swing = z[in_contact[:, f], f], z[~in_contact[:, f], f]
            if len(stance) and len(swing):
                lift[f] = max(0.0, float(swing.max() - np.median(stance)))

    # --- FSR load in counts, gated exactly as real_driver.py gates contact ---
    fsr_mean = fsr_peak = fsr_measured = None
    if sf.stop > sf.start and cal:
        load = rec.fsr[sf] - np.asarray(cal[-1][1:5])
        # Stance from the per-foot filter, not the driver's one threshold, so a rear foot whose
        # plateau never reaches it still reports the load it was carrying.
        stance = stance_mask if stance_mask.shape[0] == load.shape[0] else load > cal[-1][0]
        fsr_mean = [round(float(load[stance[:, f], f].mean()), 1) if stance[:, f].any() else 0.0
                    for f in range(4)]
        fsr_peak = [round(float(load[stance[:, f], f].max()), 1) if stance[:, f].any() else 0.0
                    for f in range(4)]
        # 1.0 per foot if real_driver.py measured that foot's zero at startup, 0.0 if it looked
        # loaded or moving and kept the configured default instead. Older recordings only carry
        # the first five floats [threshold, offsets x4]; those report None, not a false 1.0.
        if len(cal[-1]) >= 9:
            fsr_measured = [bool(v) for v in cal[-1][5:9]]

    resets = int(np.sum((rec.resets_t >= seg["t_start"]) & (rec.resets_t <= t1)))
    gaps = np.diff(tj) if len(tj) > 1 else np.array([0.0])

    return {
        "commanded_speed": round(speed, 2),
        "actual_speed": round(actual, 2),
        "foot_lift_height_cm": [round(h * 100.0, 2) for h in lift],
        "foot_swing_time_s": round(avg_swing, 2),
        "step_frequency_hz": [round(f, 2) for f in step_freqs],
        "grf_stance_N": None,
        "grf_peak_stance_N": None,
        "foot_landing_vel_ms": None,
        "fsr_stance_counts": fsr_mean,
        "fsr_peak_stance_counts": fsr_peak,
        "fsr_calibration_measured": fsr_measured,
        "phase_diff_front_percent": pair_percent("front_FL_FR"),
        "phase_diff_right_percent": pair_percent("lat_R_FR_RR"),
        "phase_diff_diag_percent": pair_percent("diag_FL_RR"),
        "phase_pairs": phase_pairs,
        "duty_factor": [round(d, 3) for d in duty],
        "contact_quality": quality,
        "contact_source": contact_source,
        "base_oscillation": {
            "std_z_vel": round(float(np.std(v[:, 2])), 2),
            "std_roll_vel": round(float(np.std(gyro[:, 0])), 2),
            "std_pitch_vel": round(float(np.std(gyro[:, 1])), 2),
        },
        "position_error": {
            "x_m": round(err_x, 4),
            "y_m": round(err_y, 4),
            "yaw_rad": round(err_yaw, 4),
        },
        # Console safety resets during the segment. Any at all means the policy was gated for part of
        # the walk, so read that speed's numbers as suspect.
        "safety_resets": resets,
        "walk_duration_s": round(duration, 2),
        "telemetry_samples": int(len(tj)),
        "telemetry_rate_hz": round(len(tj) / duration, 1),
        "max_telemetry_gap_ms": round(float(gaps.max()) * 1000.0, 1),
    }


def resolve_checkpoint(path):
    """The checkpoint on this machine. The sweep usually runs in Docker, where the repo is /app."""
    if not path:
        return None
    if os.path.exists(path):
        return os.path.abspath(path)
    if path.startswith("/app/"):
        local = os.path.join(REPO_DIR, path[len("/app/"):])
        if os.path.exists(local):
            return local
    return None


def default_report_path(ckpt_local, ckpt_recorded):
    name = os.path.splitext(os.path.basename(ckpt_local or ckpt_recorded or "unknown"))[0]
    if ckpt_local:
        return os.path.join(os.path.dirname(ckpt_local), f"real_eval_report_{name}.json")
    return os.path.join(REPO_DIR, "Mcap", "Reports", f"real_eval_report_{name}.json")


def main():
    parser = argparse.ArgumentParser(description="Eval report from Eval Sweep teleop recordings")
    parser.add_argument("recordings", nargs="+", help="recorded session directories or .mcap files")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint to file the report under, overriding the one recorded")
    parser.add_argument("--out", default=None, help="report path (default: next to the checkpoint)")
    parser.add_argument("--settle_s", type=float, default=0.0,
                        help="skip this long after each speed is commanded (default 0, as in MuJoCo)")
    parser.add_argument("--fresh", action="store_true",
                        help="replace an existing report instead of merging into it")
    args = parser.parse_args()

    segments = []
    loop_hz_all, infer_mean_all, infer_max_all = [], [], []
    for path in args.recordings:
        print(f"[sweep_report] reading {path}")
        rec = Recording(path)
        found = rec.completed_segments()
        print(f"[sweep_report]   {len(found)} completed segment(s), robot clock offset "
              f"{rec.clock_offset * 1000.0:+.0f} ms")
        if rec.calibration and len(rec.calibration[-1][1]) >= 9:
            names = ("FL", "FR", "RL", "RR")
            measured = rec.calibration[-1][1][5:9]
            rejected = [n for n, m in zip(names, measured) if not m]
            if rejected:
                print(f"[sweep_report]   WARNING: FSR zero not measured for {', '.join(rejected)} "
                      f"- kept the configured default. Load in counts is not trustworthy for "
                      f"{'that foot' if len(rejected) == 1 else 'those feet'} in this session.")
        if len(rec.loop_stats):
            loop_hz_all.append(rec.loop_stats[:, 0])
            infer_mean_all.append(rec.loop_stats[:, 1])
            infer_max_all.append(rec.loop_stats[:, 2])
        segments += found
    if not segments:
        raise SystemExit("[sweep_report] no completed segments in these recordings.")

    groups = {}
    for s in segments:
        groups.setdefault(args.checkpoint or s.get("checkpoint") or "", []).append(s)
    if args.out and len(groups) > 1:
        raise SystemExit(f"[sweep_report] these recordings cover {len(groups)} checkpoints "
                         f"({', '.join(os.path.basename(g) or '?' for g in groups)}); "
                         f"--out needs one, or pass --checkpoint.")

    for ckpt, segs in groups.items():
        latest = {}
        for s in sorted(segs, key=lambda s: s["t_wall"]):
            latest[(s["axis"], float(s["speed"]))] = s    # a later run of the same speed wins

        ckpt_local = resolve_checkpoint(ckpt)
        out = args.out or default_report_path(ckpt_local, ckpt)
        if ckpt and not ckpt_local and not args.out:
            print(f"[sweep_report] checkpoint {ckpt} is not on this machine; writing to {out}, "
                  f"where the eval viewer will not find it.")

        report = {}
        if os.path.exists(out) and not args.fresh:
            with open(out) as f:
                report = json.load(f)
        results = report.get("results", {})
        sources = set(report.get("metadata", {}).get("source_recordings", []))

        print(f"\n[sweep_report] {os.path.basename(ckpt) or '(no checkpoint recorded)'}")
        for (axis, speed), s in sorted(latest.items(), key=lambda kv: (AXIS_ORDER[kv[0][0]], kv[0][1])):
            r = measure(s, args.settle_s)
            if r is None:
                print(f"  {axis:>3} {speed:5.2f}  no telemetry in the walking window - skipped")
                continue
            results.setdefault(axis, {})[str(speed)] = r
            sources.add(s["recording"].path)
            flag = "  <- telemetry gap" if r["max_telemetry_gap_ms"] > 100 else ""
            flag += f"  <- {r['safety_resets']} safety reset(s)" if r["safety_resets"] else ""
            print(f"  {axis:>3} {speed:5.2f} {UNITS[axis]:<5} -> {r['actual_speed']:5.2f}  "
                  f"swing {r['foot_swing_time_s']:.2f} s  lift {r['foot_lift_height_cm']} cm  "
                  f"{r['telemetry_samples']} samples @ {r['telemetry_rate_hz']} Hz{flag}")

        walks = sorted({float(s.get("walk_s") or 0) for s in latest.values()})
        ramps = sorted({float(s.get("ramp_s") or 0) for s in latest.values()})
        robot = next((s.get("robot") for s in latest.values() if s.get("robot")), "go2")
        loop_hz = np.concatenate(loop_hz_all) if loop_hz_all else np.zeros(0)
        infer_mean = np.concatenate(infer_mean_all) if infer_mean_all else np.zeros(0)
        infer_max = np.concatenate(infer_max_all) if infer_max_all else np.zeros(0)
        loop_meta = None
        if len(loop_hz):
            loop_meta = {
                "control_loop_hz_median": round(float(np.median(loop_hz)), 1),
                "control_loop_hz_min": round(float(loop_hz.min()), 1),
                "infer_time_ms_mean": round(float(infer_mean.mean()), 2),
                "infer_time_ms_max": round(float(infer_max.max()), 2),
            }
            print(f"[sweep_report]   loop rate {loop_meta['control_loop_hz_median']:.0f} Hz "
                  f"(min {loop_meta['control_loop_hz_min']:.0f}), inference "
                  f"{loop_meta['infer_time_ms_mean']:.2f} ms mean / "
                  f"{loop_meta['infer_time_ms_max']:.2f} ms max")
        report = {
            "metadata": {
                "checkpoint": ckpt_local or ckpt or "None",
                "checkpoint_name": os.path.basename(ckpt) if ckpt else "None",
                "robot_type": robot,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "simulator": "real",
                "velocity_source": "state estimator",
                "walk_s": walks,
                "walk_m": sorted({float(s.get("walk_m") or 0) for s in latest.values()}),
                "ramp_s": ramps,
                "settle_s": args.settle_s,
                "source_recordings": sorted(sources),
                "loop_stats": loop_meta,
            },
            "results": {a: results[a] for a in sorted(results, key=lambda a: AXIS_ORDER.get(a, 9))},
        }
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=4)
        try:
            os.chmod(out, 0o666)
        except Exception:
            pass
        print(f"[sweep_report] report written: {out}")


if __name__ == "__main__":
    main()
