#!/usr/bin/env python3
"""Map video: the Nav2 map of a recorded run as it grows, with the robot's path, Nav2's plan, the objects, the
camera, and the agent's photos and messages.

    python3 Tools/map_video.py logs/thor/t11_qwen8b_1            # a trial folder from Tools/thor_trial.py
    python3 Tools/map_video.py Mcap/Recordings/run_x --log logs/agent/2026-11-18_10-00-00/log.md
    python3 Tools/map_video.py logs/thor/t11_qwen8b_1 --speed 2 --size 900

Writes into the trial folder (or the MCAP folder): map.mp4, map_final.png (the last frame, with the full path),
map.pgm / map.yaml (the last /map, the same files as map_save writes, so AMCL can use them), and camera.mp4: the
robot's own camera at real time and its recorded rate, from the same start as map.mp4 (for editing a demo).

The path is drawn with the map -> odom of its time, as SLAM estimated it then: after a loop closure, the old path
does not move with the map. The plan is drawn while Nav2 sends it (newer than 2 s); its end is the goal.
Agent log times are wall-clock seconds; they line up with the MCAP's log times when both come from one clock
(the robot). Needs numpy, opencv and mcap_ros2 (the Docker images have them), not ROS.
"""

import argparse
import bisect
import calendar
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from types import SimpleNamespace

import cv2
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.reader import read_ros2_messages

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src", "quadruped_common"))
from quadruped_common.map_files import FREE, OCCUPIED, UNKNOWN, origin_yaw, trinary, write_map  # noqa: E402

MAP, TF, PLAN, OBJECTS = "/map", "/tf", "/plan", "/objects"
# The robot's own camera first: go2_camera's JPEGs as the Go2 sends them; camera_bridge's copy of them only when
# go2_camera was not recorded. Never the small preview or the boxes drawn by the detector.
CAMERAS = [re.compile(r"^/go2_camera/image/compressed$"), re.compile(r"^/camera/[^/]+/image_raw/compressed$")]
BASES = ("base_footprint", "base")  # the frame SLAM uses first; the driver's base if there is no footprint
PLAN_AGE = 2.0
PATH_DT = 0.1

# BGR
COLORS = {UNKNOWN: (128, 128, 128), FREE: (250, 250, 250), OCCUPIED: (0, 0, 0)}
PATH, PLAN_C, GOAL, ROBOT, OBJ = (220, 120, 20), (40, 170, 40), (200, 40, 200), (40, 40, 230), (0, 140, 255)
KIND_COLORS = {"you": (255, 255, 255), "call": (120, 220, 255), "said": (140, 255, 140), "error": (90, 90, 255),
               "note": (200, 200, 200), "info": (200, 200, 200)}
BG, TEXT = (32, 32, 32), (235, 235, 235)


def pose2d(tr):
    q = tr.rotation
    return (tr.translation.x, tr.translation.y,
            math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))


def compose(a, b):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2])


def latest(times, t):
    """Index of the last time <= t, or -1."""
    return bisect.bisect_right(times, t) - 1


def find_inputs(path, log):
    """Returns (mcap files, out dir, agent log or None, trial.json dict)."""
    path = os.path.abspath(path)
    trial = {}
    if os.path.isdir(os.path.join(path, "mcap")):
        out, bag = path, os.path.join(path, "mcap")
        if os.path.isfile(os.path.join(path, "trial.json")):
            with open(os.path.join(path, "trial.json")) as f:
                trial = json.load(f)
        if log is None:
            logs = sorted(glob.glob(os.path.join(path, "agent", "*", "log.md")))
            log = logs[-1] if logs else None
    else:
        out, bag = (path, path) if os.path.isdir(path) else (os.path.dirname(path), path)
    files = [bag] if os.path.isfile(bag) else sorted(glob.glob(os.path.join(bag, "*.mcap")))
    if not files:
        sys.exit(f"no .mcap in {bag}")
    return files, out, log, trial


def topic_counts(files):
    """{topic: message count} from the MCAP summaries (a file cut before its summary counts as empty)."""
    counts = {}
    for f in files:
        with open(f, "rb") as s:
            summary = make_reader(s).get_summary()
        if summary is None:
            print(f"warning: {os.path.basename(f)} has no summary (the recorder was killed?)")
            continue
        per_channel = summary.statistics.channel_message_counts if summary.statistics else {}
        for cid, c in summary.channels.items():
            counts[c.topic] = counts.get(c.topic, 0) + per_channel.get(cid, 0)
    return counts


def pick_camera(counts, wanted):
    if wanted is not None:
        return wanted or None
    for pattern in CAMERAS:
        found = sorted(t for t, n in counts.items() if n and pattern.match(t))
        if found:
            return found[0]
    return None


def camera_video(files, topic, path, t0, t1, fps):
    """The camera's own JPEGs at real time (fps frames a second, a frame repeated over a gap), from t0 to t1, so
    it starts with the map video. Read a second time, frame by frame: a long demo does not fit in memory."""
    writer, last, k = None, None, 0
    for f in files:
        for m in read_ros2_messages(f, topics=[topic]):
            img = cv2.imdecode(np.frombuffer(bytes(m.ros_msg.data), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            if img.shape[1] % 2 or img.shape[0] % 2:
                img = img[:img.shape[0] // 2 * 2, :img.shape[1] // 2 * 2]
            if writer is None:
                writer = Writer(path, img.shape[1], img.shape[0], fps)
                last = img  # the first frame also fills the time before it
            if last.shape != img.shape:
                img = cv2.resize(img, (last.shape[1], last.shape[0]))
            target = int((m.log_time_ns * 1e-9 - t0) * fps)
            while k < target:
                writer.write(last)
                k += 1
            last = img
    if writer is None:
        return 0
    while k <= int((t1 - t0) * fps):
        writer.write(last)
        k += 1
    writer.close()
    return k


def read_bag(files, camera, frame_dt):
    maps, map_t, last_grid = [], [], None
    mo_t, mo, ob = [], [], {b: ([], []) for b in BASES}
    plans, plan_t, objects, obj_t, cam, cam_t = [], [], [], [], [], []
    t_first, t_last = math.inf, -math.inf
    topics = [MAP, TF, PLAN, OBJECTS] + ([camera] if camera else [])
    for f in files:
        for m in read_ros2_messages(f, topics=topics):
            t, topic, msg = m.log_time_ns * 1e-9, m.channel.topic, m.ros_msg
            t_first, t_last = min(t_first, t), max(t_last, t)
            if topic == TF:
                for tr in msg.transforms:
                    parent, child = tr.header.frame_id.lstrip("/"), tr.child_frame_id.lstrip("/")
                    if parent == "map" and child == "odom":
                        mo_t.append(t)
                        mo.append(pose2d(tr.transform))
                    elif parent == "odom" and child in ob:
                        ob[child][0].append(t)
                        ob[child][1].append(pose2d(tr.transform))
            elif topic == MAP:
                info, data = msg.info, msg.data
                cells = (np.frombuffer(data, np.int8) if isinstance(data, (bytes, bytearray))
                         else np.asarray(data, np.int8)).reshape(info.height, info.width)
                if abs(origin_yaw(info)) > 1e-3:
                    print(f"warning: the map's origin is turned {origin_yaw(info):.3f} rad; drawn as not turned")
                maps.append((info.origin.position.x, info.origin.position.y, info.resolution, trinary(cells)))
                map_t.append(t)
                last_grid = SimpleNamespace(info=info, data=cells.ravel())
            elif topic == PLAN:
                plan_t.append(t)
                plans.append(np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses]).reshape(-1, 2))
            elif topic == OBJECTS:
                try:
                    objects.append(json.loads(msg.data).get("objects", []))
                    obj_t.append(t)
                except ValueError:
                    pass
            elif topic == camera and (not cam_t or t - cam_t[-1] >= 0.9 * frame_dt):
                cam_t.append(t)
                cam.append(bytes(msg.data))
    base = next((b for b in BASES if ob[b][0]), None)
    ob_t, ob_p = ob[base] if base else ([], [])
    # The robot's path on the map: odom -> base with the map -> odom of its time.
    path_t, path = [], []
    for t, p in zip(ob_t, ob_p):
        i = latest(mo_t, t)
        if i >= 0 and (not path_t or t - path_t[-1] >= PATH_DT):
            path_t.append(t)
            path.append(compose(mo[i], p))
    return SimpleNamespace(maps=maps, map_t=map_t, last_grid=last_grid, mo_t=mo_t, mo=mo, ob_t=ob_t, ob=ob_p,
                           base=base, path_t=path_t, path=np.array(path).reshape(-1, 3), plans=plans, plan_t=plan_t,
                           objects=objects, obj_t=obj_t, cam=cam, cam_t=cam_t, t0=t_first, t1=t_last)


LINE = re.compile(r"^- `(\d\d):(\d\d):(\d\d)` (.*)$")
TASK = re.compile(r"^## (\d\d):(\d\d):(\d\d) you: (.*)$")
IMAGE = re.compile(r"^\s*!\[[^\]]*\]\(([^)]+)\)")


def read_log(path, utc_offset):
    """llm_host's log.md as events [(t, kind, text)] and photos [(t, file, tool)]. Its times are local
    HH:MM:SS; the date is the folder's name. utc_offset: the robot's, from trial.json; None = this machine's."""
    events, photos = [], []
    if not path:
        return events, photos
    day = datetime.strptime(os.path.basename(os.path.dirname(path))[:10], "%Y-%m-%d")
    midnight = (time.mktime(day.timetuple()) if utc_offset is None
                else calendar.timegm(day.timetuple()) - utc_offset)
    prev, extra = None, 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = TASK.match(line) or LINE.match(line)
            if m:
                secs = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
                if prev is not None and secs + 3600 < prev:
                    extra += 86400  # past midnight
                prev = secs
                text = m.group(4).strip()
                if m.re is TASK:
                    kind = "you"
                elif text.startswith("→ **"):
                    kind, text = "call", text[4:].rstrip("*")
                elif text.startswith("agent: "):
                    kind, text = "said", text[7:]
                else:
                    kind = text.split(":", 1)[0].split(" ", 1)[0]
                events.append((midnight + extra + secs, kind, text))
                continue
            m = IMAGE.match(line)
            if m and events:
                tool = re.sub(r"^\d+_|\.\w+$", "", m.group(1))
                photos.append((events[-1][0], os.path.join(os.path.dirname(path), m.group(1)), tool))
    return events, photos


def ascii_text(s):
    return s.replace("→", "->").encode("ascii", "replace").decode()


def put(img, text, org, scale, color, width_px=None, thick=1):
    text = ascii_text(text[:400])
    if width_px:
        while text and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0] > width_px:
            text = text[:-4] + "..." if len(text) > 4 else ""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def fit(img, w, h):
    """img letterboxed into w x h."""
    out = np.full((h, w, 3), BG, np.uint8)
    if img is None:
        return out
    s = min(w / img.shape[1], h / img.shape[0])
    iw, ih = max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))
    x, y = (w - iw) // 2, (h - ih) // 2
    out[y:y + ih, x:x + iw] = cv2.resize(img, (iw, ih), interpolation=cv2.INTER_AREA)
    return out


class Renderer:
    def __init__(self, bag, events, photos, size, panel, speed, title):
        self.bag, self.events, self.photos, self.speed, self.title = bag, events, photos, speed, title
        self.ev_t = [e[0] for e in events]
        self.ph_t = [p[0] for p in photos]
        self.S = size
        self.P = (int(size * 0.8) // 2) * 2 if panel else 0
        self.fs = size / 720
        self.line = int(26 * self.fs)
        self.CAP = (int(self.line * 4.6) // 2) * 2
        self.W, self.H = self.S + self.P, self.S + self.CAP
        # One fixed view: every map's extent and the whole path, 0.5 m around.
        xs, ys = [], []
        for ox, oy, res, img in bag.maps:
            xs += [ox, ox + img.shape[1] * res]
            ys += [oy, oy + img.shape[0] * res]
        if len(bag.path):
            xs += [bag.path[:, 0].min(), bag.path[:, 0].max()]
            ys += [bag.path[:, 1].min(), bag.path[:, 1].max()]
        if not xs:
            xs, ys = [-5, 5], [-5, 5]
        x0, x1, y0, y1 = min(xs) - 0.5, max(xs) + 0.5, min(ys) - 0.5, max(ys) + 0.5
        self.ppm = size / max(x1 - x0, y1 - y0)
        self.x0 = x0 - (size / self.ppm - (x1 - x0)) / 2
        self.y1 = y1 + (size / self.ppm - (y1 - y0)) / 2
        self.lut = np.zeros((256, 3), np.uint8)
        self.lut[:] = COLORS[UNKNOWN]
        for value, color in COLORS.items():
            self.lut[value] = color
        self.map_i, self.map_img = None, None
        self.cam_i, self.cam_img = None, None
        self.ph_i, self.ph_img = None, None
        self.task = next((e[2] for e in events if e[1] == "you"), "")

    def px(self, x, y):
        return int(round((x - self.x0) * self.ppm)), int(round((self.y1 - y) * self.ppm))

    def map_layer(self, i):
        if i == self.map_i:
            return self.map_img
        self.map_i = i
        if i < 0:
            self.map_img = np.full((self.S, self.S, 3), COLORS[UNKNOWN], np.uint8)
            return self.map_img
        ox, oy, res, img = self.bag.maps[i]
        colored = self.lut[np.flipud(img)]
        a = res * self.ppm
        m = np.float32([[a, 0, (ox - self.x0) * self.ppm], [0, a, (self.y1 - oy - img.shape[0] * res) * self.ppm]])
        self.map_img = cv2.warpAffine(colored, m, (self.S, self.S), flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=COLORS[UNKNOWN])
        return self.map_img

    def map_panel(self, t):
        b, fs = self.bag, self.fs
        img = self.map_layer(latest(b.map_t, t)).copy()
        n = bisect.bisect_right(b.path_t, t)
        if n > 1:
            pts = np.array([self.px(x, y) for x, y, _ in b.path[:n]], np.int32)
            cv2.polylines(img, [pts], False, PATH, max(1, int(2 * fs)), cv2.LINE_AA)
        i = latest(b.plan_t, t)
        if i >= 0 and t - b.plan_t[i] <= PLAN_AGE and len(b.plans[i]):
            pts = np.array([self.px(x, y) for x, y in b.plans[i]], np.int32)
            cv2.polylines(img, [pts], False, PLAN_C, max(1, int(2 * fs)), cv2.LINE_AA)
            g = tuple(pts[-1])
            cv2.drawMarker(img, g, GOAL, cv2.MARKER_TILTED_CROSS, int(16 * fs), max(1, int(2 * fs)))
        i = latest(b.obj_t, t)
        if i >= 0:
            for o in b.objects[i]:
                p = self.px(o.get("x", 0.0), o.get("y", 0.0))
                cv2.circle(img, p, int(5 * fs), OBJ, -1, cv2.LINE_AA)
                put(img, f"{o.get('id', '')} {o.get('label', '')}", (p[0] + int(7 * fs), p[1] + int(4 * fs)),
                    0.4 * fs, (0, 90, 200))
        i, j = latest(b.mo_t, t), latest(b.ob_t, t)
        if i >= 0 and j >= 0:
            x, y, yaw = compose(b.mo[i], b.ob[j])
            p = self.px(x, y)
            q = self.px(x + 0.45 * math.cos(yaw), y + 0.45 * math.sin(yaw))
            cv2.circle(img, p, int(7 * fs), ROBOT, -1, cv2.LINE_AA)
            cv2.arrowedLine(img, p, q, ROBOT, max(1, int(3 * fs)), cv2.LINE_AA, tipLength=0.35)
        # 1 m scale bar
        x0, y0 = int(12 * fs), self.S - int(14 * fs)
        cv2.line(img, (x0, y0), (x0 + int(self.ppm), y0), (0, 0, 0), max(1, int(3 * fs)))
        put(img, "1 m", (x0, y0 - int(6 * fs)), 0.45 * fs, (0, 0, 0))
        return img

    def side_panel(self, t):
        b, P, S, fs = self.bag, self.P, self.S, self.fs
        panel = np.full((S, P, 3), BG, np.uint8)
        label = int(24 * fs)
        h = S // 2 - label
        i = latest(b.cam_t, t)
        if i != self.cam_i:
            self.cam_i = i
            self.cam_img = (cv2.imdecode(np.frombuffer(b.cam[i], np.uint8), cv2.IMREAD_REDUCED_COLOR_2)
                            if i >= 0 else None)
        put(panel, "camera" if b.cam else "camera: not recorded", (int(6 * fs), int(17 * fs)), 0.5 * fs, TEXT)
        panel[label:label + h] = fit(self.cam_img, P, h)
        i = latest(self.ph_t, t)
        if i != self.ph_i:
            self.ph_i = i
            self.ph_img = cv2.imread(self.photos[i][1]) if i >= 0 else None
        text = "agent photo"
        if i >= 0:
            text += f": {self.photos[i][2]} at {self.clock(self.photos[i][0])}"
        put(panel, text, (int(6 * fs), S // 2 + int(17 * fs)), 0.5 * fs, TEXT, P - int(12 * fs))
        panel[S // 2 + label:S // 2 + label + h] = fit(self.ph_img, P, h)
        return panel

    def clock(self, t):
        s = max(0, int(t - self.bag.t0))
        return f"{s // 60:02d}:{s % 60:02d}"

    def caption(self, t):
        fs, line = self.fs, self.line
        cap = np.full((self.CAP, self.W, 3), BG, np.uint8)
        put(cap, f"{self.clock(t)}  x{self.speed:g}   {self.title}   task: {self.task}", (int(8 * fs), line - 4),
            0.55 * fs, TEXT, self.W - int(16 * fs), 1)
        n = bisect.bisect_right(self.ev_t, t)
        shown = [e for e in self.events[:n] if e[1] != "you" or e[2] != self.task][-3:]
        for k, (te, kind, text) in enumerate(shown):
            first = text.split("\n", 1)[0]
            lead = {"call": "-> ", "said": "agent: ", "you": "you: "}.get(kind, "")
            put(cap, f"{self.clock(te)}  {lead}{first}", (int(8 * fs), line * (k + 2) - 4), 0.5 * fs,
                KIND_COLORS.get(kind, (200, 200, 200)), self.W - int(16 * fs))
        return cap

    def frame(self, t):
        out = np.empty((self.H, self.W, 3), np.uint8)
        out[:self.S, :self.S] = self.map_panel(t)
        if self.P:
            out[:self.S, self.S:] = self.side_panel(t)
        out[self.S:] = self.caption(t)
        return out


class Writer:
    """H.264 through ffmpeg when it is there (plays everywhere); else OpenCV's mp4v."""

    def __init__(self, path, w, h, fps):
        self.proc = self.cv = None
        if shutil.which("ffmpeg"):
            self.proc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                          "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                                          "-pix_fmt", "yuv420p", "-crf", "23", path], stdin=subprocess.PIPE)
        else:
            self.cv = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            print("no ffmpeg: mp4v (some browsers do not play it)")

    def write(self, img):
        if self.proc:
            self.proc.stdin.write(img.tobytes())
        else:
            self.cv.write(img)

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.cv.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path", help="trial folder (with mcap/ and agent/), an MCAP folder or an .mcap file")
    ap.add_argument("--log", default=None, help="the agent's log.md (default: the trial's agent/*/log.md)")
    ap.add_argument("--camera", default=None,
                    help='camera topic (default: /go2_camera/image/compressed, else /camera/*/image_raw/compressed; '
                         '"" for none)')
    ap.add_argument("--camera_fps", type=float, default=None, help="camera.mp4 rate (default: the recorded rate)")
    ap.add_argument("--no_camera_video", action="store_true", help="no camera.mp4")
    ap.add_argument("--speed", type=float, default=4.0, help="run seconds per video second")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--size", type=int, default=720, help="map panel size in pixels (even)")
    ap.add_argument("--no_panel", action="store_true", help="no camera and agent photo panel")
    ap.add_argument("--hold", type=float, default=3.0, help="seconds the last frame stays")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    files, out, log, trial = find_inputs(args.path, args.log)
    out = args.out_dir or out
    os.makedirs(out, exist_ok=True)
    frame_dt = args.speed / args.fps
    counts = topic_counts(files)
    camera = pick_camera(counts, args.camera)
    print(f"reading {len(files)} MCAP file(s){', camera ' + camera if camera else ', no camera'} ...")
    bag = read_bag(files, None if args.no_panel else camera, frame_dt)
    if not math.isfinite(bag.t0):
        sys.exit("no /map, /tf, /plan or /objects in the recording")
    events, photos = read_log(log, trial.get("utc_offset_s"))
    print(f"{len(bag.maps)} maps, {len(bag.path)} path points (odom -> {bag.base}), {len(bag.plans)} plans, "
          f"{len(bag.cam)} camera frames, {len(events)} agent events, {len(photos)} agent photos")
    if not bag.maps:
        print("warning: no /map in the recording (was Nav2 or SLAM running?)")
    if not bag.mo_t:
        print("warning: no map -> odom on /tf: no robot and no path")
    if events and not (bag.t0 - 60 <= events[0][0] <= bag.t1 + 60):
        print("warning: the agent log's times are not inside the recording: is the clock or utc_offset wrong?")

    if bag.last_grid is not None:
        write_map(bag.last_grid, os.path.join(out, "map"))
        print(f"map: {os.path.join(out, 'map.pgm')}, .yaml")

    title = " ".join(str(trial[k]) for k in ("name", "model") if trial.get(k))
    r = Renderer(bag, events, photos, (args.size // 2) * 2, not args.no_panel, args.speed, title)
    video = os.path.join(out, "map.mp4")
    writer = Writer(video, r.W, r.H, args.fps)
    n = int((bag.t1 - bag.t0) / frame_dt) + 1
    step = max(1, n // 10)
    for k in range(n):
        writer.write(r.frame(bag.t0 + k * frame_dt))
        if k % step == 0:
            print(f"  {100 * k // n}%", end="", flush=True)
    last = r.frame(bag.t1)
    for _ in range(int(args.hold * args.fps)):
        writer.write(last)
    writer.close()
    cv2.imwrite(os.path.join(out, "map_final.png"), last)
    print(f"\nvideo: {video} ({n} frames, {r.W} x {r.H}); last frame: {os.path.join(out, 'map_final.png')}")

    if camera and not args.no_camera_video:
        fps = args.camera_fps or min(30.0, max(1.0, round(counts.get(camera, 0) / max(bag.t1 - bag.t0, 1e-3))))
        video = os.path.join(out, "camera.mp4")
        print(f"camera video from {camera} at {fps:g} fps ...")
        frames = camera_video(files, camera, video, bag.t0, bag.t1, fps)
        print(f"camera video: {video} ({frames} frames, real time)" if frames else "no camera frames: no camera.mp4")


if __name__ == "__main__":
    main()
