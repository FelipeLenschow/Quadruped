"""
Eval Sweep teleop - the Mujoco/eval_mujoco.py speed protocol, paced by hand on the real robot.

eval_mujoco.py walks a checkpoint through a fixed table of commanded velocities and measures each
one. None of that can run unattended on the robot: the room runs out, the robot has to be walked
back between speeds, and someone has to be ready to kill it. This node keeps the protocol - the
same speed table, the same standing warmup before each speed - and hands the pacing to the gamepad
(F710 in X mode):

    LB (hold)            run the selected speed. Releasing it stops the robot, counts the segment as
                         done however far it got, and selects the next speed. Released during the
                         standing warmup nothing was walked, so the segment is aborted instead
    D-pad left / right   previous / next speed, across axes (idle only). Left right after a release
                         goes back to that speed; running it again replaces it in the report
    D-pad up / down      previous / next axis      (idle only)
    RB (hold) + sticks   free drive, e.g. back to the start line. Never part of a segment
    Back                 discard the last completed segment, to redo it
    Start                end the session
    A / B / X / Y        E-STOP (safety.estop_joy_buttons, the same buttons the console uses)

It measures nothing itself. Every state change goes out on /sweep/state as a JSON string, the MCAP
recording keeps that next to the telemetry, and Tools/sweep_report.py cuts the recording into
segments afterwards.

Runs in place of teleop_twist_joy, never beside it: two /cmd_vel sources fight, and real_driver.py
holds whatever it heard last. It refuses to start if /cmd_vel already has a publisher.
"""

import os
import sys
import json
import time
import signal
import argparse

import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy, ReliabilityPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool, String

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_DIR)

from Configs.config_loader import load_config

# Same table as MujocoEvaluator._evaluation_loop in Mujoco/eval_mujoco.py, copied rather than
# imported: that module pulls in mujoco and the whole pipeline. Keep the two in step - a speed that
# exists in only one of them has nothing to be compared against in the viewer.
SWEEP_SPEEDS = {
    "x": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.00],
    "y": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50],
    "yaw": [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00],
}
SPEED_UNITS = {"x": "m/s", "y": "m/s", "yaw": "rad/s"}

# F710 in X (XInput) mode, see Configs/joy_f710.config.yaml.
BTN_LB, BTN_RB, BTN_BACK, BTN_START = 4, 5, 6, 7
# joy convention, same as the sticks: left and up are positive. If the D-pad comes out mirrored it
# only moves the selection the wrong way, and the status line shows that before anything runs.
AXIS_DPAD_X, AXIS_DPAD_Y = 6, 7
BUTTON_NAMES = {0: "A", 1: "B", 2: "X", 3: "Y"}

# joy_node republishes at autorepeat_rate (20 Hz) even with nothing pressed, so a silence this long
# means the pad or joy_node is gone - and the deadman with it, so the segment stops.
JOY_TIMEOUT_S = 0.3
# The console publishes /pipeline/mode at 10 Hz.
MODE_TIMEOUT_S = 1.0

_BOLD, _RED, _YELLOW, _GREEN, _DIM, _RESET = (
    "\033[1m", "\033[91m", "\033[93m", "\033[92m", "\033[2m", "\033[0m")


def load_free_drive_cfg():
    """Stick axes and (non-turbo) scales from the teleop_twist_joy config, so a sign fixed there is
    fixed here too."""
    path = os.path.join(REPO_DIR, "Configs", "joy_f710.config.yaml")
    try:
        with open(path) as f:
            p = yaml.safe_load(f)["teleop_twist_joy_node"]["ros__parameters"]
    except Exception:
        p = {}
    lin_axis, lin_scale = p.get("axis_linear", {}), p.get("scale_linear", {})
    ang_axis, ang_scale = p.get("axis_angular", {}), p.get("scale_angular", {})
    return {
        "x": (int(lin_axis.get("x", 1)), float(lin_scale.get("x", 0.5))),
        "y": (int(lin_axis.get("y", 0)), float(lin_scale.get("y", 0.4))),
        "yaw": (int(ang_axis.get("yaw", 3)), float(ang_scale.get("yaw", 0.5))),
    }


class SweepTeleop(Node):
    def __init__(self, args):
        super().__init__("sweep_teleop")
        self.axes = args.axes
        self.walk_s = float(args.walk_s)
        self.walk_m = float(args.walk_m)
        self.ramp_s = float(args.ramp_s)
        self.warmup_s = float(args.warmup_s)
        self.checkpoint = args.checkpoint or ""
        self.robot = args.robot
        self.session = time.strftime("%Y%m%d_%H%M%S")

        safety_cfg = load_config().get("safety", {})
        self.estop_buttons = [int(b) for b in safety_cfg.get("estop_joy_buttons", [0, 1, 2, 3])]
        self.free_drive = load_free_drive_cfg()

        self._refuse_if_cmd_vel_taken()

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.estop_pub = self.create_publisher(Bool, "/safety/estop", 10)
        # Transient local, so a recorder that joins after this node still receives session_start and
        # every marker so far. Those arrive all at once, which is why sweep_report.py times markers by
        # the "t" they carry rather than by when the recorder got them.
        marker_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1000,
                                reliability=ReliabilityPolicy.RELIABLE,
                                durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.state_pub = self.create_publisher(String, "/sweep/state", marker_qos)

        self.create_subscription(Joy, "/joy", self._joy_cb, 10)
        self.create_subscription(Bool, "/safety/estop_state", self._estop_state_cb, 10)
        self.create_subscription(String, "/pipeline/mode", self._mode_cb, 10)

        self.state = "idle"          # idle | warmup | running | estop
        self.finished = False
        self._stopped = False
        self.t_state = time.monotonic()
        self.axis_idx = 0
        self.speed_idx = 0
        self.seg = None
        self.seg_walk_s = 0.0        # walking time of the segment under way, see _walk_time
        self.seg_count = 0
        self.done = {}               # (axis, speed) -> segment number
        self.completed = []          # (segment, axis_idx, speed_idx), newest last, for Back
        self.cmd = (0.0, 0.0, 0.0)

        self.joy_buttons, self.joy_axes = [], []
        self._prev_buttons, self._prev_axes = None, None
        self.last_joy = None
        self.mode, self.last_mode = None, None
        self._last_status = 0.0

        self._emit("session_start", checkpoint=self.checkpoint, robot=self.robot,
                   walk_s=self.walk_s, walk_m=self.walk_m, ramp_s=self.ramp_s,
                   warmup_s=self.warmup_s, speeds={a: SWEEP_SPEEDS[a] for a in self.axes})
        self._print_legend()
        self.create_timer(1.0 / float(args.rate), self._tick)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _refuse_if_cmd_vel_taken(self):
        # Discovery needs a moment before the graph shows anyone else. Checked before this node
        # creates its own publisher, which count_publishers would otherwise include.
        time.sleep(1.5)
        others = self.get_publishers_info_by_topic("/cmd_vel")
        if others:
            names = ", ".join(sorted({i.node_name for i in others}))
            raise SystemExit(
                f"[Sweep] /cmd_vel already has a publisher ({names}). Stop the other teleop first: "
                f"two command sources fight, and the driver keeps whichever spoke last.")

    def _print_legend(self):
        speeds = "  ".join(f"{a}: {len(SWEEP_SPEEDS[a])} up to {SWEEP_SPEEDS[a][-1]:.2f} {SPEED_UNITS[a]}"
                           for a in self.axes)
        print(f"\n{_BOLD}=== Eval Sweep ({self.session}) ==={_RESET}")
        print(f"  checkpoint: {os.path.basename(self.checkpoint) or '(none given)'}")
        ramp = f", {self.ramp_s:.1f} s ramp" if self.ramp_s > 0 else ""
        if self.walk_m > 0:
            print(f"  per speed:  {self.warmup_s:.1f} s standing, then x/y walk {self.walk_m:.1f} m "
                  f"(at most {self.walk_s:.1f} s), 0.0 and yaw walk {self.walk_s:.1f} s{ramp}")
        else:
            print(f"  per speed:  {self.warmup_s:.1f} s standing + {self.walk_s:.1f} s walking{ramp}")
        print(f"  speeds:     {speeds}")
        linear = sorted({s for a in self.axes if a in ("x", "y") for s in SWEEP_SPEEDS[a] if s > 0})
        if self.walk_m > 0 and linear:
            print("  x/y time:   " + "  ".join(f"{s:.2f}->{self._walk_time('x', s):.1f}s" for s in linear))
        print(f"  {_BOLD}LB{_RESET} hold to run, release = done + next   "
              f"{_BOLD}D-pad{_RESET} left/right speed, up/down axis   "
              f"{_BOLD}RB{_RESET}+sticks drive   {_BOLD}Back{_RESET} redo last   "
              f"{_BOLD}Start{_RESET} quit   {_RED}{_BOLD}A/B/X/Y E-STOP{_RESET}")
        print(f"  {_DIM}Nothing is measured here - record the session and run "
              f"Tools/sweep_report.py on it.{_RESET}\n")

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    def _emit(self, event, **fields):
        # "t" is this machine's wall clock at the moment of the event.
        payload = {"event": event, "session": self.session, "t": time.time(), **fields}
        self.state_pub.publish(String(data=json.dumps(payload)))

    def _publish_cmd(self, x, y, yaw):
        self.cmd = (x, y, yaw)
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = float(x), float(y), float(yaw)
        self.cmd_pub.publish(msg)

    def _say(self, text):
        sys.stdout.write(f"\r\033[K{text}\n")
        sys.stdout.flush()
        self._last_status = 0.0

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    def _mode_cb(self, msg):
        self.mode = msg.data
        self.last_mode = time.monotonic()

    def _estop_state_cb(self, msg):
        """Follow the robot's latch, not our own request: it is the robot that decides."""
        if msg.data:
            if self.state != "estop":
                self._end_segment("aborted", "e-stop latched on the robot")
                self.state = "estop"
                self._publish_cmd(0.0, 0.0, 0.0)
                self._say(f"{_RED}{_BOLD}[Sweep] E-STOP latched on the robot.{_RESET}")
        elif self.state == "estop":
            self.state = "idle"
            self._say(f"{_YELLOW}[Sweep] E-STOP released at the robot. Idle.{_RESET}")

    def _joy_cb(self, msg):
        buttons, axes = list(msg.buttons), list(msg.axes)
        prev_b, prev_a = self._prev_buttons, self._prev_axes
        self._prev_buttons, self._prev_axes = buttons, axes
        self.joy_buttons, self.joy_axes = buttons, axes
        self.last_joy = time.monotonic()

        def held(b, arr=buttons):
            return b < len(arr) and bool(arr[b])

        # 1. Kill switch, in every state. As on the console, the first message has no previous state
        #    and a button already held counts as a press: a stuck button stops the robot.
        for b in self.estop_buttons:
            if held(b) and not (prev_b is not None and held(b, prev_b)):
                self._estop(f"gamepad button {BUTTON_NAMES.get(b, b)}")
                return

        # Everything else needs a baseline, so a held LB at startup does not start a segment.
        if prev_b is None:
            return

        def pressed(b):
            return held(b) and not held(b, prev_b)

        if self.state in ("warmup", "running"):
            if not held(BTN_LB):
                if self.state == "running":
                    # The operator's call that the run is good - usually the room ran out first.
                    self._end_segment("complete", "LB released")
                else:
                    self._abort("LB released during warmup")
            return

        if pressed(BTN_START):
            self._finish()
            return
        if self.state != "idle":
            return

        if pressed(BTN_BACK):
            self._discard_last()
            return
        dx = self._dpad_edge(axes, prev_a, AXIS_DPAD_X)
        dy = self._dpad_edge(axes, prev_a, AXIS_DPAD_Y)
        if dx:
            self._move_speed(-dx)    # left is +1 -> previous speed
        if dy:
            self._move_axis(-dy)     # up is +1 -> previous axis
        if pressed(BTN_LB) and not held(BTN_RB):
            self._start_segment()

    @staticmethod
    def _dpad_edge(axes, prev, i):
        cur = axes[i] if i < len(axes) else 0.0
        old = prev[i] if prev is not None and i < len(prev) else 0.0
        if abs(cur) > 0.5 and abs(old) <= 0.5:
            return 1 if cur > 0 else -1
        return 0

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def _selected(self):
        axis = self.axes[self.axis_idx]
        return axis, SWEEP_SPEEDS[axis][self.speed_idx]

    def _sequence(self):
        """Every (axis_idx, speed_idx) in sweep order."""
        return [(ai, si) for ai, a in enumerate(self.axes) for si in range(len(SWEEP_SPEEDS[a]))]

    def _move_speed(self, step):
        # Across axes, so that left right after finishing an axis's last speed still goes back to it.
        order = self._sequence()
        i = order.index((self.axis_idx, self.speed_idx)) + step
        self.axis_idx, self.speed_idx = order[max(0, min(len(order) - 1, i))]

    def _move_axis(self, step):
        self.axis_idx = (self.axis_idx + step) % len(self.axes)
        axis = self.axes[self.axis_idx]
        undone = [i for i, s in enumerate(SWEEP_SPEEDS[axis]) if (axis, s) not in self.done]
        self.speed_idx = undone[0] if undone else 0

    def _advance(self):
        """Select the next speed not yet done, walking forward through the axes."""
        order = self._sequence()
        start = order.index((self.axis_idx, self.speed_idx))
        for ai, si in order[start + 1:] + order[:start + 1]:
            axis = self.axes[ai]
            if (axis, SWEEP_SPEEDS[axis][si]) not in self.done:
                self.axis_idx, self.speed_idx = ai, si
                return
        self._say(f"{_GREEN}{_BOLD}[Sweep] Every speed is done. Start ends the session.{_RESET}")

    # ------------------------------------------------------------------
    # Segments
    # ------------------------------------------------------------------
    def _joy_fresh(self, now):
        return self.last_joy is not None and now - self.last_joy < JOY_TIMEOUT_S

    def _mode_ok(self, now):
        return (self.mode == "policy" and self.last_mode is not None
                and now - self.last_mode < MODE_TIMEOUT_S)

    def _walk_time(self, axis, speed):
        """Walking time for one speed.

        With --walk_m, x and y cover that distance at the commanded speed - ramp included - and walk
        no longer than walk_s, or the slow end would take minutes. 0.0 and yaw cover no distance, so
        they walk walk_s. The robot's real speed differs from the command, so the distance it covers
        does too; the room needs some slack beyond walk_m."""
        if self.walk_m <= 0 or axis not in ("x", "y") or speed <= 0:
            return self.walk_s
        if self.walk_m / speed >= self.ramp_s / 2:
            # The ramp covers half of its own time's worth of distance.
            t = self.walk_m / speed + self.ramp_s / 2
        else:
            # The distance runs out before the ramp reaches the speed.
            t = (2 * self.walk_m * self.ramp_s / speed) ** 0.5
        return min(self.walk_s, t)

    def _start_segment(self):
        now = time.monotonic()
        # A segment walked in pose mode is the stand pose with a velocity command nobody reads.
        if not self._mode_ok(now):
            why = (f"pipeline is in '{self.mode}' mode" if self.mode and self.last_mode
                   and now - self.last_mode < MODE_TIMEOUT_S else "no /pipeline/mode from the console")
            self._say(f"{_YELLOW}[Sweep] Not starting: {why}. Switch to policy on the console.{_RESET}")
            return
        axis, speed = self._selected()
        self.seg_count += 1
        self.seg = {"segment": self.seg_count, "axis": axis, "speed": speed}
        self.seg_walk_s = self._walk_time(axis, speed)
        self.state = "warmup"
        self.t_state = now
        self._emit("segment_start", **self.seg, checkpoint=self.checkpoint, robot=self.robot,
                   warmup_s=self.warmup_s, walk_s=self.seg_walk_s, walk_m=self.walk_m,
                   ramp_s=self.ramp_s)
        self._say(f"[Sweep] #{self.seg_count} {axis} = {speed:.2f} {SPEED_UNITS[axis]}: "
                  f"standing {self.warmup_s:.1f} s, then walking {self.seg_walk_s:.1f} s")

    def _end_segment(self, status, reason=""):
        if self.state not in ("warmup", "running") or self.seg is None:
            return
        walked = time.monotonic() - self.t_state if self.state == "running" else 0.0
        self._emit("segment_end", **self.seg, status=status, reason=reason, walked_s=round(walked, 2))
        seg, self.seg = self.seg, None
        self.state = "idle"
        self._publish_cmd(0.0, 0.0, 0.0)
        if status == "complete":
            self.done[(seg["axis"], seg["speed"])] = seg["segment"]
            self.completed.append((seg["segment"], self.axis_idx, self.speed_idx))
            self._say(f"{_GREEN}[Sweep] #{seg['segment']} {seg['axis']} = {seg['speed']:.2f} "
                      f"done after {walked:.1f} s" + (f" ({reason})" if reason else "")
                      + f". Left goes back to it.{_RESET}")
            self._advance()
        else:
            self._say(f"{_YELLOW}[Sweep] #{seg['segment']} {seg['axis']} = {seg['speed']:.2f} "
                      f"aborted: {reason}. LB to run it again.{_RESET}")

    def _abort(self, reason):
        self._end_segment("aborted", reason)

    def _discard_last(self):
        if not self.completed:
            self._say("[Sweep] Nothing to discard.")
            return
        segment, ai, si = self.completed.pop()
        axis = self.axes[ai]
        speed = SWEEP_SPEEDS[axis][si]
        self.done.pop((axis, speed), None)
        self.axis_idx, self.speed_idx = ai, si
        self._emit("discard", segment=segment, axis=axis, speed=speed)
        self._say(f"{_YELLOW}[Sweep] #{segment} {axis} = {speed:.2f} discarded and selected again."
                  f"{_RESET}")

    def _estop(self, source):
        """Latch the robot's e-stop. Releasing it stays where it always is: ENTER on the driver."""
        for _ in range(3):
            self.estop_pub.publish(Bool(data=True))
        self._end_segment("aborted", f"e-stop ({source})")
        self._publish_cmd(0.0, 0.0, 0.0)
        self.state = "estop"
        self._say(f"{_RED}{_BOLD}[Sweep] E-STOP sent ({source}). Release it with ENTER on the "
                  f"driver terminal.{_RESET}")

    def _finish(self):
        self.stop("Start pressed")
        self.finished = True

    def stop(self, reason):
        """Zero the command and close the session.

        Published a few times with a flush window: real_driver.py keeps the last /cmd_vel it heard
        for as long as it runs, so this message is what stops the robot."""
        if self._stopped:
            return
        self._stopped = True
        try:
            self._end_segment("aborted", reason)
            self._emit("session_end", reason=reason)
            for _ in range(3):
                self._publish_cmd(0.0, 0.0, 0.0)
            time.sleep(0.1)
        except Exception:
            pass
        sys.stdout.write(f"\r\033[K[Sweep] Session ended ({reason}).\n")
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------
    def _tick(self):
        if self._stopped:
            return
        now = time.monotonic()

        if self.state in ("warmup", "running"):
            if not self._joy_fresh(now):
                self._abort(f"gamepad silent for more than {JOY_TIMEOUT_S:.1f} s")
            elif not self._mode_ok(now):
                self._abort("pipeline left policy mode" if self.mode != "policy"
                            else "lost /pipeline/mode from the console")

        cmd = (0.0, 0.0, 0.0)
        if self.state == "warmup" and now - self.t_state >= self.warmup_s:
            self.state = "running"
            self.t_state = now
            self._emit("walk_start", **self.seg)
        elif self.state == "running":
            t = now - self.t_state
            if t >= self.seg_walk_s:
                self._end_segment("complete")
            else:
                k = min(1.0, t / self.ramp_s) if self.ramp_s > 0 else 1.0
                v = self.seg["speed"] * k
                cmd = {"x": (v, 0.0, 0.0), "y": (0.0, v, 0.0), "yaw": (0.0, 0.0, v)}[self.seg["axis"]]
        elif self.state == "idle" and self._joy_fresh(now) and BTN_RB < len(self.joy_buttons) \
                and self.joy_buttons[BTN_RB]:
            cmd = tuple(
                (self.joy_axes[ax] * scale if ax < len(self.joy_axes) else 0.0)
                for ax, scale in (self.free_drive["x"], self.free_drive["y"], self.free_drive["yaw"]))

        self._publish_cmd(*cmd)

        if now - self._last_status >= 0.1:
            self._last_status = now
            self._draw_status(now)

    def _draw_status(self, now):
        if self.state == "idle":
            axis, speed = self._selected()
            n = len(SWEEP_SPEEDS[axis])
            tag = "done - LB runs it again" if (axis, speed) in self.done else "LB to run"
            body = (f"IDLE    {axis} = {speed:.2f} {SPEED_UNITS[axis]} ({self.speed_idx + 1}/{n}), "
                    f"{self._walk_time(axis, speed):.1f} s  {tag}")
            if any(abs(c) > 1e-6 for c in self.cmd):
                body += f"  {_DIM}driving {self.cmd[0]:+.2f} {self.cmd[1]:+.2f} {self.cmd[2]:+.2f}{_RESET}"
        elif self.state == "warmup":
            body = (f"{_BOLD}WARMUP{_RESET}  {self.seg['axis']} = {self.seg['speed']:.2f}  "
                    f"standing {now - self.t_state:4.1f}/{self.warmup_s:.1f} s")
        elif self.state == "running":
            body = (f"{_GREEN}{_BOLD}RUN{_RESET}     {self.seg['axis']} = {self.seg['speed']:.2f}  "
                    f"{now - self.t_state:4.1f}/{self.seg_walk_s:.1f} s")
        else:
            body = f"{_RED}{_BOLD}E-STOP{_RESET}  release with ENTER on the driver (Start quits)"

        progress = "  ".join(
            f"{a} {sum(1 for (da, _) in self.done if da == a)}/{len(SWEEP_SPEEDS[a])}"
            for a in self.axes)
        warn = ""
        if not self._joy_fresh(now):
            warn += f"  {_YELLOW}[no /joy - is the pad connected?]{_RESET}"
        if self.state == "idle" and not self._mode_ok(now):
            warn += f"  {_YELLOW}[pipeline: {self.mode or '?'}]{_RESET}"
        sys.stdout.write(f"\r\033[K[Sweep] {body}  | {progress}{warn}")
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Gamepad-paced eval sweep for the real robot")
    parser.add_argument("--robot", type=str, default="go2")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="checkpoint the robot is running; only recorded, for the report")
    parser.add_argument("--walk_s", type=float, default=10.0,
                        help="walking time per speed; with --walk_m, the most any speed walks")
    parser.add_argument("--walk_m", type=float, default=0.0,
                        help="walk x and y speeds this far instead of for walk_s (0 = off)")
    parser.add_argument("--warmup_s", type=float, default=2.0,
                        help="standing time before each speed (eval_mujoco.py uses 2 s)")
    parser.add_argument("--ramp_s", type=float, default=0.0,
                        help="ramp from 0 to the speed; 0 steps straight to it, like eval_mujoco.py")
    parser.add_argument("--axes", type=str, default="x,y,yaw")
    parser.add_argument("--rate", type=float, default=50.0, help="/cmd_vel publish rate")
    parser.add_argument("--max_lin_speed", type=float, default=1.0,
                        help="leave out x and y speeds above this, m/s (the table goes to 1.0)")
    parser.add_argument("--max_yaw_rate", type=float, default=1.0,
                        help="leave out yaw speeds above this, rad/s (the table goes to 1.0)")
    args = parser.parse_args()

    args.axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    bad = [a for a in args.axes if a not in SWEEP_SPEEDS]
    if bad or not args.axes:
        parser.error(f"--axes must be a subset of {','.join(SWEEP_SPEEDS)} (got {bad or 'nothing'})")
    if args.walk_s <= 0 or args.walk_m < 0 or args.warmup_s < 0 or args.ramp_s < 0:
        parser.error("--walk_s must be positive, --walk_m, --warmup_s and --ramp_s not negative")
    if args.max_lin_speed < 0 or args.max_yaw_rate < 0:
        parser.error("--max_lin_speed and --max_yaw_rate must not be negative")

    # Trim the table before the node reads it. Entries are dropped, never rescaled, so every speed
    # left still lines up with the same entry of a MuJoCo sweep in the viewer. 0.0 always stays.
    for axis, cap in (("x", args.max_lin_speed), ("y", args.max_lin_speed), ("yaw", args.max_yaw_rate)):
        SWEEP_SPEEDS[axis] = [s for s in SWEEP_SPEEDS[axis] if s <= cap + 1e-9]

    rclpy.init()
    try:
        node = SweepTeleop(args)
    except SystemExit:
        rclpy.shutdown()
        raise

    # Taken from rclpy for the same reason as in Operator/console.py: its own SIGINT handler tears
    # the context down before spin returns, and then the zero command cannot be published. A second
    # Ctrl-C hard-kills.
    def _sigint_handler(signum, frame):
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        node.stop("Ctrl-C")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    except BaseException:
        node.stop("node error")
        raise
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
