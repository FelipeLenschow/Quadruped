"""MCP server - the robot's tools for an LLM, over HTTP (MCP's streamable HTTP transport, JSON responses).

An LLM host (Claude Code, an Ollama loop, ...) on another machine connects to http://<robot>:<port>/mcp.
Only what the tools allow reaches the robot: skills through /skill and Nav2 goals (move too: a short goal from
where the robot stands), so Nav2's costmap keeps it off walls. The one velocity command is visit's last step,
from where Nav2 stopped to the trick's pose beside the object (slow, short, on /cmd_vel/llm, twist_mux's lowest
priority). Every motion ends inside its call.

Standard library only (the Jetson image is Python 3.8; the mcp package needs 3.10). config.yaml's mcp:.
Set QUADRUPED_MCP_TOKEN (or --token) to require "Authorization: Bearer <token>".
"""

import argparse
import base64
import contextlib
import json
import math
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from quadruped_common.config_loader import load_config
from quadruped_interfaces.srv import Skill

PROTOCOL_VERSIONS = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"]
SKILLS = ["walk", "stand", "sit", "lie", "wiggle", "localize"]
VISIT_SKILLS = ["sniff", "pee_left", "pee_right", "wiggle"]
SIDE = {"pee_left": 1.0, "pee_right": -1.0}
NAV_STATUS = {4: "arrived", 5: "canceled", 6: "aborted: no path or stuck"}

INSTRUCTIONS = """You control a Unitree Go2 quadruped. Positions are metres in the map frame; yaw and bearings are
radians/degrees, counter-clockwise, 0 along the map's x axis. Call skill('walk') before moving. Motions block
until done and report the result. A human with a joystick or the phone app can always take over.
The tricks are harmless dog poses for fun: sniff points the nose at an object, pee_left / pee_right lift a hind leg
beside it (nothing is released), wiggle wags the body."""


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def text(msg, error=False):
    return {"content": [{"type": "text", "text": msg}], "isError": error}


class Robot(Node):
    def __init__(self, cfg):
        super().__init__("mcp_server")
        self.cfg = cfg
        self.state = None
        self.state_time = 0.0
        self.objects = []
        self.visits = {}
        self.motion = threading.Lock()
        self.map_error = ""
        self.stopping = threading.Event()
        self.goal = None
        # TF only around tool calls: a Python TF listener decodes every /tf message (100/s on the robot, 20% of
        # an Orin Nano core with nothing asked of it). Opened by tf_in_use, closed tf_idle_s after the last call.
        self.tf_buffer = Buffer()
        self.tf_listener = None
        self.tf_lock = threading.Lock()
        self.tf_users = 0
        self.tf_opened = 0.0
        self.tf_last = 0.0
        self.create_timer(5.0, self._tf_close_idle)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/robot_state", self._state_cb, 10)
        self.create_subscription(String, "/objects", self._objects_cb, latched)
        self.grid = None
        self.create_subscription(OccupancyGrid, "/map", self._map_cb, latched)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel/llm", 10)
        self.skill_client = self.create_client(Skill, "/skill")
        try:
            from nav2_msgs.action import NavigateToPose
            from rclpy.action import ActionClient
            self.NavigateToPose = NavigateToPose
            self.nav = ActionClient(self, NavigateToPose, "navigate_to_pose")
        except ImportError:
            self.nav = None

    def _state_cb(self, msg):
        self.state = json.loads(msg.data)
        self.state_time = time.monotonic()

    def _map_cb(self, msg):
        self.grid = msg

    def _objects_cb(self, msg):
        self.objects = json.loads(msg.data).get("objects", [])
        ids = {o["id"] for o in self.objects}
        self.visits = {k: v for k, v in self.visits.items() if k in ids}

    def objects_on(self):
        """config.yaml mcp.objects: true, false, or auto (on while object_mapper publishes /objects)."""
        mode = self.cfg.get("objects", "auto")
        if mode == "auto":
            return self.count_publishers("/objects") > 0
        return bool(mode)

    def fresh(self):
        if self.state is None or time.monotonic() - self.state_time > 1.0:
            return None
        return self.state

    @contextlib.contextmanager
    def tf_in_use(self):
        with self.tf_lock:
            if self.tf_listener is None:
                self.tf_buffer = Buffer()  # new: the old one stopped being filled when it closed
                self.tf_listener = TransformListener(self.tf_buffer, self)
                self.tf_opened = time.monotonic()
            self.tf_users += 1
        try:
            yield
        finally:
            with self.tf_lock:
                self.tf_users -= 1
                self.tf_last = time.monotonic()

    def _tf_close_idle(self):
        with self.tf_lock:
            if (self.tf_listener is not None and self.tf_users == 0
                    and time.monotonic() - self.tf_last > float(self.cfg.get("tf_idle_s", 30.0))):
                self.tf_listener.unregister()
                self.tf_listener = None

    def pose(self):
        """(x, y, yaw, frame) of base_footprint: map when Nav2/SLAM runs, else odom. Just after tf_in_use opened
        the listener, waits up to a second for the first transforms."""
        end = self.tf_opened + 1.0
        while True:
            p = self._pose()
            if (p is not None and p[3] == "map") or time.monotonic() >= end:
                return p
            time.sleep(0.02)

    def _pose(self):
        for frame in ("map", "odom"):
            try:
                t = self.tf_buffer.lookup_transform(frame, "base_footprint", Time())
            except TransformException as e:
                if frame == "map":
                    self.map_error = str(e).splitlines()[0][:200]
                continue
            p = t.transform.translation
            return p.x, p.y, yaw_of(t.transform.rotation), frame
        return None

    def wait(self, future, timeout):
        end = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() > end:
                return False
            time.sleep(0.02)
        return True

    # --- tools ---

    def get_state(self):
        s = self.fresh()
        if s is None:
            return text("no /robot_state: is the driver running?", True)
        out = {k: s.get(k) for k in ("mode", "posture", "policy", "estop", "safety_blocked", "trip_reason",
                                     "numb", "torque_on", "tilt_deg", "velocity")}
        p = self.pose()
        out["position"] = None if p is None else {"x": round(p[0], 2), "y": round(p[1], 2),
                                                  "yaw_deg": round(math.degrees(p[2])), "frame": p[3]}
        out["navigating"] = self.goal is not None
        out["busy"] = self.motion.locked()
        return text(json.dumps(out))

    def list_objects(self, label=""):
        p = self.pose()
        out = []
        for o in self.objects:
            if label and label.lower() not in o["label"].lower():
                continue
            item = {"id": o["id"], "label": o["label"], "x": o["x"], "y": o["y"], "sightings": o["sightings"],
                    "done": self.visits.get(o["id"], [])}
            if p is not None:
                dx, dy = o["x"] - p[0], o["y"] - p[1]
                bearing = math.atan2(dy, dx) - p[2]
                item["distance_m"] = round(math.hypot(dx, dy), 2)
                item["bearing_deg"] = round(math.degrees(math.atan2(math.sin(bearing), math.cos(bearing))))
            out.append(item)
        out.sort(key=lambda o: o.get("distance_m", 0.0))
        if not out:
            return text("no objects mapped yet" + (f" matching '{label}'" if label else "")
                        + " (is object_mapper running?)")
        return text(json.dumps(out))

    def frontiers(self):
        g = self.grid
        if g is None:
            return text("no /map yet: is navigation (SLAM) running?", True)
        w, h, res = g.info.width, g.info.height, g.info.resolution
        ox, oy = g.info.origin.position.x, g.info.origin.position.y
        d = g.data
        clear = max(1, int(round(float(self.cfg.get("frontier_clearance", 0.2)) / res)))

        def free(i):
            return 0 <= d[i] < 25

        def near_wall(i):
            x, y = i % w, i // w
            for yy in range(max(0, y - clear), min(h, y + clear + 1)):
                row = yy * w
                for xx in range(max(0, x - clear), min(w, x + clear + 1)):
                    if d[row + xx] >= 65:
                        return True
            return False

        edge = set()
        for i in range(w, w * (h - 1)):
            x = i % w
            if 0 < x < w - 1 and free(i) and -1 in (d[i - 1], d[i + 1], d[i - w], d[i + w]) and not near_wall(i):
                edge.add(i)
        p = self.pose()
        p = p if p is not None and p[3] == "map" else None
        min_cells = float(self.cfg.get("frontier_min_m", 0.6)) / res
        out = []
        while edge:
            todo, cells = [edge.pop()], []
            while todo:
                i = todo.pop()
                cells.append(i)
                for n in (i - 1, i + 1, i - w, i + w, i - w - 1, i - w + 1, i + w - 1, i + w + 1):
                    if n in edge:
                        edge.remove(n)
                        todo.append(n)
            if len(cells) < min_cells:
                continue
            cx = sum(i % w for i in cells) / len(cells)
            cy = sum(i // w for i in cells) / len(cells)
            c = min(cells, key=lambda i: (i % w - cx) ** 2 + (i // w - cy) ** 2)
            item = {"x": round(ox + (c % w + 0.5) * res, 2), "y": round(oy + (c // w + 0.5) * res, 2),
                    "size_m": round(len(cells) * res, 1)}
            if p is not None:
                dx, dy = item["x"] - p[0], item["y"] - p[1]
                b = math.atan2(dy, dx) - p[2]
                item["distance_m"] = round(math.hypot(dx, dy), 1)
                item["bearing_deg"] = round(math.degrees(math.atan2(math.sin(b), math.cos(b))))
            out.append(item)
        out.sort(key=lambda o: o.get("distance_m", -o["size_m"]))
        explored = round(sum(1 for v in d if v >= 0) * res * res)
        if not out:
            return text(f"no frontiers: the map is closed ({explored} m2 explored)")
        return text(json.dumps({"explored_m2": explored, "frontiers": out[:int(self.cfg.get("frontiers_max", 8))]}))

    def go_to(self, x, y, yaw_deg=None):
        return self._motion("go_to", self._go_to, float(x), float(y), yaw_deg)

    def go_to_object(self, id, distance=None):
        o = next((o for o in self.objects if o["id"] == int(id)), None)
        if o is None:
            return text(f"no object {id}; call list_objects", True)
        p = self.pose()
        if p is None or p[3] != "map":
            return text(f"robot position on the map unknown ({self.map_error}): is navigation running?", True)
        d = float(distance if distance is not None else self.cfg.get("approach", 0.6))
        dx, dy = o["x"] - p[0], o["y"] - p[1]
        dist = math.hypot(dx, dy)
        yaw = math.atan2(dy, dx)
        gx, gy = (p[0], p[1]) if dist <= d else (o["x"] - d * dx / dist, o["y"] - d * dy / dist)
        return self._motion("go_to_object", self._go_to, gx, gy, math.degrees(yaw), o)

    def skill(self, name, seconds=0.0):
        name = str(name).strip().lower()
        if name in VISIT_SKILLS and name not in SKILLS:
            return text(f"{name} is done at an object: call visit('{name}', id) or visit('{name}', x=, y=)", True)
        if name not in SKILLS:
            return text(f"unknown skill '{name}'; one of {', '.join(SKILLS)}", True)
        return self._motion(name, self._skill, name, float(seconds or 0.0))

    def visit(self, trick, id=None, x=None, y=None):
        trick = str(trick).strip().lower()
        if trick not in VISIT_SKILLS:
            return text(f"unknown trick '{trick}'; one of {', '.join(VISIT_SKILLS)}", True)
        if id is None:
            if x is None or y is None:
                return text("give the object's id, or its x and y on the map", True)
            return self._motion("visit", self._visit, {"x": float(x), "y": float(y)}, trick)
        o = next((o for o in self.objects if o["id"] == int(id)), None)
        if o is None:
            return text(f"no object {id}; call list_objects", True)
        return self._motion("visit", self._visit, o, trick)

    def move(self, forward=0.0, left=0.0, turn_deg=0.0):
        return self._motion("move", self._move, float(forward or 0.0), float(left or 0.0), float(turn_deg or 0.0))

    def stop(self):
        self.stopping.set()
        goal = self.goal
        if goal is not None:
            self.wait(goal.cancel_goal_async(), 2.0)
            return text("stopped: navigation canceled")
        self.cmd_pub.publish(Twist())
        if self.motion.locked():
            return text("a skill is running; it ends on its own (the joystick or app can take over)")
        return text("stopped")

    def look(self, boxes=False):
        cam = self.cfg.get("camera") or "front"
        topic = f"/camera/{cam}/detections/image/compressed" if boxes else f"/camera/{cam}/image_raw/compressed"
        got = []
        sub = self.create_subscription(CompressedImage, topic, got.append, qos_profile_sensor_data)
        try:
            end = time.monotonic() + 3.0
            while not got and time.monotonic() < end:
                time.sleep(0.02)
        finally:
            self.destroy_subscription(sub)
        if not got:
            what = "object_detector" if boxes else "camera_bridge (launcher [I])"
            return text(f"no image on {topic} in 3 s: is {what} running?", True)
        data, mime = self._shrink(bytes(got[-1].data), got[-1].format)
        return {"content": [{"type": "image", "data": base64.b64encode(data).decode(), "mimeType": mime}],
                "isError": False}

    # --- helpers ---

    def _motion(self, what, fn, *args):
        if not self.motion.acquire(blocking=False):
            return text("busy with another motion; wait or call stop", True)
        self.stopping.clear()
        try:
            return fn(*args)
        except Exception as e:
            return text(f"{what} failed: {e}", True)
        finally:
            self.motion.release()

    def _ready(self):
        s = self.fresh()
        if s is None:
            return "no /robot_state: is the driver running?"
        if s["estop"]:
            return "e-stop is latched"
        if s["mode"] != "policy":
            return "not walking; call skill('walk') first"
        return None

    def _go_to(self, x, y, yaw_deg, obj=None):
        why = self._ready()
        if why:
            return text(why, True)
        if self.nav is None:
            return text("nav2_msgs is not installed here", True)
        if not self.nav.wait_for_server(timeout_sec=2.0):
            return text("Nav2 is not running (launcher [N])", True)
        p = self.pose()
        if yaw_deg is None:
            yaw = math.atan2(y - p[1], x - p[0]) if p else 0.0
        else:
            yaw = math.radians(float(yaw_deg))
        goal = self.NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = "map"
        goal.pose.pose.position.x, goal.pose.pose.position.y = x, y
        goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        sent = self.nav.send_goal_async(goal)
        if not self.wait(sent, 5.0) or not sent.result().accepted:
            return text("Nav2 refused the goal", True)
        self.goal = sent.result()
        try:
            result = self.goal.get_result_async()
            timeout = float(self.cfg.get("goal_timeout", 120.0))
            if not self.wait(result, timeout):
                self.wait(self.goal.cancel_goal_async(), 2.0)
                return text(f"not there after {timeout:.0f} s; canceled", True)
            status = result.result().status
        finally:
            self.goal = None
        p = self.pose()
        where = f" now at x={p[0]:.2f} y={p[1]:.2f}" if p else ""
        target = f"{obj['label']} {obj['id']}" if obj else f"({x:.2f}, {y:.2f})"
        if status == 4:
            # The planner's tolerance ends the path at the nearest free cell when the goal is in an obstacle, and Nav2
            # succeeds there: say so, or the LLM believes it arrived and keeps walking into the obstacle.
            short = math.hypot(x - p[0], y - p[1]) if p and obj is None else 0.0
            if short > float(self.cfg.get("arrive_tolerance", 0.3)):
                return text(f"stopped {short:.2f} m short of {target}: it is in or against an obstacle, the nearest "
                            f"free spot is as far as Nav2 goes;{where}")
            return text(f"arrived at {target};{where}")
        return text(f"{target}: {NAV_STATUS.get(status, f'status {status}')};{where}", True)

    def _visit(self, o, trick):
        p = self.pose()
        if p is None or p[3] != "map":
            return text(f"robot position on the map unknown ({self.map_error}): is navigation running?", True)
        dx, dy = p[0] - o["x"], p[1] - o["y"]
        d = math.hypot(dx, dy) or 1.0
        ux, uy = dx / d, dy / d
        side = SIDE.get(trick, 0.0)
        yaw = math.atan2(-uy, -ux) - side * math.pi / 2
        back = float(self.cfg.get("pee_back", 0.15)) if side else 0.0

        def spot(gap):
            return (o["x"] + ux * gap + math.cos(yaw) * back, o["y"] + uy * gap + math.sin(yaw) * back)

        gap = float(self.cfg.get("pee_distance", 0.4) if side else self.cfg.get("sniff_distance", 0.5))
        if trick == "wiggle":
            gap = float(self.cfg.get("approach", 0.6))
        nav_gap = max(gap, float(self.cfg.get("nav_standoff", 0.8)))
        nx, ny = spot(nav_gap)
        obj = o if "id" in o else None
        res = self._go_to(nx, ny, math.degrees(yaw), obj)
        if res["isError"] or self.stopping.is_set():
            return res
        if gap < nav_gap and self.cfg.get("final_step", True):
            why = self._final_step(*spot(gap), yaw)
            if why:
                return text(f"{why}; no {trick}", True)
        return self._skill(trick, 0.0, obj)

    def _final_step(self, x, y, yaw):
        p = self.pose()
        if p is None:
            return "lost the robot's position"
        if math.hypot(x - p[0], y - p[1]) > float(self.cfg.get("final_step_max", 0.6)):
            return f"Nav2 stopped {math.hypot(x - p[0], y - p[1]):.2f} m from the trick's spot, too far for the last step"
        vmax = float(self.cfg.get("final_step_speed", 0.15))
        end = time.monotonic() + float(self.cfg.get("final_step_s", 6.0))
        cmd = Twist()
        try:
            while time.monotonic() < end and not self.stopping.is_set():
                p = self.pose()
                if p is None:
                    return "lost the robot's position"
                ex, ey = x - p[0], y - p[1]
                eyaw = math.atan2(math.sin(yaw - p[2]), math.cos(yaw - p[2]))
                if math.hypot(ex, ey) < 0.04 and abs(eyaw) < 0.08:
                    break
                fx = math.cos(p[2]) * ex + math.sin(p[2]) * ey
                fy = -math.sin(p[2]) * ex + math.cos(p[2]) * ey
                k = min(1.5, vmax / max(1e-6, math.hypot(fx, fy)))
                cmd.linear.x, cmd.linear.y = k * fx, k * fy
                cmd.angular.z = max(-0.5, min(0.5, 1.5 * eyaw))
                self.cmd_pub.publish(cmd)
                time.sleep(0.05)
        finally:
            for _ in range(5):
                self.cmd_pub.publish(Twist())
                time.sleep(0.05)
        return "stopped" if self.stopping.is_set() else None

    def _skill(self, name, seconds, obj=None):
        if not self.skill_client.wait_for_service(timeout_sec=2.0):
            return text("skill_server is not running", True)
        future = self.skill_client.call_async(Skill.Request(name=name, duration=seconds))
        if not self.wait(future, 120.0):
            return text(f"{name}: no answer in 120 s", True)
        res = future.result()
        if res.success and name in VISIT_SKILLS:
            near = obj or self._nearest(float(self.cfg.get("visit_radius", 1.0)))
            if near is not None:
                self.visits.setdefault(near["id"], []).append(name)
                return text(f"{res.message} (at {near['label']} {near['id']})")
        return text(res.message, not res.success)

    def _nearest(self, radius):
        p = self.pose()
        if p is None:
            return None
        best = min(self.objects, key=lambda o: math.hypot(o["x"] - p[0], o["y"] - p[1]), default=None)
        if best is None or math.hypot(best["x"] - p[0], best["y"] - p[1]) > radius:
            return None
        return best

    def _move(self, forward, left, turn_deg):
        if forward == 0.0 and left == 0.0 and turn_deg == 0.0:
            return text("nothing to do: give forward, left or turn_deg", True)
        p = self.pose()
        if p is None or p[3] != "map":
            return text(f"robot position on the map unknown ({self.map_error}): is navigation running?", True)
        reach = float(self.cfg.get("max_move_m", 2.0))
        d = math.hypot(forward, left)
        if d > reach:
            forward, left = forward * reach / d, left * reach / d
        c, s = math.cos(p[2]), math.sin(p[2])
        x, y = p[0] + c * forward - s * left, p[1] + s * forward + c * left
        return self._go_to(x, y, math.degrees(p[2]) + turn_deg)

    def _shrink(self, data, fmt):
        mime = "image/png" if "png" in fmt else "image/jpeg"
        width = int(self.cfg.get("image_width", 640))
        try:
            import cv2
            import numpy as np
        except ImportError:
            return data, mime
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[1] <= width:
            return data, mime
        img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png" if mime == "image/png" else ".jpg", img)
        return (buf.tobytes(), mime) if ok else (data, mime)


def schema(props=None, required=()):
    return {"type": "object", "properties": props or {}, "required": list(required)}


TOOLS = [
    ("get_state", "The robot's mode, posture, safety flags and position on the map.", schema()),
    ("list_objects", "Objects seen by the camera and placed on the map, nearest first: id, label, position, "
     "distance and bearing from the robot (degrees, + is left), and the skills already done at each.",
     schema({"label": {"type": "string", "description": "only labels containing this, e.g. 'cone'"}})),
    ("frontiers", "Unexplored edges of the map: places where seen free floor meets space not seen yet, nearest "
     "first, with their size. go_to one, then turn and look to see what lies beyond. None left: the room is mapped.",
     schema()),
    ("go_to", "Walk to a point on the map with Nav2, avoiding obstacles. Blocks until there.",
     schema({"x": {"type": "number"}, "y": {"type": "number"},
             "yaw_deg": {"type": "number", "description": "heading on arrival; default: facing the way it walked"}},
            ["x", "y"])),
    ("go_to_object", "Walk up to an object from list_objects and stop facing it, a short distance away.",
     schema({"id": {"type": "integer"},
             "distance": {"type": "number", "description": "metres from the object (default 0.6)"}}, ["id"])),
    ("skill", "Run a skill and wait for it: walk (stand up and get ready to move), stand, sit, lie, wiggle, "
     "localize (find itself on a saved map). Tricks at an object go through visit.",
     schema({"name": {"type": "string", "enum": SKILLS},
             "seconds": {"type": "number", "description": "for tricks; 0 = the default length"}}, ["name"])),
    ("visit", "Walk to an object (its id from list_objects, or its x and y on the map), take the trick's pose there "
     "and do the trick: sniff (nose at it), pee_left / pee_right (it beside that hind leg), wiggle (facing it). "
     "Needs walk first.",
     schema({"trick": {"type": "string", "enum": VISIT_SKILLS}, "id": {"type": "integer"},
             "x": {"type": "number"}, "y": {"type": "number"}}, ["trick"])),
    ("move", "Step or turn from where the robot stands, with Nav2 (it goes around obstacles or refuses), e.g. to "
     "turn and look around. At most 2 m. Blocks until there.",
     schema({"forward": {"type": "number", "description": "metres, - is back"},
             "left": {"type": "number", "description": "metres, - is right"},
             "turn_deg": {"type": "number", "description": "degrees to turn, + is left"}})),
    ("stop", "Cancel navigation and stop moving.", schema()),
    ("look", "A photo from the front camera.",
     schema({"boxes": {"type": "boolean", "description": "with the object detector's boxes drawn"}})),
]
OBJECT_TOOLS = {"list_objects", "go_to_object"}
VISIT_AT_POINT = ("visit", "Walk to a point on the map (an object seen in a photo), take the trick's pose there and do "
                  "the trick: sniff (nose at it), pee_left / pee_right (it beside that hind leg), wiggle (facing "
                  "it). Needs walk first.",
                  schema({"trick": {"type": "string", "enum": VISIT_SKILLS},
                          "x": {"type": "number"}, "y": {"type": "number"}}, ["trick", "x", "y"]))


def tools_for(objects):
    """TOOLS, or without the object tools and look's boxes, and visit by point only, when there is no object
    detection."""
    if objects:
        return TOOLS
    return [VISIT_AT_POINT if n == "visit" else (n, d, schema() if n == "look" else s)
            for n, d, s in TOOLS if n not in OBJECT_TOOLS]


class Handler(BaseHTTPRequestHandler):
    robot = None
    token = ""
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body=None, headers=()):
        data = b"" if body is None else json.dumps(body).encode()
        self.send_response(code)
        if body is not None:
            self.send_header("Content-Type", "application/json")
        for k, v in headers:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _allowed(self):
        if self.path.split("?")[0].rstrip("/") != "/mcp":
            self._send(404)
            return False
        if self.headers.get("Origin"):
            self._send(403, {"error": "browser requests are not allowed"})
            return False
        if self.token and self.headers.get("Authorization", "") != f"Bearer {self.token}":
            self._send(401, {"error": "missing or wrong bearer token"})
            return False
        return True

    def do_GET(self):
        if self._allowed():
            self._send(405, headers=[("Allow", "POST")])

    do_DELETE = do_GET

    def do_POST(self):
        if not self._allowed():
            return
        try:
            msg = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        except ValueError:
            self._send(400, {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}})
            return
        replies = [r for r in (self._handle(m) for m in (msg if isinstance(msg, list) else [msg])) if r]
        if not replies:
            self._send(202)
        else:
            self._send(200, replies if isinstance(msg, list) else replies[0])

    def _handle(self, m):
        if "id" not in m:
            return None
        method, params = m.get("method"), m.get("params") or {}
        try:
            if method == "initialize":
                asked = params.get("protocolVersion")
                result = {"protocolVersion": asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                          "capabilities": {"tools": {"listChanged": False}},
                          "serverInfo": {"name": "go2", "version": "0.1.0"},
                          "instructions": INSTRUCTIONS}
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                tools = tools_for(self.robot.objects_on())
                result = {"tools": [{"name": n, "description": d, "inputSchema": s} for n, d, s in tools]}
            elif method == "tools/call":
                name = params.get("name")
                if name not in [t[0] for t in tools_for(self.robot.objects_on())]:
                    why = " (object detection is off)" if name in OBJECT_TOOLS else ""
                    return {"jsonrpc": "2.0", "id": m["id"],
                            "error": {"code": -32602, "message": f"unknown tool {name}{why}"}}
                args = params.get("arguments") or {}
                self.robot.get_logger().info(f"{name} {json.dumps(args)}")
                with self.robot.tf_in_use():
                    result = getattr(self.robot, name)(**args)
                self.robot.get_logger().info(f"{name} -> {result['content'][0].get('text', 'image')[:200]}")
            else:
                return {"jsonrpc": "2.0", "id": m["id"],
                        "error": {"code": -32601, "message": f"method not found: {method}"}}
        except TypeError as e:
            result = text(f"bad arguments: {e}", True)
        except Exception as e:
            result = text(f"{method} failed: {e}", True)
        return {"jsonrpc": "2.0", "id": m["id"], "result": result}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="", help="default config.yaml mcp.host")
    ap.add_argument("--port", type=int, default=0, help="default config.yaml mcp.port")
    ap.add_argument("--token", default=os.environ.get("QUADRUPED_MCP_TOKEN", ""))
    ap.add_argument("--objects", choices=["auto", "on", "off"], default="",
                    help="object tools (list_objects, go_to_object, visit); default config.yaml mcp.objects")
    args = ap.parse_args(remove_ros_args()[1:])
    cfg = load_config().get("mcp") or {}
    if args.objects:
        cfg["objects"] = {"auto": "auto", "on": True, "off": False}[args.objects]

    rclpy.init()
    robot = Robot(cfg)
    executor = MultiThreadedExecutor()
    executor.add_node(robot)
    threading.Thread(target=executor.spin, daemon=True).start()

    Handler.robot, Handler.token = robot, args.token
    host, port = args.host or cfg.get("host", "0.0.0.0"), args.port or int(cfg.get("port", 8765))
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    robot.get_logger().info(f"MCP on http://{host}:{port}/mcp" + (" (token required)" if args.token else "")
                            + f", object tools: {cfg.get('objects', 'auto')}")
    try:
        server.serve_forever()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        server.server_close()
        executor.shutdown()
        robot.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
