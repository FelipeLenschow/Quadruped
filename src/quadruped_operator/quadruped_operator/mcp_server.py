"""MCP server - the robot's tools for an LLM, over HTTP (MCP's streamable HTTP transport, JSON responses).

An LLM host (Claude Code, an Ollama loop, ...) on another machine connects to http://<robot>:<port>/mcp.
Only what the tools allow reaches the robot: skills through /skill, Nav2 goals, short moves on /cmd_vel/llm
(twist_mux's lowest priority). Every motion ends inside its call, so a lost link leaves the robot still.

Standard library only (the Jetson image is Python 3.8; the mcp package needs 3.10). config.yaml's mcp:.
Set QUADRUPED_MCP_TOKEN (or --token) to require "Authorization: Bearer <token>".
"""

import argparse
import base64
import json
import math
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
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
SKILLS = ["walk", "stand", "sit", "lie", "sniff", "pee_left", "pee_right", "wiggle", "localize"]
VISIT_SKILLS = {"sniff", "pee_left", "pee_right", "wiggle"}
NAV_STATUS = {4: "arrived", 5: "canceled", 6: "aborted: no path or stuck"}

INSTRUCTIONS = """You control a Unitree Go2 quadruped. Positions are metres in the map frame; yaw and bearings are
radians/degrees, counter-clockwise, 0 along the map's x axis. Call skill('walk') before moving. Motions block
until done and report the result. A human with a joystick or the phone app can always take over."""


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
        self.stopping = threading.Event()
        self.goal = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/robot_state", self._state_cb, 10)
        self.create_subscription(String, "/objects", self._objects_cb, latched)
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

    def _objects_cb(self, msg):
        self.objects = json.loads(msg.data).get("objects", [])
        ids = {o["id"] for o in self.objects}
        self.visits = {k: v for k, v in self.visits.items() if k in ids}

    def fresh(self):
        if self.state is None or time.monotonic() - self.state_time > 1.0:
            return None
        return self.state

    def pose(self):
        """(x, y, yaw, frame) of base_footprint: map when Nav2/SLAM runs, else odom."""
        for frame in ("map", "odom"):
            try:
                t = self.tf_buffer.lookup_transform(frame, "base_footprint", Time())
            except TransformException:
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

    def go_to(self, x, y, yaw_deg=None):
        return self._motion("go_to", self._go_to, float(x), float(y), yaw_deg)

    def go_to_object(self, id, distance=None):
        o = next((o for o in self.objects if o["id"] == int(id)), None)
        if o is None:
            return text(f"no object {id}; call list_objects", True)
        p = self.pose()
        if p is None or p[3] != "map":
            return text("robot position on the map unknown: is navigation running?", True)
        d = float(distance if distance is not None else self.cfg.get("approach", 0.6))
        dx, dy = o["x"] - p[0], o["y"] - p[1]
        dist = math.hypot(dx, dy)
        yaw = math.atan2(dy, dx)
        gx, gy = (p[0], p[1]) if dist <= d else (o["x"] - d * dx / dist, o["y"] - d * dy / dist)
        return self._motion("go_to_object", self._go_to, gx, gy, math.degrees(yaw), o)

    def skill(self, name, seconds=0.0):
        name = str(name).strip().lower()
        if name not in SKILLS:
            return text(f"unknown skill '{name}'; one of {', '.join(SKILLS)}", True)
        return self._motion(name, self._skill, name, float(seconds or 0.0))

    def move(self, forward=0.0, left=0.0, turn=0.0, seconds=1.0):
        return self._motion("move", self._move, float(forward), float(left), float(turn), float(seconds))

    def stop(self):
        self.stopping.set()
        notes = []
        goal = self.goal
        if goal is not None:
            self.wait(goal.cancel_goal_async(), 2.0)
            notes.append("navigation canceled")
        for _ in range(10):
            self.cmd_pub.publish(Twist())
            time.sleep(0.05)
        if not self.motion.locked():
            self.stopping.clear()
        elif not notes:
            notes.append("a skill is running; it ends on its own (the joystick or app can take over)")
        return text("stopped" + (": " + "; ".join(notes) if notes else ""))

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
            return text(f"arrived at {target};{where}")
        return text(f"{target}: {NAV_STATUS.get(status, f'status {status}')};{where}", True)

    def _skill(self, name, seconds):
        if not self.skill_client.wait_for_service(timeout_sec=2.0):
            return text("skill_server is not running", True)
        future = self.skill_client.call_async(Skill.Request(name=name, duration=seconds))
        if not self.wait(future, 120.0):
            return text(f"{name}: no answer in 120 s", True)
        res = future.result()
        if res.success and name in VISIT_SKILLS:
            near = self._nearest(float(self.cfg.get("visit_radius", 1.0)))
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

    def _move(self, forward, left, turn, seconds):
        why = self._ready()
        if why:
            return text(why, True)
        vmax, wmax = float(self.cfg.get("max_speed", 0.4)), float(self.cfg.get("max_turn", 0.8))
        cmd = Twist()
        cmd.linear.x = max(-vmax, min(vmax, forward))
        cmd.linear.y = max(-vmax, min(vmax, left))
        cmd.angular.z = max(-wmax, min(wmax, turn))
        seconds = max(0.0, min(float(self.cfg.get("max_move_s", 3.0)), seconds))
        end = time.monotonic() + seconds
        while time.monotonic() < end and not self.stopping.is_set():
            self.cmd_pub.publish(cmd)
            time.sleep(0.05)
        for _ in range(5):
            self.cmd_pub.publish(Twist())
            time.sleep(0.05)
        if self.stopping.is_set():
            return text("move stopped early")
        return text(f"moved {cmd.linear.x:.2f} m/s forward, {cmd.linear.y:.2f} m/s left, "
                    f"{cmd.angular.z:.2f} rad/s turn for {seconds:.1f} s")

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
    ("go_to", "Walk to a point on the map with Nav2, avoiding obstacles. Blocks until there.",
     schema({"x": {"type": "number"}, "y": {"type": "number"},
             "yaw_deg": {"type": "number", "description": "heading on arrival; default: facing the way it walked"}},
            ["x", "y"])),
    ("go_to_object", "Walk up to an object from list_objects and stop facing it, a short distance away.",
     schema({"id": {"type": "integer"},
             "distance": {"type": "number", "description": "metres from the object (default 0.6)"}}, ["id"])),
    ("skill", "Run a skill and wait for it: walk (stand up and get ready to move), stand, sit, lie, sniff, "
     "pee_left, pee_right, wiggle, localize (find itself on a saved map). Tricks need walk first.",
     schema({"name": {"type": "string", "enum": SKILLS},
             "seconds": {"type": "number", "description": "for tricks; 0 = the default length"}}, ["name"])),
    ("move", "Walk with a set velocity for a few seconds (at most 3), e.g. to turn and look around. Lowest "
     "priority: Nav2 and people override it.",
     schema({"forward": {"type": "number", "description": "m/s"}, "left": {"type": "number", "description": "m/s"},
             "turn": {"type": "number", "description": "rad/s, + is left"},
             "seconds": {"type": "number"}})),
    ("stop", "Cancel navigation and stop moving.", schema()),
    ("look", "A photo from the front camera.",
     schema({"boxes": {"type": "boolean", "description": "with the object detector's boxes drawn"}})),
]


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
                result = {"tools": [{"name": n, "description": d, "inputSchema": s} for n, d, s in TOOLS]}
            elif method == "tools/call":
                name = params.get("name")
                if name not in [t[0] for t in TOOLS]:
                    return {"jsonrpc": "2.0", "id": m["id"],
                            "error": {"code": -32602, "message": f"unknown tool {name}"}}
                args = params.get("arguments") or {}
                self.robot.get_logger().info(f"{name} {json.dumps(args)}")
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
    args = ap.parse_args(remove_ros_args()[1:])
    cfg = load_config().get("mcp") or {}

    rclpy.init()
    robot = Robot(cfg)
    executor = MultiThreadedExecutor()
    executor.add_node(robot)
    threading.Thread(target=executor.spin, daemon=True).start()

    Handler.robot, Handler.token = robot, args.token
    host, port = args.host or cfg.get("host", "0.0.0.0"), args.port or int(cfg.get("port", 8765))
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    robot.get_logger().info(f"MCP on http://{host}:{port}/mcp" + (" (token required)" if args.token else ""))
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
