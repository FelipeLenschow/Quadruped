#!/usr/bin/env python3
"""Thor trial: one agent trial of THOR_TESTS.md, recorded, in one command.

    python3 Tools/thor_trial.py t11_qwen8b_1 "Go out of this room, turn left, and stop next to the fridge." \\
        --model qwen3-vl:8b
    python3 Tools/thor_trial.py demo_corridor_1 "<task>" --model <model> --minutes 0 --max_calls 200 --lidar
    python3 Tools/thor_trial.py t07_30b_1 "<task>" --model <model> --keep 30         # unknown flags go to llm_host

Run it on the robot, in the ROS 2 environment, with the driver ([6]) and the MCP server running. It:
  1. starts the cameras (camera.launch.py, as launcher [I], with no detection) if they do not run yet,
  2. starts a new Nav2 (nav.launch.py, as launcher [N]): SLAM from an empty map, or --map <name> with AMCL at
     --initial_pose. A new one each trial, so each trial starts with nothing mapped; it refuses to start while
     another Nav2 runs (stop [N], or --keep_nav to use it),
  3. waits until Nav2 is active, /map has come and a camera frame has come,
  4. clears /objects (what find_something and object_mapper found in the last trial),
  5. starts the MCAP recorder (record.launch.py: the camera always, the lidar clouds with --lidar),
  6. runs Tools/llm_host.py --auto --done_tool with the fixed limits (--ctx 16384, --minutes 5, --max_calls 60
     unless given): the agent ends the task itself with done(success, summary), or the limits end it,
  7. waits for the last /map, then stops the recorder, Nav2 and the cameras it started,
  8. asks for the result (success, or the failure type of THOR_TESTS.md),
  9. makes the map video and the camera video (Tools/map_video.py; it needs opencv and mcap_ros2, which the
     Docker images have).

Everything goes into logs/thor/<name>/: trial.json (the task, the model, the arguments, the git commit, times, the
result), mcap/, agent/<time>/log.md with the agent's photos, record.log, nav.log, cameras.log, map.mp4, camera.mp4
(the robot's camera, real time), map_final.png, map.pgm/.yaml. Launcher [A] -> [2] runs this with a task from
Tools/agent_tasks.yaml.
A name that exists is never used again. Ctrl+C stops the agent (and the robot) and still saves everything; a second
Ctrl+C ends the agent at once. Once the agent has ended, Ctrl+C is ignored until the MCAP is closed and Nav2 and
the cameras are stopped. Needs rclpy (the ROS 2 environment) for the checks of step 3.
"""

import argparse
import glob
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLM_HOST = os.path.join(REPO, "Tools", "llm_host.py")
MAP_VIDEO = os.path.join(REPO, "Tools", "map_video.py")
NAME = re.compile(r"[A-Za-z0-9_-]{1,60}")
# THOR_TESTS.md, "For each trial, log".
FAILURES = ["wrong_tool", "bad_arguments", "loop", "gave_up", "timeout", "false_grounding", "stack_lag", "other"]
MAP_WAIT = 3.0  # slam_toolbox sends /map every 2 s (slam.yaml map_update_interval)
NAV_TIMEOUT, MAP_TIMEOUT, CAMERA_TIMEOUT = 90.0, 60.0, 45.0
# Nodes of nav.launch.py: one of them running means a Nav2 runs already.
NAV_NODES = {"lifecycle_manager_navigation", "bt_navigator", "controller_server", "planner_server", "slam_toolbox",
             "amcl"}
CAMERA_TOPIC = re.compile(r"^/camera/[^/]+/image_raw/compressed$")  # camera_bridge's, one per camera


class Abort(Exception):
    pass


class Graph:
    """A small rclpy node for the checks before a trial (imported here, so --help works without ROS 2)."""

    def __init__(self):
        import rclpy
        from rclpy.node import Node
        self.rclpy = rclpy
        rclpy.init()
        self.node = Node("thor_trial")
        self.spin(2.0)  # discovery: the graph is empty at first

    def spin(self, seconds, until=None):
        end = time.time() + seconds
        while time.time() < end:
            if until is not None and until():
                return True
            self.rclpy.spin_once(self.node, timeout_sec=0.1)
        return bool(until()) if until is not None else True

    def nodes(self):
        return set(self.node.get_node_names())

    def topics(self, pattern):
        return sorted(n for n, _ in self.node.get_topic_names_and_types() if pattern.match(n))

    def wait_message(self, topics, msg_type, qos, timeout, proc=None):
        """The first topic a message came on, or None. Stops early when proc (the launch that should publish) ends."""
        got = []
        subs = [self.node.create_subscription(msg_type, t, lambda m, t=t: got.append(t), qos) for t in topics]
        self.spin(timeout, lambda: bool(got) or (proc is not None and proc.poll() is not None))
        for sub in subs:
            self.node.destroy_subscription(sub)  # a camera subscriber keeps go2_camera sending
        return got[0] if got else None

    def nav_active(self, timeout, proc):
        """Nav2's lifecycle manager says all its nodes are active."""
        from std_srvs.srv import Trigger
        client = self.node.create_client(Trigger, "/lifecycle_manager_navigation/is_active")
        end = time.time() + timeout
        try:
            while time.time() < end and proc.poll() is None:
                if client.wait_for_service(timeout_sec=1.0):
                    future = client.call_async(Trigger.Request())
                    self.spin(5.0, future.done)
                    if future.done() and future.result() is not None and future.result().success:
                        return True
                self.spin(1.0)
            return False
        finally:
            self.node.destroy_client(client)

    def close(self):
        self.node.destroy_node()
        self.rclpy.shutdown()


def git(*cmd):
    try:
        out = subprocess.run(["git", "-C", REPO] + list(cmd), capture_output=True, text=True, timeout=10)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def save(path, info):
    with open(path + ".tmp", "w") as f:
        json.dump(info, f, indent=2)
    os.replace(path + ".tmp", path)


def clear_objects():
    cmd = ["ros2", "topic", "pub", "--once", "-w", "1", "/objects/clear", "std_msgs/msg/String", "{}"]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15, check=True)
        print("/objects cleared")
        return True
    except (subprocess.SubprocessError, OSError):
        print("WARNING: /objects not cleared (is the MCP server running?)")
        return False


def launch(name, log_path, **launch_args):
    """ros2 launch quadruped_bringup <name> in its own session (the terminal's Ctrl+C goes to the agent only).
    Empty values are left out (ros2 launch rejects 'name:='), as in the launcher."""
    args = [f"{k}:={str(v).lower() if isinstance(v, bool) else v}" for k, v in launch_args.items()
            if v not in ("", None)]
    return subprocess.Popen(["ros2", "launch", "quadruped_bringup", name] + args, stdout=open(log_path, "w"),
                            stderr=subprocess.STDOUT, start_new_session=True)


def stop_group(proc, name):
    """SIGINT, as Ctrl+C (ros2 bag then closes the MCAP and writes metadata.yaml). True when that was enough."""
    for sig, wait in ((signal.SIGINT, 30), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return True
        try:
            proc.wait(wait)
            return sig == signal.SIGINT
        except subprocess.TimeoutExpired:
            print(f"{name} did not stop on {sig.name}")
    return False


def start_cameras(graph, args, trial, started):
    """camera.launch.py when camera_bridge does not run; then waits for a frame. True when this trial started them."""
    ours = "camera_bridge" not in graph.nodes()
    if ours:
        print("starting the cameras ...")
        proc = launch("camera.launch.py", os.path.join(trial, "cameras.log"), sim=args.sim, interface=args.interface)
        started.append(("cameras", proc))
    else:
        print("the cameras run already ([I]): using them")
        proc = None
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage
    topics = []

    def found():
        topics[:] = graph.topics(CAMERA_TOPIC)
        return bool(topics) or (proc is not None and proc.poll() is not None)

    graph.spin(CAMERA_TIMEOUT, found)
    if not topics or not graph.wait_message(topics, CompressedImage, qos_profile_sensor_data, CAMERA_TIMEOUT, proc):
        raise Abort(f"no camera frame in {CAMERA_TIMEOUT:g} s (is the driver running?)"
                    + (f": see {os.path.join(trial, 'cameras.log')}" if ours else ""))
    print(f"camera: {', '.join(topics)}")
    return ours


def check_nav(graph, args):
    """Before anything starts: no Nav2 may run (one with --keep_nav)."""
    running = sorted(NAV_NODES & graph.nodes())
    if args.keep_nav and not running:
        raise Abort("--keep_nav, but no Nav2 runs: start [N], or leave out --keep_nav")
    if args.keep_nav:
        print(f"using the Nav2 that runs ({', '.join(running)})")
    elif running:
        raise Abort(f"a Nav2 runs already ({', '.join(running)}): stop [N]. The trial starts its own, so that each "
                    "trial starts with nothing mapped (--keep_nav uses the running one)")


def start_nav(graph, args, trial, started):
    """A new nav.launch.py, active, with a first /map. False with --keep_nav (the running one is used)."""
    if args.keep_nav:
        return False
    log_path = os.path.join(trial, "nav.log")
    print("starting Nav2: " + (f"map {args.map}, AMCL at {args.initial_pose}" if args.map else "SLAM, empty map")
          + " ...")
    proc = launch("nav.launch.py", log_path, sim=args.sim, rviz=args.rviz, map=args.map,
                  initial_pose=args.initial_pose, auto_localize=False)
    started.append(("Nav2", proc))
    if not graph.nav_active(NAV_TIMEOUT, proc):
        raise Abort(f"Nav2 not active in {NAV_TIMEOUT:g} s: see {log_path}")
    from nav_msgs.msg import OccupancyGrid
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    if not graph.wait_message(["/map"], OccupancyGrid, latched, MAP_TIMEOUT, proc):
        raise Abort(f"no /map in {MAP_TIMEOUT:g} s (no lidar scan?): see {log_path}")
    print("Nav2 active, first map in")
    return True


def start_recorder(mcap_dir, lidar, camera, log_path, started):
    proc = launch("record.launch.py", log_path, path=mcap_dir, lidar=lidar, camera=camera)
    started.append(("recorder", proc))
    deadline = time.time() + 30
    while time.time() < deadline:
        if proc.poll() is not None:
            raise Abort(f"the recorder stopped at once: see {log_path}")
        if glob.glob(os.path.join(mcap_dir, "*.mcap")):
            time.sleep(3.0)  # ros2 bag finds the topics and subscribes
            return proc
        time.sleep(0.5)
    raise Abort(f"the recorder made no .mcap in 30 s: see {log_path}")


def run_agent(cmd):
    """llm_host gets Ctrl+C too (same terminal): it stops the robot and ends. Here we only wait for it."""
    proc = subprocess.Popen(cmd)
    interrupted = 0
    while True:
        try:
            return proc.wait(), interrupted
        except KeyboardInterrupt:
            interrupted += 1
            if interrupted >= 2:
                print("\nending the agent")
                proc.terminate()


def ask_result(agent):
    if not sys.stdin.isatty():
        return {}
    end = agent.get("end", "?")
    claim = f": success {agent.get('success')}, \"{agent.get('summary', '')}\"" if end == "done" else ""
    print(f"\nThe run ended by {end}{claim}")
    try:
        a = input("\nResult? [s]uccess, [f]ail, Enter = fill in trial.json later: ").strip().lower()
        if a.startswith("s"):
            out = {"success": True}
        elif a.startswith("f"):
            print("  " + "  ".join(f"[{i + 1}] {f}" for i, f in enumerate(FAILURES)))
            k = input("Failure type: ").strip()
            out = {"success": False,
                   "failure": FAILURES[int(k) - 1] if k.isdigit() and 0 < int(k) <= len(FAILURES) else k or None}
        else:
            return {}
        note = input("Note (Enter = none): ").strip()
        if note:
            out["note"] = note
        return out
    except (EOFError, KeyboardInterrupt):
        print()
        return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 epilog="Other arguments (--keep, --think, --mcp, --ollama ...) go to llm_host.py.")
    ap.add_argument("name", help="trial name, for example t11_qwen8b_1 (letters, digits, _ and -)")
    ap.add_argument("task", help="the task prompt, as a person says it")
    ap.add_argument("--model", required=True)
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--max_calls", type=int, default=60)
    ap.add_argument("--ctx", type=int, default=16384, help="fixed in THOR_TESTS.md (the 8B needs it on the Orin NX)")
    ap.add_argument("--task_id", default="", help="the task's id in Tools/agent_tasks.yaml, for trial.json")
    ap.add_argument("--criterion", default="", help="the task's success criterion, for trial.json")
    ap.add_argument("--root", default=os.path.join(REPO, "logs", "thor"))
    ap.add_argument("--lidar", action="store_true", help="also record the lidar and LIO clouds (to run SLAM again)")
    ap.add_argument("--no_camera", action="store_true", help="do not record the camera")
    ap.add_argument("--map", default="", help="saved map in maps/ for AMCL (default: SLAM from an empty map)")
    ap.add_argument("--initial_pose", default="0,0,0", help="--map: the robot's start x,y,yaw on it")
    ap.add_argument("--keep_nav", action="store_true", help="use the Nav2 that runs ([N]); no new one")
    ap.add_argument("--sim", action="store_true", help="Gazebo: Nav2 on /clock, the cameras from Gazebo")
    ap.add_argument("--interface", default="", help="network interface for the Unitree SDK (go2_camera)")
    ap.add_argument("--rviz", action="store_true", help="also start rviz with Nav2")
    ap.add_argument("--no_clear", action="store_true", help="keep /objects from the last trial")
    ap.add_argument("--no_ask", action="store_true", help="do not ask for the result")
    ap.add_argument("--no_video", action="store_true", help="no map video now (make it later with map_video.py)")
    args, extra = ap.parse_known_args()
    sys.stdout.reconfigure(line_buffering=True)  # our lines in order with the children's, also through a pipe

    if not NAME.fullmatch(args.name):
        sys.exit(f"bad name {args.name!r}: use letters, digits, _ and -")
    if shutil.which("ros2") is None:
        sys.exit("no ros2: source ROS 2 and install/setup.bash (or run in Docker)")
    trial = os.path.join(os.path.abspath(args.root), args.name)
    if os.path.exists(trial):
        sys.exit(f"{trial} exists: give the trial a new name")
    os.makedirs(trial)
    path = os.path.join(trial, "trial.json")
    mcap_dir, agent_dir = os.path.join(trial, "mcap"), os.path.join(trial, "agent")
    result_path = os.path.join(trial, "agent_result.json")
    agent_cmd = [sys.executable, LLM_HOST, "--auto", "--done_tool", "--result_file", result_path, "--model", args.model,
                 "--minutes", str(args.minutes), "--max_calls", str(args.max_calls), "--ctx", str(args.ctx),
                 "--log_dir", agent_dir] + extra
    agent_cmd.append(args.task)
    info = {
        "name": args.name, "task_id": args.task_id, "task": args.task, "criterion": args.criterion,
        "model": args.model, "minutes": args.minutes, "max_calls": args.max_calls, "ctx": args.ctx,
        "llm_host_args": extra, "lidar": args.lidar, "camera": not args.no_camera,
        "nav": "running" if args.keep_nav else args.map or "slam",
        "initial_pose": args.initial_pose if args.map else None,
        "host": socket.gethostname(), "commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
        # log.md times are local; map_video needs this to put them on the MCAP's clock
        "utc_offset_s": time.localtime().tm_gmtoff,
        "start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "success": None, "failure": None,
    }
    save(path, info)

    started = []  # (name, process group) this trial started: stopped in reverse order at the end
    code, interrupted = None, 0
    try:
        graph = Graph()
        try:
            check_nav(graph, args)
            info["cameras_started"] = start_cameras(graph, args, trial, started)
            info["nav_started"] = start_nav(graph, args, trial, started)
        finally:
            graph.close()
        if not args.no_clear:
            info["objects_cleared"] = clear_objects()
        print(f"recording to {mcap_dir}")
        start_recorder(mcap_dir, args.lidar, not args.no_camera, os.path.join(trial, "record.log"), started)
        t0 = time.time()
        code, interrupted = run_agent(agent_cmd)
        info["agent_s"] = round(time.time() - t0, 1)
    except Abort as e:
        info["aborted"] = str(e)
    except KeyboardInterrupt:
        info["aborted"] = "Ctrl+C before the agent started"
    finally:
        # A Ctrl+C now (the launcher's third one kills us) would leave the MCAP without its summary.
        handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        if any(name == "recorder" for name, _ in started) and code is not None:
            print(f"waiting {MAP_WAIT:g} s for the last map ...")
            time.sleep(MAP_WAIT)
        for name, proc in reversed(started):
            print(f"stopping {name} ...")
            clean = stop_group(proc, name)
            if name == "recorder":
                info["recorder_clean_stop"] = clean
        info["end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        save(path, info)
        signal.signal(signal.SIGINT, handler)
    if "aborted" in info:
        sys.exit(f"ABORTED: {info['aborted']}\n(trial folder kept: {trial})")
    info["agent_exit"], info["interrupted"] = code, interrupted > 0
    try:
        with open(result_path) as f:
            # end: done (the agent's own success and summary), time, max_calls, interrupted, error
            info["agent"] = json.load(f)
    except (OSError, ValueError):
        info["agent"] = {}
    save(path, info)  # complete before the question: a kill there loses only the operator's result
    if code != 0 and not interrupted:
        print(f"WARNING: llm_host ended with code {code} (MCP server or Ollama down?): see its output above")
    if not os.path.isfile(os.path.join(mcap_dir, "metadata.yaml")):
        print(f"WARNING: no metadata.yaml in {mcap_dir}: the MCAP may be cut (see record.log)")
    if not args.no_ask:
        info.update(ask_result(info["agent"]))
    save(path, info)

    video = [sys.executable, MAP_VIDEO, trial]
    if not args.no_video:
        print("making the map video ...")
        if subprocess.run(video).returncode != 0:
            print("map video failed; later: " + " ".join(video))
    print(f"\ntrial: {trial}")


if __name__ == "__main__":
    main()
