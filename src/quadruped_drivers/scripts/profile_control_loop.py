#!/usr/bin/env python3
"""Profile quadruped_drivers/real_driver.py's control loop without moving the robot.

Builds the real driver control path -- LocomotionPipeline (TelemetryManager + LKF, PolicyRunner,
CommandSafetyProcessor, Distributor) plus send_to_sdk's CRC and DDS write -- around a synthetic
standing LowState_, and times every stage. Safe to run on the robot itself:

  * LowCmd is written to rt/lowcmd_profile_only, never rt/lowcmd, and SDK DDS is bound to the
    loopback interface. Nothing reaches the motors.
  * ROS runs on its own domain (77), so the robot's ROS graph never sees these topics.
  * The console heartbeat is faked in-process; no console is needed.

Do not run it while real_driver.py is running on the same machine -- not for safety, but
because the two would compete for the CPU and the numbers would mean nothing.

Modes:
  --mode suite   (default) the three phases below, one after the other, then a summary:
                   1. tight loop -- pure compute cost of each stage, nothing else running
                   2. the driver's timers, no LowState stream
                   3. the driver's timers + a 500 Hz LowState stream, as on the robot
  --mode tight   phase 1 only.
  --mode timer   the driver's own timers under rclpy.spin, exactly as deployed: reports the
                 policy-step period (target 20 ms), the LowCmd write interval (target 5 ms) and
                 the time from the start of a policy step to its motor write.
Options:
  --lowstate_hz N    (--mode timer) stream LowState_ at N Hz, like the robot's 500 Hz
                     rt/lowstate, into the driver's own on-demand reader. The stream comes from
                     a SEPARATE process, as the firmware's does on the robot, so only the
                     receiving side competes with the loop, exactly as in real_driver.py.
  --torch_threads N  run the policy with N intra-op threads (sets QUADRUPED_TORCH_THREADS);
                     0 keeps PolicyRunner's default of 1. Use the core count (6 on the Go2's
                     Orin Nano) to reproduce the old multithreaded behaviour.

Examples (inside the robot's Docker image, with install/setup.bash sourced):
  ros2 run quadruped_drivers profile_control_loop.py
  ros2 run quadruped_drivers profile_control_loop.py --mode timer --seconds 20 --lowstate_hz 500
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ros2 run quadruped_drivers profile_control_loop.py --mode tight
"""
import argparse
import collections
import contextlib
import glob
import io
import os
import re
import subprocess
import sys
import time

import numpy as np

from quadruped_core import paths

os.chdir(paths.REPO)

DEFAULT_CKPT = os.path.join(
    "IsaacLab_Tasks", "Simple", "logs", "rsl_rl",
    "unitree_go2_velocity_sigma_vel_foot_rough_deploy", "Noises", "model_2999.pt",
)
LOWSTATE_TOPIC = "rt/lowstate_profile_only"
ROBOT_LOWSTATE_HZ = 500
# SDK motor order (FR, FL, RR, RL x hip, thigh, calf), standing.
STANCE_SDK = [-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5]

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--mode", default="suite", choices=["suite", "tight", "timer"])
ap.add_argument("--pipeline_mode", default="policy", choices=["pose", "policy"])
ap.add_argument("--seconds", type=float, default=15.0, help="duration of each timer phase")
ap.add_argument("--steps", type=int, default=1000, help="policy steps in the tight phase")
ap.add_argument("--lowstate_hz", type=float, default=0.0)
ap.add_argument("--torch_threads", type=int, default=0)
ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="policy checkpoint (any format PolicyRunner loads)")
ap.add_argument("--_lowstate_publisher", type=float, default=0.0, help=argparse.SUPPRESS)
args = ap.parse_args()

os.environ.pop("CYCLONEDDS_URI", None)
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

ChannelFactoryInitialize(0, "lo")


def synthetic_low_state():
    msg = unitree_go_msg_dds__LowState_()
    for i, q in enumerate(STANCE_SDK):
        msg.motor_state[i].q = q
    msg.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
    msg.imu_state.accelerometer = [0.0, 0.0, 9.81]
    msg.foot_force = [30, 30, 30, 30]  # above the gate on all four feet: 4 LKF updates/step
    return msg


if args._lowstate_publisher:
    # Child process standing in for the robot firmware's rt/lowstate. Runs until killed.
    pub = ChannelPublisher(LOWSTATE_TOPIC, LowState_)
    pub.Init()
    msg = synthetic_low_state()
    period, nxt = 1.0 / args._lowstate_publisher, time.perf_counter()
    while True:
        pub.Write(msg)
        nxt += period
        time.sleep(max(0.0, nxt - time.perf_counter()))

if args.torch_threads > 0:
    os.environ["QUADRUPED_TORCH_THREADS"] = str(args.torch_threads)

os.environ["ROS_DOMAIN_ID"] = "77"
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

rclpy.init()

from quadruped_drivers import real_driver as rd
from quadruped_core.pipeline import LocomotionPipeline
import torch

# --------------------------------------------------------------------------- timing
T = collections.defaultdict(list)  # (stage, loop) -> [seconds]
_loop = ["control"]                # which loop the current call belongs to


def timed(obj, name, label):
    fn = getattr(obj, name)

    def wrapper(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            T[(label, _loop[0])].append(time.perf_counter() - t)

    setattr(obj, name, wrapper)


# --------------------------------------------------------------------------- driver
class ProfDriver(rd.RealDriver):
    """RealDriver with the hardware parts of __init__ replaced: no SportClient/MotionSwitcher,
    no VUI, LowCmd on a dummy topic. Everything the control loops touch is the real code."""

    def __init__(self):
        Node.__init__(self, "real_driver_profile")
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.WARN)
        self.robot_type = "go2"
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd_profile_only", LowCmd_)
        self.lowcmd_publisher.Init()
        self._init_lowstate_reader(LOWSTATE_TOPIC)
        self.low_state = None
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.crc = CRC()
        self.vui = type("NoVui", (), {"SetBrightness": staticmethod(lambda b: None)})()
        self._current_brightness = 0
        self.pipeline = LocomotionPipeline(node=self, robot_type="go2", checkpoint=args.ckpt,
                                           obs_dim=45, use_estimator=True, sim_dt=rd.POLICY_DT)
        self.pipeline.decimation = 1
        self.pipeline.policy_dt = rd.POLICY_DT
        self.cmds_vel = np.zeros(4)
        self.kp, self.kd, self.emergency_kd = 25.0, 0.5, 5.0
        self.contact_threshold = 10.0
        self.fsr_offset = np.full(4, 17.0)
        self._calib_done = True
        self._calib_samples = []
        self.foot_force_pub = self.create_publisher(Float32MultiArray, "/sensors/foot_force", 10)
        self._startup_console_check = False
        self._init_low_cmd()
        self._last_targets = None


build_log = io.StringIO()
with contextlib.redirect_stdout(build_log):
    drv = ProfDriver()
if "main" not in drv.pipeline.policy_manager.policies:
    print(build_log.getvalue())
    sys.exit(f"Policy failed to load from {args.ckpt} -- see the log above.")
for line in build_log.getvalue().splitlines():
    if "layout" in line and "Using" not in line or "WARNING" in line:
        print(line)

drv.low_state = synthetic_low_state()
rng = np.random.default_rng(0)

# Fake console: heartbeat always fresh, 55 % torque, chosen pipeline mode.
sp = drv.pipeline.safety_processor
sp.has_received_heartbeat = True
sp.max_torque_percent = 55.0
sp._recompute_safety_limits()
drv.pipeline.mode = args.pipeline_mode
drv.pipeline.mode_transition_active = False

timed(drv, "_read_low_state", "read LowState")
timed(drv, "_get_raw_sensor_data", "raw sensor read")
timed(drv.pipeline.telemetry, "process_state", "process_state (incl. LKF)")
timed(drv.pipeline.telemetry.estimator, "update", "  LKF update")
timed(drv.pipeline.policy_manager, "step_single", "policy / pose step")
runner = drv.pipeline.policy_manager.policies["main"]
timed(runner, "build_obs", "  build_obs")
timed(runner, "get_action", "  inference")
timed(sp, "process", "safety.process")
timed(drv.pipeline.distributor, "send", "distributor.send")
timed(drv.pipeline.telemetry, "publish", "telemetry.publish")
timed(drv, "send_to_sdk", "send_to_sdk")
timed(drv.crc, "Crc", "  CRC pack + crc")

policy_stamps, write_stamps = [], []
step_latency = []               # policy step start -> its motor write, seconds
fresh = [0]                     # policy steps that got a new LowState sample
_step = {"t0": 0.0, "written": True}
_write = drv.lowcmd_publisher.Write


def write_stamped(*a, **k):
    write_stamps.append(time.perf_counter())
    if _loop[0] == "control" and not _step["written"]:
        step_latency.append(write_stamps[-1] - _step["t0"])
        _step["written"] = True
    t = time.perf_counter()
    try:
        return _write(*a, **k)
    finally:
        T[("  DDS write", _loop[0])].append(time.perf_counter() - t)


drv.lowcmd_publisher.Write = write_stamped

_control, _lowcmd = rd.RealDriver.control_loop, rd.RealDriver.lowcmd_loop


def control_loop():
    _loop[0] = "control"
    sp.last_heartbeat_time = time.time()
    for i, q in enumerate(STANCE_SDK):  # a little sensor noise so nothing is cached
        drv.low_state.motor_state[i].q = q + float(rng.normal(0, 0.01))
        drv.low_state.motor_state[i].dq = float(rng.normal(0, 0.2))
    policy_stamps.append(time.perf_counter())
    last_sample = drv._lowstate_time
    t = time.perf_counter()
    _step.update(t0=t, written=False)
    _control(drv)
    T[("TOTAL", "control")].append(time.perf_counter() - t)
    fresh[0] += drv._lowstate_time != last_sample


def lowcmd_loop():
    _loop[0] = "lowcmd"
    t = time.perf_counter()
    _lowcmd(drv)
    T[("TOTAL", "lowcmd")].append(time.perf_counter() - t)


drv.control_loop, drv.lowcmd_loop = control_loop, lowcmd_loop


# --------------------------------------------------------------------------- environment
def process_threads():
    with open("/proc/self/status") as f:
        return int(re.search(r"Threads:\s+(\d+)", f.read()).group(1))


def environment_report():
    cfg = torch.__config__.show()
    blas = re.search(r"BLAS_INFO=(\w+)", cfg)
    print(f"torch {torch.__version__} | BLAS {blas.group(1) if blas else '?'} | "
          f"OpenMP {'on' if 'USE_OPENMP=ON' in cfg else 'off'} | "
          f"intra-op threads {torch.get_num_threads()} | "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '-')} "
          f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS', '-')}")
    try:
        cpus = sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq"))
        govs = {open(f"{c}/scaling_governor").read().strip() for c in cpus}
        cur = [int(open(f"{c}/scaling_cur_freq").read()) / 1000 for c in cpus]
        mx = [int(open(f"{c}/scaling_max_freq").read()) / 1000 for c in cpus]
        print(f"CPU: {len(os.sched_getaffinity(0))} usable | governor {','.join(sorted(govs))} | "
              f"now {min(cur):.0f}-{max(cur):.0f} MHz of max {max(mx):.0f} MHz")
    except (OSError, ValueError):
        print(f"CPU: {len(os.sched_getaffinity(0))} usable | frequency not readable here")


# --------------------------------------------------------------------------- phases
def run_tight(steps):
    for _ in range(steps):
        drv.control_loop()
        for _ in range(round(rd.POLICY_DT / rd.LOWCMD_DT) - 1):
            drv.lowcmd_loop()


def run_timer(seconds):
    timers = [drv.create_timer(rd.POLICY_DT, drv.control_loop),
              drv.create_timer(rd.LOWCMD_DT, drv.lowcmd_loop)]
    t_end = time.time() + seconds
    while time.time() < t_end:
        rclpy.spin_once(drv, timeout_sec=0.1)
    for timer in timers:
        drv.destroy_timer(timer)


def start_lowstate_stream(hz):
    """Publisher in a child process; the driver's own on-demand reader receives it here."""
    child = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--_lowstate_publisher", str(hz)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t_end = time.time() + 30.0
    while drv._lowstate_time is None and time.time() < t_end:
        drv._read_low_state()
        time.sleep(0.05)
    if drv._lowstate_time is None:
        print("WARNING: the LowState stream did not start; this phase runs without it.")
    return child


def reset_stats():
    T.clear()
    policy_stamps.clear()
    write_stamps.clear()
    step_latency.clear()
    fresh[0] = 0


def report(title, seconds=None):
    """Per-stage table; returns the numbers the suite summary needs."""
    print(f"\n=== {title}")
    print(f"{'stage':28s} {'loop':8s} {'n':>6s} {'mean ms':>8s} {'p50':>7s} {'p95':>7s} {'p99':>7s} {'max':>7s}")
    order = ["TOTAL", "read LowState", "raw sensor read", "process_state (incl. LKF)", "  LKF update", "policy / pose step",
             "  build_obs", "  inference", "safety.process", "distributor.send", "telemetry.publish",
             "send_to_sdk", "  CRC pack + crc", "  DDS write"]
    for stage in order:
        for loop in ("control", "lowcmd"):
            x = T.get((stage, loop))
            if not x:
                continue
            x = 1000 * np.asarray(x)
            print(f"{stage:28s} {loop:8s} {len(x):6d} {x.mean():8.3f} {np.median(x):7.3f} "
                  f"{np.percentile(x, 95):7.3f} {np.percentile(x, 99):7.3f} {x.max():7.2f}")

    def ms(key):
        x = 1000 * np.asarray(T.get(key) or [np.nan])
        return x.mean(), np.percentile(x, 99)

    lat = 1000 * np.asarray(step_latency or [np.nan])
    out = {"step": ms(("TOTAL", "control")), "inference": ms(("  inference", "control")),
           "lkf": ms(("  LKF update", "control")), "latency": (lat.mean(), np.percentile(lat, 99)),
           "hz": np.nan, "late": np.nan, "gap_max": np.nan, "gaps10": np.nan}
    if step_latency:
        print(f"step start -> motor write: mean {lat.mean():.2f} ms, p99 {np.percentile(lat, 99):.2f} ms, "
              f"max {lat.max():.2f} ms | new LowState sample on {fresh[0]} of {len(policy_stamps)} steps")
    if seconds and len(policy_stamps) > 10:
        d = 1000 * np.diff(policy_stamps)
        w = 1000 * np.diff(write_stamps)
        out.update(hz=1000 / d.mean(), late=int(np.sum(d > 22)), gap_max=w.max(), gaps10=int(np.sum(w > 10)))
        print(f"policy step period : median {np.median(d):.2f} ms, mean {d.mean():.2f} ms "
              f"-> {out['hz']:.1f} Hz (target {1000 * rd.POLICY_DT:.0f} ms)")
        print(f"  late steps       : >22 ms {out['late']}, >25 ms {np.sum(d > 25)}, max {d.max():.1f} ms "
              f"of {len(d)}")
        print(f"LowCmd write gap   : median {np.median(w):.2f} ms, p99 {np.percentile(w, 99):.2f} ms, "
              f"max {w.max():.2f} ms -> {len(write_stamps) / seconds:.0f} writes/s "
              f"(Unitree asks for a write every 1-10 ms)")
        print(f"  gaps > 10 ms     : {out['gaps10']} of {len(w)}")
    return out


# --------------------------------------------------------------------------- run
environment_report()
threads_before = process_threads()
for _ in range(50):  # warm-up: first inference, lazy thread pools, caches
    drv.control_loop()
print(f"process threads: {threads_before} before the first inference, {process_threads()} after "
      f"(a jump means torch/BLAS started its own worker pool)")
print(f"mode={args.mode} pipeline={args.pipeline_mode}")
reset_stats()

children = []
try:
    if args.mode == "tight":
        run_tight(args.steps)
        report(f"tight loop, {args.steps} policy steps")
    elif args.mode == "timer":
        if args.lowstate_hz > 0:
            children.append(start_lowstate_stream(args.lowstate_hz))
            reset_stats()
        run_timer(args.seconds)
        report(f"driver timers, {args.seconds:g} s, LowState stream {args.lowstate_hz:g} Hz", args.seconds)
    else:
        summary = []
        run_tight(args.steps)
        summary.append(("1 tight loop", report(f"1/3 tight loop, {args.steps} policy steps")))
        reset_stats()
        run_timer(args.seconds)
        summary.append(("2 timers", report(f"2/3 driver timers, {args.seconds:g} s, no LowState stream",
                                           args.seconds)))
        children.append(start_lowstate_stream(ROBOT_LOWSTATE_HZ))
        reset_stats()
        run_timer(args.seconds)
        summary.append(("3 timers+LowState", report(
            f"3/3 driver timers, {args.seconds:g} s, {ROBOT_LOWSTATE_HZ} Hz LowState from another process",
            args.seconds)))

        print(f"\n=== summary (ms: mean / p99)")
        print(f"{'phase':20s} {'control step':>15s} {'inference':>15s} {'LKF':>7s} {'state->cmd':>15s} "
              f"{'policy Hz':>10s} {'late >22ms':>11s} {'LowCmd max gap':>15s} {'gaps >10ms':>11s}")
        for name, s in summary:
            print(f"{name:20s} {s['step'][0]:7.2f} /{s['step'][1]:6.2f} {s['inference'][0]:7.2f} /{s['inference'][1]:6.2f} "
                  f"{s['lkf'][0]:7.2f} {s['latency'][0]:7.2f} /{s['latency'][1]:6.2f} "
                  f"{s['hz']:10.1f} {s['late']:11} {s['gap_max']:15.2f} {s['gaps10']:11}")
finally:
    for child in children:
        child.terminate()
        child.wait(timeout=5)
