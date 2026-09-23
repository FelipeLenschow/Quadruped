"""Profile Unitree/real_driver.py's control loop without moving the robot.

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
  --mode tight   control_loop back-to-back: pure compute cost of each stage, no scheduling.
  --mode timer   the driver's own timers under rclpy.spin, exactly as deployed: reports the
                 policy-step period (target 20 ms) and the LowCmd write interval (target 5 ms).
Options:
  --lowstate_hz N    also stream LowState_ over SDK DDS into a ChannelSubscriber, like the
                     robot's 500 Hz stream, to include its deserialisation cost.
  --torch_threads N  run the policy with N intra-op threads (sets QUADRUPED_TORCH_THREADS);
                     0 keeps PolicyRunner's default of 1. Use the core count (6 on the Go2's
                     Orin Nano) to reproduce the old multithreaded behaviour.

Examples (inside the robot's Docker image, from the repo root):
  python3 Tools/profile_control_loop.py --mode tight
  python3 Tools/profile_control_loop.py --mode timer --seconds 20 --lowstate_hz 500
  python3 Tools/profile_control_loop.py --mode timer --lowstate_hz 500 --torch_threads 6
"""
import argparse
import collections
import os
import sys
import threading
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "Unitree"))
sys.path.insert(0, os.path.join(REPO, "Unitree", "unitree_sdk2_python"))
os.chdir(REPO)  # real_driver and the pipeline read Configs/ relative to the repo root

DEFAULT_CKPT = os.path.join(
    "IsaacLab_Tasks", "unitree_rl_lab", "logs", "rsl_rl",
    "unitree_go2_velocity_sigma_vel_foot_rough_deploy", "Noises", "model_2999.pt",
)

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--mode", default="tight", choices=["tight", "timer"])
ap.add_argument("--pipeline_mode", default="policy", choices=["pose", "policy"])
ap.add_argument("--seconds", type=float, default=15.0, help="duration of --mode timer")
ap.add_argument("--steps", type=int, default=1000, help="policy steps in --mode tight")
ap.add_argument("--lowstate_hz", type=float, default=0.0)
ap.add_argument("--torch_threads", type=int, default=0)
ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="policy checkpoint (any format PolicyRunner loads)")
args = ap.parse_args()

if args.torch_threads > 0:
    os.environ["QUADRUPED_TORCH_THREADS"] = str(args.torch_threads)

os.environ.pop("CYCLONEDDS_URI", None)
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

ChannelFactoryInitialize(0, "lo")

os.environ["ROS_DOMAIN_ID"] = "77"
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

rclpy.init()

import real_driver as rd
from pipeline import LocomotionPipeline
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
        self.robot_type = "go2"
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd_profile_only", LowCmd_)
        self.lowcmd_publisher.Init()
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


drv = ProfDriver()
if "main" not in drv.pipeline.policy_manager.policies:
    sys.exit(f"Policy failed to load from {args.ckpt} -- see the PolicyManager error above.")

# Synthetic standing LowState_, SDK motor order (FR, FL, RR, RL x hip, thigh, calf).
low_state = unitree_go_msg_dds__LowState_()
STANCE_SDK = [-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5]
for i, q in enumerate(STANCE_SDK):
    low_state.motor_state[i].q = q
low_state.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
low_state.imu_state.accelerometer = [0.0, 0.0, 9.81]
low_state.foot_force = [30, 30, 30, 30]  # above the gate on all four feet: 4 LKF updates/step
drv.low_state = low_state
rng = np.random.default_rng(0)

# Fake console: heartbeat always fresh, 55 % torque, chosen pipeline mode.
sp = drv.pipeline.safety_processor
sp.has_received_heartbeat = True
sp.max_torque_percent = 55.0
sp._recompute_safety_limits()
drv.pipeline.mode = args.pipeline_mode
drv.pipeline.mode_transition_active = False

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
_write = drv.lowcmd_publisher.Write


def write_stamped(*a, **k):
    write_stamps.append(time.perf_counter())
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
    t = time.perf_counter()
    _control(drv)
    T[("TOTAL", "control")].append(time.perf_counter() - t)


def lowcmd_loop():
    _loop[0] = "lowcmd"
    t = time.perf_counter()
    _lowcmd(drv)
    T[("TOTAL", "lowcmd")].append(time.perf_counter() - t)


drv.control_loop, drv.lowcmd_loop = control_loop, lowcmd_loop

# Optional LowState stream, like the robot's 500 Hz rt/lowstate, into an SDK subscriber.
stop = threading.Event()
received = [0]
if args.lowstate_hz > 0:
    ls_pub = ChannelPublisher("rt/lowstate_profile_only", LowState_)
    ls_pub.Init()
    ls_sub = ChannelSubscriber("rt/lowstate_profile_only", LowState_)
    ls_sub.Init(lambda msg: received.__setitem__(0, received[0] + 1), 10)

    def stream():
        period, nxt = 1.0 / args.lowstate_hz, time.perf_counter()
        while not stop.is_set():
            ls_pub.Write(low_state)
            nxt += period
            time.sleep(max(0.0, nxt - time.perf_counter()))

    # In-process, so the publisher's serialisation also costs this process GIL time -- a
    # little pessimistic next to the robot, where the firmware publishes.
    threading.Thread(target=stream, daemon=True).start()

print(f"\ntorch intra-op threads: {torch.get_num_threads()} | mode={args.mode} "
      f"pipeline={args.pipeline_mode} lowstate_hz={args.lowstate_hz:g} | usable cpus={len(os.sched_getaffinity(0))}")

# --------------------------------------------------------------------------- run
if args.mode == "tight":
    for _ in range(50):
        drv.control_loop()
    T.clear()
    policy_stamps.clear()
    write_stamps.clear()
    for _ in range(args.steps):
        drv.control_loop()
        for _ in range(round(rd.POLICY_DT / rd.LOWCMD_DT) - 1):
            drv.lowcmd_loop()
else:
    drv.create_timer(rd.POLICY_DT, drv.control_loop)
    drv.create_timer(rd.LOWCMD_DT, drv.lowcmd_loop)
    t_end = time.time() + args.seconds
    while time.time() < t_end:
        rclpy.spin_once(drv, timeout_sec=0.1)
stop.set()

# --------------------------------------------------------------------------- report
print(f"\n{'stage':28s} {'loop':8s} {'n':>6s} {'mean ms':>8s} {'p50':>7s} {'p95':>7s} {'p99':>7s} {'max':>7s}")
ORDER = ["TOTAL", "raw sensor read", "process_state (incl. LKF)", "  LKF update", "policy / pose step",
         "  build_obs", "  inference", "safety.process", "distributor.send", "telemetry.publish",
         "send_to_sdk", "  CRC pack + crc", "  DDS write"]
for stage in ORDER:
    for loop in ("control", "lowcmd"):
        x = T.get((stage, loop))
        if not x:
            continue
        x = 1000 * np.asarray(x)
        print(f"{stage:28s} {loop:8s} {len(x):6d} {x.mean():8.3f} {np.median(x):7.3f} "
              f"{np.percentile(x, 95):7.3f} {np.percentile(x, 99):7.3f} {x.max():7.2f}")

if args.mode == "timer" and len(policy_stamps) > 10:
    d = 1000 * np.diff(policy_stamps)
    print(f"\npolicy step period : median {np.median(d):.2f} ms, mean {d.mean():.2f} ms "
          f"-> {1000 / d.mean():.1f} Hz (target {1000 * rd.POLICY_DT:.0f} ms)")
    print(f"  late steps       : >22 ms {np.sum(d > 22)}, >25 ms {np.sum(d > 25)}, max {d.max():.1f} ms "
          f"of {len(d)}")
    w = 1000 * np.diff(write_stamps)
    print(f"LowCmd write gap   : median {np.median(w):.2f} ms, p99 {np.percentile(w, 99):.2f} ms, "
          f"max {w.max():.2f} ms -> {len(write_stamps) / args.seconds:.0f} writes/s "
          f"(Unitree asks for a write every 1-10 ms)")
    print(f"  gaps > 10 ms     : {np.sum(w > 10)} of {len(w)}")
if args.lowstate_hz > 0:
    print(f"LowState samples delivered: {received[0]}")
