#!/usr/bin/env python3
"""
Gazebo Sim Sim2Sim: uses a standalone PolicyRunner and Gazebo Sim (Harmonic).
Enables the third 'party member' simulations.
Aligned with MuJoCo Sim2Sim for consistent I/O.
"""

import argparse
import os
import sys
import time
import threading
import subprocess
from pathlib import Path
import numpy as np
import torch

# Gazebo Transport & Msgs
try:
    from gz.transport13 import Node
    from gz.msgs10 import double_pb2, model_pb2, imu_pb2, odometry_pb2
except ImportError:
    print("[ERROR] Gazebo (Harmonic) Python bindings not found.")
    raise

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Deployment.policy_runner import PolicyRunner

# ── Environment Setup ──────────────────────────────────────────────────────────
os.environ["GZ_PARTITION"] = "quadruped_party"
os.environ["IGN_PARTITION"] = "quadruped_party"
root_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["GZ_SIM_RESOURCE_PATH"] = (
    root_dir
    + os.pathsep
    + os.path.join(root_dir, "models")
    + os.pathsep
    + os.environ.get("GZ_SIM_RESOURCE_PATH", "")
)

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True, help="Path to policy checkpoint")
parser.add_argument("--robot", default="go1", choices=["go1", "a1", "go2"])
parser.add_argument("--no-render", action="store_true")
parser.add_argument("--vx", type=float, default=0.0)
args = parser.parse_args()

# ── Constants ──────────────────────────────────────────────────────────────────
ACTION_SCALE = 0.25
DECIMATION = 20 # 50Hz Policy if physics is 1000Hz (but we use wall clock)

# Standard Isaac Lab Grouping: all HAA, then all HFE, then all KFE
# Leg Order: FL, FR, RL, RR (Matches MuJoCo)
JOINT_NAMES = [
    "lf_haa_joint", "rf_haa_joint", "lh_haa_joint", "rh_haa_joint",
    "lf_hfe_joint", "rf_hfe_joint", "lh_hfe_joint", "rh_hfe_joint",
    "lf_kfe_joint", "rf_kfe_joint", "lh_kfe_joint", "rh_kfe_joint",
]

# ── Standard Abstractions ──────────────────────────────────────────────────────

class LowState:
    def __init__(self):
        self.imu = type("IMU", (), {"quaternion": np.array([1.0, 0, 0, 0]), "gyroscope": np.zeros(3)})()
        self.motorState = [type("Motor", (), {"q": 0.0, "dq": 0.0})() for _ in range(12)]
        self.base_lin_vel = np.zeros(3)
        self.base_pos = np.zeros(3)

class LowCmd:
    def __init__(self):
        self.motorCmd = [type("MotorCmd", (), {"tau": 0.0, "q": 0.0})() for _ in range(12)]

class GazeboBridge:
    def __init__(self, robot_name="go1"):
        self.node = Node()
        self.robot_name = robot_name
        self.state = LowState()

        # Subscriptions
        self.node.subscribe(imu_pb2.IMU, f"/model/{robot_name}/link/base/sensor/imu/imu", self._imu_cb)
        self.node.subscribe(model_pb2.Model, f"/model/{robot_name}/joint_state", self._joint_cb)
        self.node.subscribe(odometry_pb2.Odometry, f"/model/{robot_name}/odometry", self._odom_cb)

        # Publishers
        self.joint_pubs = []
        for jname in JOINT_NAMES:
            topic = f"/model/{robot_name}/joint/{jname}/cmd_pos"
            self.joint_pubs.append(self.node.advertise(topic, double_pb2.Double))

    def _imu_cb(self, msg):
        self.state.imu.quaternion = np.array([msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z])
        self.state.imu.gyroscope = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])

    def _joint_cb(self, msg):
        for joint in msg.joint:
            if joint.name in JOINT_NAMES:
                idx = JOINT_NAMES.index(joint.name)
                self.state.motorState[idx].q = joint.axis1.position
                self.state.motorState[idx].dq = joint.axis1.velocity

    def _odom_cb(self, msg):
        self.state.base_lin_vel = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
        self.state.base_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

    def send_commands(self, targets):
        """Sends position targets to Gazebo (SDF is currently set to PositionController)."""
        for i, target in enumerate(targets):
            msg = double_pb2.Double()
            msg.data = float(target)
            self.joint_pubs[i].publish(msg)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    world_path = os.path.join(root, "scene.sdf")

    # Start Gazebo
    gz_args = ["gz", "sim", world_path]
    if args.no_render: gz_args.append("-s")
    print(f"[Gazebo] Launching: {' '.join(gz_args)}")
    proc = subprocess.Popen(gz_args)
    time.sleep(4)

    # 1. Initialize Policy & ActuatorNet
    runner = PolicyRunner(args.checkpoint)
    
    # Load standardized ActuatorNet
    act_net_path = Path(__file__).parent.parent / "Deployment" / "unitree_quadruped.pt"
    if not act_net_path.exists():
        act_net_path = Path(__file__).parent / "unitree_quadruped.pt"
    print(f"[Gazebo] Loading ActuatorNet: {act_net_path}")
    act_net = torch.jit.load(str(act_net_path), map_location="cpu").eval()

    bridge = GazeboBridge(args.robot)

    # 2. Teleop State
    cmd_vx, cmd_vy, cmd_wz = args.vx, 0.0, 0.0
    commands = np.array([cmd_vx, cmd_vy, cmd_wz, 0.0], dtype=np.float32)
    _stop = threading.Event()

    def _keyboard_thread():
        if not sys.stdin.isatty(): return
        import tty, termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        speed_mult = 1.0
        try:
            tty.setraw(fd)
            nonlocal cmd_vx, cmd_vy, cmd_wz
            while not _stop.is_set():
                ch = sys.stdin.read(1).upper()
                upd = True
                if ch == "W": cmd_vx = min(1.0, cmd_vx + 0.1)
                elif ch == "S": cmd_vx = max(-1.0, cmd_vx - 0.1)
                elif ch == "A": cmd_vy = min(1.0, cmd_vy + 0.1)
                elif ch == "D": cmd_vy = max(-1.0, cmd_vy - 0.1)
                elif ch == "Q": cmd_wz = min(1.0, cmd_wz + 0.1)
                elif ch == "E": cmd_wz = max(-1.0, cmd_wz - 0.1)
                elif ch == "R": cmd_vx = cmd_vy = cmd_wz = 0.0
                elif ch == "=": speed_mult = min(3.0, speed_mult + 0.1)
                elif ch == "-": speed_mult = max(0.1, speed_mult - 0.1)
                elif ch == "\x03": _stop.set(); break
                else: upd = False
                if upd:
                    commands[:] = [cmd_vx*speed_mult, cmd_vy*speed_mult, cmd_wz*speed_mult, 0.0]
                    print(f"\r[Cmd] vx={commands[0]:+.2f} vy={commands[1]:+.2f} wz={commands[2]:+.2f}  ", end="", flush=True)
        finally: termios.tcsetattr(fd, termios.TCSADRAIN, old)

    threading.Thread(target=_keyboard_thread, daemon=True).start()

    # 3. Control Loop
    last_actions = np.zeros(12, dtype=np.float32)
    pos_err_hist = np.zeros((3, 12), dtype=np.float32)
    vel_hist = np.zeros((3, 12), dtype=np.float32)
    
    desired_qpos = np.array([
        0.1, -0.1, 0.1, -0.1,   # Hips (FL, FR, RL, RR)
        0.8, 0.8, 1.0, 1.0,     # Thighs
        -1.5, -1.5, -1.5, -1.5  # Calves
    ], dtype=np.float32)

    print("[Gazebo] Sim2Sim running (50Hz Policy, 200Hz Actuator). Press Ctrl+C to stop.")
    
    count = 0
    start_time = time.time()
    try:
        while not _stop.is_set():
            loop_start = time.time()
            
            # A. Policy Loop (50Hz)
            state = bridge.state
            obs = runner.build_obs(state, commands, last_actions, desired_qpos, np.arange(12))
            actions = runner.get_action(obs)
            targets = actions * ACTION_SCALE + desired_qpos
            
            # B. Actuator/Sub-loop (Synchronous 4 sub-steps to reach 200Hz)
            for _ in range(4):
                state = bridge.state # Update state reference
                
                # Calculate Net Input (Same as MuJoCo/Real)
                mj_qpos = np.array([state.motorState[j].q for j in range(12)])
                mj_qvel = np.array([state.motorState[j].dq for j in range(12)])
                
                pos_err = mj_qpos - targets
                pos_err_hist = np.roll(pos_err_hist, 1, 0); pos_err_hist[0] = pos_err
                vel_hist = np.roll(vel_hist, 1, 0); vel_hist[0] = mj_qvel
                
                # NOTE: We keep using Position targets as the current SDF plugin expects it.
                # However, the history buffers are now accurately maintained.
                bridge.send_commands(targets)
                time.sleep(0.005) # 200Hz sub-loop

            last_actions[:] = actions
            count += 1
            if count % 50 == 0:
                print(f"\r[Step {count:6d}] h={state.base_pos[2]:.3f} vx={state.base_lin_vel[0]:+.2f} ObsShape: {obs.shape} ", end="", flush=True)

            # Sync to 50Hz
            elapsed = time.time() - loop_start
            if elapsed < 0.020:
                time.sleep(0.020 - elapsed)

    except KeyboardInterrupt:
        print("\n[Gazebo] Stopping...")
    finally:
        _stop.set()
        proc.terminate()

if __name__ == "__main__":
    main()
