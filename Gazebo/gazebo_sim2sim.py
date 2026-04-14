#!/usr/bin/env python3
"""
Gazebo Sim2Sim: Clean, direct policy-to-ActuatorNet control.
MuJoCo parity implementation for Gazebo Harmonic.
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
    from gz.transport13 import Node as GzTransportNode
    from gz.msgs10 import (
        double_pb2,
        model_pb2,
        imu_pb2,
        pose_pb2,
        world_stats_pb2,
        world_control_pb2,
        odometry_pb2,
    )
except ImportError:
    print("[ERROR] Gazebo (Harmonic) Python bindings not found.")
    raise

# Add root to sys.path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Controller.policy_runner import PolicyRunner
from Mujoco.unitree_sdk_mock import quat_to_rot_matrix

# ── constants ──────────────────────────────────────────────────────────────────
ACTION_SCALE = 0.25
JOINT_NAMES = [
    "lf_haa_joint",
    "rf_haa_joint",
    "lh_haa_joint",
    "rh_haa_joint",
    "lf_hfe_joint",
    "rf_hfe_joint",
    "lh_hfe_joint",
    "rh_hfe_joint",
    "lf_kfe_joint",
    "rf_kfe_joint",
    "lh_kfe_joint",
    "rh_kfe_joint",
]

# ── Gazebo Sim2Sim Bridge ──────────────────────────────────────────────────────


class LowState:
    def __init__(self):
        self.imu = type(
            "IMU",
            (),
            {"quaternion": np.array([1.0, 0, 0, 0]), "gyroscope": np.zeros(3)},
        )()
        self.motorState = [
            type("Motor", (), {"q": 0.0, "dq": 0.0})() for _ in range(12)
        ]
        self.base_lin_vel = np.zeros(3)
        self.base_pos = np.zeros(3)


class GazeboSimBridge:
    def __init__(self, robot_name="go1", world_name="quadruped_world"):
        self.node = GzTransportNode()
        self.robot_name = robot_name
        self.world_name = world_name
        self.state = LowState()
        self.sim_time = 0.0

        # Subscriptions
        self.node.subscribe(
            imu_pb2.IMU, f"/model/{robot_name}/link/base/sensor/imu/imu", self._imu_cb
        )
        self.node.subscribe(
            model_pb2.Model, f"/model/{robot_name}/joint_state", self._joint_cb
        )
        self.node.subscribe(
            odometry_pb2.Odometry, f"/model/{robot_name}/odometry", self._odom_cb
        )
        self.gz_stats_sub = self.node.subscribe(
            world_stats_pb2.WorldStatistics,
            f"/world/{world_name}/stats",
            self._stats_cb,
        )

        # Publishers
        self.joint_pubs = []
        for jname in JOINT_NAMES:
            topic = f"/model/{robot_name}/joint/{jname}/cmd_force"
            self.joint_pubs.append(self.node.advertise(topic, double_pb2.Double))

        self.world_control_pub = self.node.advertise(
            f"/world/{world_name}/control", world_control_pb2.WorldControl
        )

    def _stats_cb(self, msg):
        self.sim_time = msg.sim_time.sec + msg.sim_time.nsec * 1e-9

    def _imu_cb(self, msg):
        self.state.imu.quaternion = np.array(
            [msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z]
        )
        self.state.imu.gyroscope = np.array(
            [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        )

    def _joint_cb(self, msg):
        for joint in msg.joint:
            if joint.name in JOINT_NAMES:
                idx = JOINT_NAMES.index(joint.name)
                self.state.motorState[idx].q = joint.axis1.position
                self.state.motorState[idx].dq = joint.axis1.velocity

    def _odom_cb(self, msg):
        self.state.base_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.state.imu.quaternion = np.array(
            [msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]
        )
        # World frame linear velocity
        v_world = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
        # Rotate to body frame
        R = quat_to_rot_matrix(self.state.imu.quaternion)
        self.state.base_lin_vel[:] = R.T @ v_world

    def reset_world(self):
        msg = world_control_pb2.WorldControl()
        msg.reset.all = True
        for _ in range(3):
            self.world_control_pub.publish(msg)
            time.sleep(0.05)
        print("\n[Gazebo] Reset command sent.")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to policy checkpoint")
    parser.add_argument("--robot", default="go1")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--vx", type=float, default=0.0)
    args = parser.parse_args()

    # Sanitization: Partitioning for Gazebo
    os.environ["GZ_PARTITION"] = "quadruped_sim"

    # Load Models
    runner = PolicyRunner(args.checkpoint)
    act_net_path = Path(__file__).parent.parent / "Mujoco" / "unitree_quadruped.pt"
    print(f"[Gazebo] Loading ActuatorNet: {act_net_path}")
    act_net = torch.jit.load(str(act_net_path), map_location="cpu").eval()

    # Launch Gazebo
    root = os.path.dirname(os.path.abspath(__file__))
    world_path = os.path.join(root, "scene.sdf")

    # Make sure resources are found
    os.environ["GZ_SIM_RESOURCE_PATH"] = (
        root
        + os.pathsep
        + os.path.join(root, "models")
        + os.pathsep
        + os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    )

    # Optimization for remote/VDI: enable software rendering if requested
    if os.environ.get("FORCE_SOFTWARE_RENDER", "0") == "1":
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
        print("[Gazebo] Forcing software rendering (LIBGL_ALWAYS_SOFTWARE=1)")

    gz_args = ["gz", "sim", "-r", world_path]
    if args.no_render:
        gz_args.append("-s")
    print(f"[Gazebo] Launching: {' '.join(gz_args)}")
    proc = subprocess.Popen(gz_args)

    bridge = GazeboSimBridge(args.robot)
    _stop = threading.Event()

    # Teleop state
    cmd_vx, cmd_vy, cmd_wz = args.vx, 0.0, 0.0
    commands = np.array([cmd_vx, 0.0, 0.0, 0.0], dtype=np.float32)

    def _keyboard_thread():
        if not sys.stdin.isatty():
            return
        import tty, termios

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while not _stop.is_set():
                ch = sys.stdin.read(1).upper()
                nonlocal cmd_vx, cmd_vy, cmd_wz
                if ch == "W":
                    cmd_vx = min(1.0, cmd_vx + 0.05)
                elif ch == "S":
                    cmd_vx = max(-1.0, cmd_vx - 0.05)
                elif ch == "A":
                    cmd_vy = min(1.0, cmd_vy + 0.05)
                elif ch == "D":
                    cmd_vy = max(-1.0, cmd_vy - 0.05)
                elif ch == "Q":
                    cmd_wz = min(1.0, cmd_wz + 0.05)
                elif ch == "E":
                    cmd_wz = max(-1.0, cmd_wz - 0.05)
                elif ch == "R":
                    cmd_vx = cmd_vy = cmd_wz = 0.0
                elif ch == "\x03":
                    _stop.set()
                    break
                commands[:] = [cmd_vx, cmd_vy, cmd_wz, 0.0]
                print(
                    f"\r[Cmd] vx={commands[0]:+.2f} vy={commands[1]:+.2f} wz={commands[2]:+.2f}  ",
                    end="",
                    flush=True,
                )
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    threading.Thread(target=_keyboard_thread, daemon=True).start()

    # Control Buffers
    # Wait for initial state
    print("[Gazebo] Waiting for initial state...")
    while bridge.sim_time == 0 and not _stop.is_set():
        time.sleep(0.1)
    
    last_actions = np.zeros(12, dtype=np.float32)
    desired_qpos = np.array([0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5], dtype=np.float32)
    
    pos_err_hist = np.zeros((3, 12), dtype=np.float32)
    vel_hist = np.zeros((3, 12), dtype=np.float32)
    # Initialize history with current state to avoid impulsive start
    pos_err_hist[:] = np.array([bridge.state.motorState[j].q for j in range(12)]) - desired_qpos
    vel_hist[:] = np.array([bridge.state.motorState[j].dq for j in range(12)])

    print("[Gazebo] Sim2Sim Loop Start (50Hz Policy, 200Hz Actuator)")

    _last_pose = np.zeros(3)
    _last_pose_time = 0.0
    count = 0
    start_wall = time.time()

    try:
        while not _stop.is_set():
            state = bridge.state
            loop_sim_start = bridge.sim_time

            # 1. Velocity and IMU state are updated in callbacks
            pass

            # 2. Policy Step (50Hz)
            obs = runner.build_obs(
                state, commands, last_actions, desired_qpos, np.arange(12)
            )
            actions = runner.get_action(obs)
            targets = actions * ACTION_SCALE + desired_qpos

            # 3. Actuator Sub-loop (200Hz - 4 steps per policy step)
            for sub_idx in range(4):
                next_act_time = loop_sim_start + (sub_idx + 1) * 0.005
                while bridge.sim_time < next_act_time and not _stop.is_set():
                    time.sleep(0.0001)

                cur_q = np.array([state.motorState[j].q for j in range(12)])
                cur_dq = np.array([state.motorState[j].dq for j in range(12)])

                pos_err_hist = np.roll(pos_err_hist, 1, 0)
                pos_err_hist[0] = cur_q - targets
                vel_hist = np.roll(vel_hist, 1, 0)
                vel_hist[0] = cur_dq

                net_in = torch.zeros((12, 6))
                net_in[:, :3] = torch.from_numpy(pos_err_hist.T)
                net_in[:, 3:] = torch.from_numpy(vel_hist.T)

                with torch.no_grad():
                    torques = act_net(net_in).numpy().flatten()

                torques = np.clip(torques, -23.7, 23.7)
                for j_idx, torque in enumerate(torques):
                    msg = double_pb2.Double()
                    msg.data = float(torque)
                    bridge.joint_pubs[j_idx].publish(msg)

            last_actions[:] = actions
            count += 1
            if count % 50 == 0:
                print(
                    f"\r[Step {count:6d}] t={bridge.sim_time:.2f} pos=[{state.base_pos[0]:.2f}, {state.base_pos[1]:.2f}, {state.base_pos[2]:.2f}] vx={state.base_lin_vel[0]:+.2f}  ",
                    end="",
                    flush=True,
                )

            # Wall-time padding if sim is too fast
            elapsed = time.time() - start_wall
            expected = count * 0.020
            if expected > elapsed:
                time.sleep(expected - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        _stop.set()
        proc.terminate()
        print("\n[Gazebo] Done.")


if __name__ == "__main__":
    main()
