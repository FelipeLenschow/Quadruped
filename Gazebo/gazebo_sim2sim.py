#!/usr/bin/env python3
"""
Gazebo Sim2Sim: Clean, direct PD torque control.
MuJoCo parity implementation for Gazebo Harmonic.
Uses the same DCMotor model (kp=25, kd=0.5) as IsaacLab training.
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
from Controller.policy_runner import quat_to_rot_matrix

# ── constants ──────────────────────────────────────────────────────────────────
ACTION_SCALE = 0.25
DECIMATION = 20
SIM_DT = 0.001
STEP_DT = SIM_DT * DECIMATION
KP = 25.0
KD = 0.5
EFFORT_LIMIT = 23.5
SATURATION_EFFORT = 23.5
VEL_LIMIT = 30.0
JOINT_NAMES = [
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
]
HAA_SIGN = np.array([1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)

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
    def __init__(self, robot_name="go2", world_name="quadruped_world"):
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
                self.state.motorState[idx].q = joint.axis1.position * HAA_SIGN[idx]
                self.state.motorState[idx].dq = joint.axis1.velocity * HAA_SIGN[idx]

    def _odom_cb(self, msg):
        self.state.base_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.state.imu.quaternion = np.array(
            [msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]
        )
        # Gazebo OdometryPublisher reports twist in BODY frame (child_frame)
        self.state.base_lin_vel[:] = [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z]
        self.state.imu.gyroscope = [msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z]

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
    parser.add_argument("--robot", default="go2")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--vx", type=float, default=0.0)
    args = parser.parse_args()

    # Sanitization: Partitioning for Gazebo
    os.environ["GZ_PARTITION"] = "quadruped_sim"

    # Load Policy (Go2 uses DCMotor PD control — no ActuatorNet needed)
    runner = PolicyRunner(args.checkpoint)

    # Launch Gazebo
    root = os.path.dirname(os.path.abspath(__file__))
    world_path = os.path.join(root, "scene.sdf")

    # Make sure resources are found
    os.environ["GZ_SIM_RESOURCE_PATH"] = (
        root
        + os.pathsep
        + os.path.join(root, "models")
        + os.pathsep
        + os.path.abspath(os.path.join(root, "..", "Unitree_Go2", "models"))
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
    desired_qpos = np.array([0.1, 0.1, 0.1, 0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5], dtype=np.float32)
    isaac_identity = np.arange(12)

    print("[Gazebo] Sim2Sim Loop Start (50Hz Policy, PD Torque Control)")

    count = 0
    step_count = 0
    start_wall = time.time()

    try:
        while not _stop.is_set():
            state = bridge.state
            loop_sim_start = bridge.sim_time

            # 1. Policy Step (50Hz = every DECIMATION physics steps)
            obs = runner.build_obs(
                state, commands, last_actions, desired_qpos, isaac_identity
            )
            actions = runner.get_action(obs)
            targets = actions * ACTION_SCALE + desired_qpos

            # 2. PD Torque Sub-loop (match MuJoCo DECIMATION=20 at 1kHz physics)
            for sub_idx in range(DECIMATION):
                next_step_time = loop_sim_start + (sub_idx + 1) * SIM_DT
                while bridge.sim_time < next_step_time and not _stop.is_set():
                    time.sleep(0.0001)

                # Reread joint state for sub-stepping accuracy
                if sub_idx % 5 == 0:
                    cur_q = np.array([state.motorState[j].q for j in range(12)])
                    cur_dq = np.array([state.motorState[j].dq for j in range(12)])

                    # PD control matching IsaacLab DCMotor (kp=25, kd=0.5)
                    pos_err = cur_q - targets
                    torques = -KP * pos_err - KD * cur_dq

                    # DCMotor velocity-dependent saturation (four-quadrant model)
                    vel_at_limit = VEL_LIMIT * (1 + EFFORT_LIMIT / SATURATION_EFFORT)
                    vel_clamped = np.clip(cur_dq, -vel_at_limit, vel_at_limit)
                    t_top = SATURATION_EFFORT * (1.0 - vel_clamped / VEL_LIMIT)
                    t_bot = SATURATION_EFFORT * (-1.0 - vel_clamped / VEL_LIMIT)
                    max_eff = np.minimum(t_top, EFFORT_LIMIT)
                    min_eff = np.maximum(t_bot, -EFFORT_LIMIT)
                    torques = np.clip(torques, min_eff, max_eff)

                    # Publish torques to Gazebo (flip signs back for right-side HAA)
                    for j_idx, torque in enumerate(torques):
                        msg = double_pb2.Double()
                        msg.data = float(torque * HAA_SIGN[j_idx])
                        bridge.joint_pubs[j_idx].publish(msg)

            last_actions[:] = actions
            step_count += 1
            if step_count % 50 == 0:
                print(
                    f"\r[Step {step_count:6d}] t={bridge.sim_time:.2f} h={state.base_pos[2]:.3f} vx={state.base_lin_vel[0]:+.2f}  ",
                    end="",
                    flush=True,
                )

            # Wall-time padding if sim is too fast
            elapsed = time.time() - start_wall
            expected = step_count * STEP_DT
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
