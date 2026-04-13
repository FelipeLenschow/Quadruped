#!/usr/bin/env python3
"""
MuJoCo Sim2Sim: uses a standalone PolicyRunner and a Mock Unitree SDK.
Allows the exact same deployment logic to be used in simulation and hardware.
Aligned with Gazebo Sim2Sim for consistent I/O.
"""

import argparse
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
import mujoco
import mujoco.viewer

# Add root to sys.path so we can import Deployment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Deployment.policy_runner import PolicyRunner
# Logger removed - use ROS 2 bag/plotjuggler instead
from unitree_sdk_mock import MockUDP, LowState, LowCmd

# ── Constants ──────────────────────────────────────────────────────────────────

ACTION_DIM = 12
ACTION_SCALE = 0.25
DECIMATION = 20
SIM_DT = 0.001
STEP_DT = SIM_DT * DECIMATION

ISAAC_JOINT_NAMES = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]

# ── Download MJCF ──────────────────────────────────────────────────────────────

ASSETS_DIR = Path(__file__).parent / "mujoco_menagerie"
MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie/archive/refs/heads/main.zip"
ROBOT_DIRS = {"go1": "unitree_go1", "a1": "unitree_a1", "go2": "unitree_go2"}

def ensure_mjcf(robot: str) -> Path:
    robot_dir = ASSETS_DIR / ROBOT_DIRS[robot]
    scene_xml = robot_dir / "scene.xml"
    if scene_xml.exists(): return scene_xml
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = ASSETS_DIR / "menagerie.zip"
    print(f"[Sim2Sim] Downloading MuJoCo Menagerie for '{robot}'…")
    urllib.request.urlretrieve(MENAGERIE_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        prefix = f"mujoco_menagerie-main/{ROBOT_DIRS[robot]}/"
        for member in [m for m in zf.namelist() if m.startswith(prefix)]:
            rel = member[len("mujoco_menagerie-main/") :]
            dest = ASSETS_DIR / rel
            if member.endswith("/"): dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(dest, "wb") as dst: dst.write(src.read())
    zip_path.unlink()
    return scene_xml

# ── Helpers ────────────────────────────────────────────────────────────────────

def resolve_joint_order(model) -> np.ndarray:
    mj_names = []
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE: continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name: mj_names.append(name)
    def _norm(s: str) -> str: return s.replace("_joint", "").lower()
    mj_norm = [_norm(n) for n in mj_names]
    mj_to_isaac = np.zeros(ACTION_DIM, dtype=np.int32)
    for isaac_idx, isaac_name in enumerate(ISAAC_JOINT_NAMES):
        norm = _norm(isaac_name)
        try: mj_idx = mj_norm.index(norm)
        except ValueError:
            matches = [i for i, n in enumerate(mj_norm) if norm in n]
            mj_idx = matches[0] if matches else isaac_idx
        mj_to_isaac[isaac_idx] = mj_idx
    return mj_to_isaac

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to best_agent.pt")
    parser.add_argument("--robot", default="go1", choices=["go1", "a1", "go2"])
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--vx", type=float, default=0.0)
    args = parser.parse_args()

    scene_xml = ensure_mjcf(args.robot)
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    model.opt.timestep = SIM_DT

    mj_to_isaac = resolve_joint_order(model)
    isaac_to_mj = np.zeros(ACTION_DIM, dtype=np.int32)
    for i, mi in enumerate(mj_to_isaac): isaac_to_mj[i] = mi

    # Default pose (standard quadruped home)
    desired_qpos_isaac = np.array([0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5], dtype=np.float32)
    desired_qpos_mj = np.zeros(ACTION_DIM, dtype=np.float64)
    for i, mi in enumerate(mj_to_isaac): desired_qpos_mj[mi] = desired_qpos_isaac[i]

    # Actuator config
    for i in range(model.nu):
        model.actuator_gainprm[i, 0] = 1.0
        model.actuator_biasprm[i, 1] = 0.0
        model.actuator_ctrllimited[i] = 0
        model.actuator_forcerange[i, :2] = [-23.7, 23.7]
    for i in range(model.njnt):
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
            model.dof_damping[model.jnt_dofadr[i]] = 0.15
            model.dof_frictionloss[model.jnt_dofadr[i]] = 0.05

    mujoco.mj_resetData(model, data)
    data.qpos[7 : 7 + ACTION_DIM] = desired_qpos_mj
    data.qpos[2] = 0.50
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    # Load policy and ActuatorNet
    runner = PolicyRunner(args.checkpoint)
    
    # Try local first, then shared Deployment folder
    act_net_path = Path(__file__).parent / f"unitree_{args.robot}.pt"
    if not act_net_path.exists():
        act_net_path = Path(__file__).parent.parent / "Deployment" / "unitree_quadruped.pt"
    
    print(f"[Sim2Sim] Loading ActuatorNet: {act_net_path}")
    act_net = torch.jit.load(str(act_net_path), map_location="cpu").eval()

    # Shared state for teleop
    import threading
    cmd_vx, cmd_vy, cmd_wz, speed_mult = args.vx, 0.0, 0.0, 1.0
    commands = np.array([cmd_vx, 0.0, 0.0, 0.0], dtype=np.float32)
    _stop = threading.Event()
    show_debug = False

    def _keyboard_thread():
        if not sys.stdin.isatty(): return
        import tty, termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while not _stop.is_set():
                ch = sys.stdin.read(1).upper()
                upd = True
                nonlocal cmd_vx, cmd_vy, cmd_wz, speed_mult, show_debug
                if ch == 'W': cmd_vx = min(1.0, cmd_vx + 0.05)
                elif ch == 'S': cmd_vx = max(-1.0, cmd_vx - 0.05)
                elif ch == 'A': cmd_vy = min(1.0, cmd_vy + 0.05)
                elif ch == 'D': cmd_vy = max(-1.0, cmd_vy - 0.05)
                elif ch == 'Q': cmd_wz = min(1.0, cmd_wz + 0.05)
                elif ch == 'E': cmd_wz = max(-1.0, cmd_wz - 0.05)
                elif ch == 'R': cmd_vx = cmd_vy = cmd_wz = 0.0
                elif ch == 'O': show_debug = not show_debug; print(f"\n[DEBUG] Show obs: {show_debug}")
                elif ch == '=': speed_mult = min(3.0, speed_mult + 0.1)
                elif ch == '-': speed_mult = max(0.1, speed_mult - 0.1)
                elif ch == '\x03': _stop.set(); break
                else: upd = False
                if upd:
                    commands[:] = [cmd_vx*speed_mult, cmd_vy*speed_mult, cmd_wz*speed_mult, 0.0]
                    print(f"\r[Cmd] vx={commands[0]:+.2f} vy={commands[1]:+.2f} wz={commands[2]:+.2f} speed={speed_mult:.1f}x  ", end="", flush=True)
        finally: termios.tcsetattr(fd, termios.TCSADRAIN, old)

    threading.Thread(target=_keyboard_thread, daemon=True).start()

    # Sim Loop
    last_actions = np.zeros(ACTION_DIM, dtype=np.float32)
    pos_err_hist = np.zeros((3, 12), dtype=np.float32)
    vel_hist = np.zeros((3, 12), dtype=np.float32)
    udp = MockUDP(model, data, mj_to_isaac, isaac_to_mj)
    step_count = 0
    start_wall = time.time()

    def run_step():
        nonlocal last_actions, step_count, pos_err_hist, vel_hist
        state = udp.Recv()
        obs = runner.build_obs(state, commands, last_actions, desired_qpos_isaac, mj_to_isaac)
        actions = runner.get_action(obs)
        targets = actions * ACTION_SCALE + desired_qpos_isaac
        
        low_cmd = LowCmd()
        for i in range(DECIMATION):
            if i % 5 == 0:
                mj_qpos = np.array([state.motorState[j].q for j in range(12)])
                mj_qvel = np.array([state.motorState[j].dq for j in range(12)])
                pos_err = mj_qpos[mj_to_isaac] - targets
                pos_err_hist = np.roll(pos_err_hist, 1, 0); pos_err_hist[0] = pos_err
                vel_hist = np.roll(vel_hist, 1, 0); vel_hist[0] = mj_qvel[mj_to_isaac]
                net_in = torch.zeros((12, 6))
                net_in[:, :3] = torch.from_numpy(pos_err_hist.T)
                net_in[:, 3:] = torch.from_numpy(vel_hist.T)
                with torch.no_grad(): torques = act_net(net_in).squeeze().numpy()
                for j, mj_idx in enumerate(isaac_to_mj): low_cmd.motorCmd[mj_idx].tau = np.clip(torques[j], -23.7, 23.7)
                udp.Send(low_cmd)
            mujoco.mj_step(model, data)
            if i % 5 == 0 and i < DECIMATION - 1: state = udp.Recv()

        last_actions[:] = actions
        ros_logger.log(state, commands, last_actions, ISAAC_JOINT_NAMES)
        step_count += 1
        if step_count % 50 == 0:
            print(f"\r[Step {step_count:6d}] h={data.qpos[2]:.3f} vx={state.base_lin_vel[0]:+.2f}  ", end="", flush=True)

    try:
        if args.no_render:
            while not _stop.is_set():
                run_step()
                if args.duration > 0 and step_count * STEP_DT >= args.duration: break
                time.sleep(max(0, (start_wall + step_count*STEP_DT) - time.time()))
        else:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
                while viewer.is_running() and not _stop.is_set():
                    run_step()
                    viewer.sync()
                    time.sleep(max(0, (start_wall + step_count*STEP_DT) - time.time()))
    except KeyboardInterrupt: pass
    finally:
        _stop.set()
        print(f"\n[Sim2Sim] Done: {step_count} steps.")

if __name__ == "__main__":
    main()
