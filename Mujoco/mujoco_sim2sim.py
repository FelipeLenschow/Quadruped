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

# Add root to sys.path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Controller.policy_runner import PolicyRunner
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
ROBOT_DIRS = {"go2": "unitree_go2"}

def ensure_mjcf(robot: str = "go2") -> Path:
    robot_dir = ASSETS_DIR / "unitree_go2"
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

def resolve_joint_order(model) -> dict:
    # Type-Grouped: [all hips, all thighs, all calves]
    isaac_names = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    ]
    
    mapping = {
        "qpos_addr": np.zeros(12, dtype=np.int32),
        "qvel_addr": np.zeros(12, dtype=np.int32),
        "ctrl_idx": np.zeros(12, dtype=np.int32),
        "names": isaac_names
    }

    for i, name in enumerate(isaac_names):
        # 1. Joint Address
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j_id == -1: j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name.replace("_joint", ""))
        if j_id == -1: raise RuntimeError(f"Joint {name} not found in MJCF")
        mapping["qpos_addr"][i] = model.jnt_qposadr[j_id]
        mapping["qvel_addr"][i] = model.jnt_dofadr[j_id]
        
        # 2. Actuator Index
        act_name = name.replace("_joint", "")
        a_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        if a_id == -1: a_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        mapping["ctrl_idx"][i] = a_id

    return mapping

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to best_agent.pt")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--vx", type=float, default=0.0)
    args = parser.parse_args()

    robot = "go2"
    scene_xml = ensure_mjcf(robot)
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    model.opt.timestep = SIM_DT

    mapping = resolve_joint_order(model)

    # Default pose (Type-Grouped: Hips, Thighs, Calves)
    desired_qpos_isaac = np.array([
        0.1, -0.1, 0.1, -0.1,  # Hips
        0.8, 0.8, 1.0, 1.0,    # Thighs
        -1.5, -1.5, -1.5, -1.5 # Calves
    ], dtype=np.float32)
    desired_qpos_mj = np.zeros(model.nq, dtype=np.float64)
    # Set base pose
    desired_qpos_mj[2] = 0.50
    desired_qpos_mj[3:7] = [1.0, 0.0, 0.0, 0.0]
    # Set joint poses
    for i, addr in enumerate(mapping["qpos_addr"]):
        desired_qpos_mj[addr] = desired_qpos_isaac[i]

    # Actuator config
    for i in range(model.nu):
        model.actuator_gainprm[i, 0] = 1.0
        model.actuator_biasprm[i, 1] = 0.0
        model.actuator_ctrllimited[i] = 0
        model.actuator_forcerange[i, :2] = [-23.7, 23.7]
    for i in range(model.njnt):
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
            # Disable MuJoCo internal damping to match IsaacLab's DCMotor (damping=0.5)
            # We apply the 0.5 damping explicitly in the PD loop
            model.dof_damping[model.jnt_dofadr[i]] = 0.0
            model.dof_frictionloss[model.jnt_dofadr[i]] = 0.01

    mujoco.mj_resetData(model, data)
    data.qpos[:] = desired_qpos_mj
    mujoco.mj_forward(model, data)

    # Load policy (Go2 uses DCMotor — no ActuatorNet needed)
    runner = PolicyRunner(args.checkpoint)

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
    model.opt.timestep = 0.001
    
    # Identity mapping: MockUDP.Recv() already returns Isaac-ordered motor states
    isaac_identity = np.arange(12)
    
    # Match IsaacLab ground sliding friction (1.0)
    FOOT_GEOMS = {20, 32, 44, 56}
    for i in range(model.ngeom):
        name = model.geom(i).name
        if "floor" in name or "ground" in name:
            model.geom_friction[i, 0] = 1.0  # sliding friction



    pos_err_hist = np.zeros((3, 12), dtype=np.float32)
    vel_hist = np.zeros((3, 12), dtype=np.float32)
    udp = MockUDP(model, data, mapping)
    step_count = 0
    start_wall = time.time()

    def run_step():
        nonlocal last_actions, step_count, pos_err_hist, vel_hist
        state = udp.Recv()
        # MockUDP returns motor states already in Isaac order, so pass identity mapping
        obs = runner.build_obs(state, commands, last_actions, desired_qpos_isaac, isaac_identity)
        actions = runner.get_action(obs)
        targets = actions * ACTION_SCALE + desired_qpos_isaac
        
        low_cmd = LowCmd()
        for i in range(DECIMATION):
            if i % 5 == 0:
                # Motor states from MockUDP are already in Isaac order
                qpos_isaac = np.array([state.motorState[j].q for j in range(12)])
                qvel_isaac = np.array([state.motorState[j].dq for j in range(12)])
                pos_err = qpos_isaac - targets
                pos_err_hist = np.roll(pos_err_hist, 1, 0); pos_err_hist[0] = pos_err
                vel_hist = np.roll(vel_hist, 1, 0); vel_hist[0] = qvel_isaac
                # Aligned with IsaacLab Go2 DCMotor: stiffness=25.0, damping=0.5
                torques = -25.0 * pos_err_hist[0] - 0.5 * vel_hist[0]
                
                # DCMotor velocity-dependent saturation (four-quadrant model)
                effort_limit = 23.5
                saturation_effort = 23.5
                vel_limit = 30.0
                vel_at_limit = vel_limit * (1 + effort_limit / saturation_effort)
                vel_clamped = np.clip(vel_hist[0], -vel_at_limit, vel_at_limit)
                t_top = saturation_effort * (1.0 - vel_clamped / vel_limit)
                t_bot = saturation_effort * (-1.0 - vel_clamped / vel_limit)
                max_eff = np.minimum(t_top, effort_limit)
                min_eff = np.maximum(t_bot, -effort_limit)
                torques = np.clip(torques, min_eff, max_eff)
                
                # Apply torques via Isaac-ordered index → MuJoCo ctrl index
                for j in range(12):
                    low_cmd.motorCmd[j].tau = torques[j]
                udp.Send(low_cmd)
            mujoco.mj_step(model, data)
            if i % 5 == 0 and i < DECIMATION - 1: state = udp.Recv()

        last_actions[:] = actions
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
