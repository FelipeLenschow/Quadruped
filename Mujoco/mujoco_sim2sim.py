#!/usr/bin/env python3
"""
Cross-simulator Sim2Sim: run an Isaac Lab–trained Quadruped policy inside MuJoCo.

Requirements (NO Isaac Lab needed):
    pip install mujoco torch numpy

Usage:
    python scripts/mujoco_sim2sim.py --checkpoint logs/skrl/quadruped_direct/cp.../checkpoints/best_agent.pt

Controls (type in the TERMINAL while the viewer is open):
    W / S  — increase / decrease forward velocity command
    A / D  — increase / decrease lateral velocity command
    Q / E  — increase / decrease yaw rate command
    R      — reset all velocity commands to zero
    = / -  — speed multiplier up / down
    Ctrl+C — quit
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import zipfile
import time
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── CLI ────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True, help="Path to best_agent.pt")
parser.add_argument(
    "--robot",
    default="go1",
    choices=["go1", "a1", "go2"],
    help="Robot model to load from MuJoCo Menagerie",
)
parser.add_argument(
    "--duration",
    type=float,
    default=0.0,
    help="Run for N seconds then exit (0 = run forever)",
)
parser.add_argument("--no-render", action="store_true", help="Run headless (no viewer)")
parser.add_argument(
    "--vx",
    type=float,
    default=0.0,
    help="Initial forward velocity command (for testing)",
)
args = parser.parse_args()

# ── Constants matching Isaac Lab training ──────────────────────────────────────

OBS_DIM = 49
ACTION_DIM = 12
ACTION_SCALE = 0.25
DECIMATION = 20  # physics steps per policy step
SIM_DT = 0.001  # Increased frequency for stability with high gains
STEP_DT = SIM_DT * DECIMATION

# ── Joint ordering ─────────────────────────────────────────────────────────────
# MuJoCo Menagerie Quadruped joint order (from XML body tree):
#   FR_hip, FR_thigh, FR_calf,
#   FL_hip, FL_thigh, FL_calf,
#   RR_hip, RR_thigh, RR_calf,
#   RL_hip, RL_thigh, RL_calf
#
# Isaac Lab regex '.*_hip_joint|.*_thigh_joint|.*_calf_joint' on the USD asset
# returns joints in USD prim order, which for Unitree Quadruped is typically:
#   FL_hip, FL_thigh, FL_calf,
#   FR_hip, FR_thigh, FR_calf,
#   RL_hip, RL_thigh, RL_calf,
#   RR_hip, RR_thigh, RR_calf
#
# Permutation to remap ISAAC→MUJOCO (index i goes to position ISAAC_TO_MJ[i]):
# CP7 expected order: Leg-by-leg [FR, FL, RR, RL]
ISAAC_JOINT_NAMES = [
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

# Will be resolved at runtime against model.joint_names
MJ_TO_ISAAC_IDX: list[int] | None = None

# ── Download MJCF if needed ────────────────────────────────────────────────────

ASSETS_DIR = Path(__file__).parent / "mujoco_menagerie"
MENAGERIE_URL = (
    "https://github.com/google-deepmind/mujoco_menagerie/archive/refs/heads/main.zip"
)

ROBOT_DIRS = {
    "go1": "unitree_go1",
    "a1": "unitree_a1",
    "go2": "unitree_go2",
}


def ensure_mjcf(robot: str) -> Path:
    robot_dir = ASSETS_DIR / ROBOT_DIRS[robot]
    scene_xml = robot_dir / "scene.xml"
    if scene_xml.exists():
        return scene_xml

    print(f"[Sim2Sim] MJCF not found — downloading MuJoCo Menagerie for '{robot}'…")
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = ASSETS_DIR / "menagerie.zip"

    def _progress(block, block_size, total):
        done = block * block_size
        if total > 0:
            print(f"\r  {done/1e6:.1f}/{total/1e6:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(MENAGERIE_URL, zip_path, _progress)
    print()

    print("[Sim2Sim] Extracting…")
    with zipfile.ZipFile(zip_path, "r") as zf:
        prefix = f"mujoco_menagerie-main/{ROBOT_DIRS[robot]}/"
        members = [m for m in zf.namelist() if m.startswith(prefix)]
        for member in members:
            rel = member[len("mujoco_menagerie-main/") :]
            dest = ASSETS_DIR / rel
            if member.endswith("/"):
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(dest, "wb") as dst:
                    dst.write(src.read())

    zip_path.unlink()
    print(f"[Sim2Sim] Downloaded to {robot_dir}")
    return scene_xml


# ── Policy network (mirrors skrl_ppo_cfg.yaml architecture) ───────────────────
# Network: [obs] → ELU(512) → ELU(256) → ELU(128) → [mean_actions]
# Checkpoint also contains RunningStandardScaler state for obs and values.


class PolicyMLP(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net_container = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
        )
        self.policy_layer = nn.Linear(128, act_dim)
        # log_std is a learnable parameter in GaussianMixin
        self.log_std_parameter = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.net_container(obs)
        return self.policy_layer(x)


class RunningScaler:
    """Mirrors skrl's RunningStandardScaler."""

    def __init__(self, running_mean: torch.Tensor, running_var: torch.Tensor):
        self.mean = running_mean
        self.std = running_var.sqrt().clamp(min=1e-8)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / self.std).clamp(-5.0, 5.0)


def load_policy(ckpt_path: str):
    """Load policy weights + obs scaler from an skrl checkpoint."""
    state = torch.load(ckpt_path, map_location="cpu")

    # skrl saves nested state dicts; structure depends on version
    # Try common layouts
    def _find(d, key):
        if key in d:
            return d[key]
        for v in d.values():
            if isinstance(v, dict):
                r = _find(v, key)
                if r is not None:
                    return r
        return None

    policy = PolicyMLP(OBS_DIM, ACTION_DIM)

    # Extract policy weights — skrl PPO stores policy under "policy"
    policy_sd = _find(state, "policy")
    if policy_sd is None:
        policy_sd = state

    # Robust key extraction: some skrl versions prefix with "_model." or "best_model."
    # We look for the dictionary that actually contains the "net.0.weight" or similar keys.
    def _is_state_dict(sd):
        if not hasattr(sd, "keys"):
            return False
        return any(
            k.startswith("net.")
            or k.startswith("net_container.")
            or k.startswith("policy_layer.")
            or k.startswith("log_std")
            for k in sd.keys()
        )

    if not _is_state_dict(policy_sd):
        # Search deeper for a dict that LOOKS like a state dict
        found = False
        for k, v in state.items():
            if isinstance(v, (dict, type(state))) and _is_state_dict(v):
                policy_sd = v
                found = True
                print(f"[Sim2Sim] Found valid state dict in '{k}'")
                break
        if not found:
            print(
                f"[WARN] No valid policy state dict found. Keys: {list(policy_sd.keys()) if hasattr(policy_sd, 'keys') else 'Not a dict'}"
            )

    # Filter to only policy-related keys
    net_keys = {
        k: v
        for k, v in policy_sd.items()
        if k.startswith("net.")
        or k.startswith("net_container.")
        or k.startswith("policy_layer.")
        or k.startswith("log_std")
    }

    if net_keys:
        policy.load_state_dict(net_keys, strict=False)
    else:
        print("[WARN] No matching policy weights found — trying full load")
        try:
            policy.load_state_dict(policy_sd, strict=False)
        except Exception as e:
            print(f"[WARN] Full load failed: {e}")

    policy.eval()

    # Extract obs scaler (RunningStandardScaler)
    scaler = None
    scaler_mean = _find(state, "running_mean")
    scaler_var = _find(state, "running_variance")
    if scaler_mean is not None and scaler_var is not None:
        scaler = RunningScaler(scaler_mean.float(), scaler_var.float())
        print("[Sim2Sim] Loaded RunningStandardScaler")
    else:
        print("[Sim2Sim] No obs scaler found — running without normalization")

    print(f"[Sim2Sim] Policy loaded from: {ckpt_path}")
    return policy, scaler


# ── MuJoCo helpers ────────────────────────────────────────────────────────────


def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """MuJoCo quaternion (w, x, y, z) → 3×3 rotation matrix (world←body)."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def build_obs(
    data,
    mj_to_isaac: np.ndarray,
    last_actions: np.ndarray,
    commands: np.ndarray,
    desired_qpos: np.ndarray,
    num_joints: int,
) -> np.ndarray:
    """Build a 49-dim obs vector matching Isaac Lab's _get_observations()."""

    # Base quaternion (w, x, y, z) in MuJoCo
    quat = data.qpos[3:7]
    R = quat_to_rot_matrix(quat)  # world ← body

    # Linear velocity in body frame:  v_b = R^T @ v_w
    lin_vel_w = data.qvel[0:3]
    lin_vel_b = R.T @ lin_vel_w  # [3]

    # Angular velocity in body frame (MuJoCo gives body frame for freejoint)
    ang_vel_b = data.qvel[3:6]

    # Projected gravity: rotate world-down into body frame
    gravity_w = np.array([0.0, 0.0, -1.0])
    proj_grav = R.T @ gravity_w  # [3]

    # Joint positions / velocities in Isaac Lab joint order
    mj_qpos = data.qpos[7 : 7 + num_joints]  # MuJoCo actuated joints
    mj_qvel = data.qvel[6 : 6 + num_joints]
    jpos_isaac = mj_qpos[mj_to_isaac]  # [12]
    jvel_isaac = mj_qvel[mj_to_isaac]  # [12]

    obs = np.concatenate(
        [
            lin_vel_b,  # [3]
            ang_vel_b,  # [3]
            proj_grav,  # [3]
            commands,  # [4]  vx, vy, wz, 0
            jpos_isaac - desired_qpos,  # [12] Position relative to default (fixed)
            jvel_isaac,  # [12]
            last_actions,  # [12]
        ]
    )  # total = 49
    return obs.astype(np.float32)


def resolve_joint_order(model) -> np.ndarray:
    """
    Build an index array `mj_to_isaac` such that
    `mj_qpos[mj_to_isaac]` gives joints in Isaac Lab order.
    """
    import mujoco

    # Get MuJoCo joint names (actuated joints only; skip freejoint)
    mj_names = []
    for i in range(model.njnt):
        jtype = model.jnt_type[i]
        if jtype == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            mj_names.append(name)

    print(f"\n[Sim2Sim] MuJoCo actuated joints ({len(mj_names)}):")
    for idx, n in enumerate(mj_names):
        print(f"  [{idx:2d}] {n}")

    # Strip "_joint" suffix from Isaac names to match MuJoCo (which may not have suffix)
    def _norm(s: str) -> str:
        return s.replace("_joint", "").lower()

    mj_norm = [_norm(n) for n in mj_names]

    mj_to_isaac = np.zeros(ACTION_DIM, dtype=np.int32)
    for isaac_idx, isaac_name in enumerate(ISAAC_JOINT_NAMES):
        norm = _norm(isaac_name)
        try:
            mj_idx = mj_norm.index(norm)
        except ValueError:
            # Fall back: try without suffix
            norm2 = norm.replace("_hip", "").replace("_thigh", "").replace("_calf", "")
            matches = [i for i, n in enumerate(mj_norm) if norm in n]
            if not matches:
                print(
                    f"[WARN] Could not map Isaac joint '{isaac_name}' — using index {isaac_idx}"
                )
                mj_idx = isaac_idx
            else:
                mj_idx = matches[0]
        mj_to_isaac[isaac_idx] = mj_idx

    print(f"\n[Sim2Sim] Isaac→MuJoCo joint remap:")
    for i, mi in enumerate(mj_to_isaac):
        print(
            f"  Isaac[{i:2d}] {ISAAC_JOINT_NAMES[i]:30s} ← MuJoCo[{mi:2d}] {mj_names[mi]}"
        )
    print()
    return mj_to_isaac


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    import mujoco
    import mujoco.viewer

    # 1. Load MJCF
    scene_xml = ensure_mjcf(args.robot)
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)

    model.opt.timestep = SIM_DT

    num_joints = ACTION_DIM  # 12 actuated joints

    # 2. Joint order remap
    mj_to_isaac = resolve_joint_order(model)
    # Inverse: isaac_to_mj[isaac_idx] = mj_idx
    isaac_to_mj = np.zeros(ACTION_DIM, dtype=np.int32)
    for i, mi in enumerate(mj_to_isaac):
        isaac_to_mj[i] = mi
    for i in range(ACTION_DIM):
        print(f"Isaac[{i}] -> MuJoCo[{isaac_to_mj[i]}]")

    # 3. Default joint positions (resting pose from Isaac Lab training)
    # Different robots use slightly different default poses.
    if args.robot == "go2":
        desired_qpos_isaac = np.array(
            [
                0.1,
                -0.1,
                0.1,
                -0.1,  # Hips: FL, FR, RL, RR
                0.8,
                0.8,
                1.0,
                1.0,  # Thighs: FL, FR, RL, RR
                -1.5,
                -1.5,
                -1.5,
                -1.5,  # Calves: FL, FR, RL, RR
            ],
            dtype=np.float32,
        )
    elif args.robot == "a1":
        desired_qpos_isaac = np.array(
            [
                0.1,
                -0.1,
                0.1,
                -0.1,  # Hips: FL, FR, RL, RR
                0.8,
                0.8,
                1.0,
                1.0,  # Thighs: FL, FR, RL, RR
                -1.5,
                -1.5,
                -1.5,
                -1.5,  # Calves: FL, FR, RL, RR
            ],
            dtype=np.float32,
        )
    else:  # quadruped
        desired_qpos_isaac = np.array(
            [
                0.1,
                -0.1,
                0.1,
                -0.1,  # Hips: FL, FR, RL, RR
                0.8,
                0.8,
                1.0,
                1.0,  # Thighs: FL, FR, RL, RR
                -1.5,
                -1.5,
                -1.5,
                -1.5,  # Calves: FL, FR, RL, RR
            ],
            dtype=np.float32,
        )

    # Remap back to MuJoCo order for resetting the robot
    desired_qpos_mj = np.zeros(num_joints, dtype=np.float64)
    for i, mi in enumerate(isaac_to_mj):
        desired_qpos_mj[mi] = desired_qpos_isaac[i]

    # Reconfigure actuators for raw TORQUE control (using ActuatorNet)
    for i in range(model.nu):
        model.actuator_gainprm[i, 0] = 1.0  # kp
        model.actuator_biasprm[i, 1] = 0.0  # remove position bias
        model.actuator_ctrllimited[i] = 0  # remove position limits on ctrl
        model.actuator_forcerange[i, 0] = -23.7
        model.actuator_forcerange[i, 1] = 23.7
    for i in range(model.njnt):
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
            # Match physical joint properties from Isaac Lab training (avg values)
            model.dof_damping[model.jnt_dofadr[i]] = 0.15
            model.dof_frictionloss[model.jnt_dofadr[i]] = 0.05

    mujoco.mj_resetData(model, data)
    # Set to the Isaac home pose
    data.qpos[7 : 7 + num_joints] = desired_qpos_mj
    # Safe base height (touching the ground)
    data.qpos[2] = 0.50  # Match actual standing height to avoid freefall panic
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # Verify upright orientation
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    print(f"[Sim2Sim] Reset to Isaac training home pose")

    # 4. Load policy
    policy, scaler = load_policy(args.checkpoint)

    # 4.5 Load ActuatorNet
    act_net_path = Path(__file__).parent / f"unitree_{args.robot}.pt"
    if act_net_path.exists():
        print(f"[Sim2Sim] Loading ActuatorNet: {act_net_path}")
        act_net = torch.jit.load(str(act_net_path), map_location="cpu").eval()
    else:
        # Fallback to Quadruped model if specific one is missing, as they share similar dynamics
        print(
            f"[Sim2Sim] act_net not found: {act_net_path} — Falling back to unitree_quadruped.pt"
        )
        act_net = torch.jit.load(str(Path(__file__).parent / "unitree_quadruped.pt"), map_location="cpu").eval()

    pos_err_hist = np.zeros((3, 12), dtype=np.float32)
    vel_hist = np.zeros((3, 12), dtype=np.float32)

    # 5. Teleop state — shared between main loop and keyboard thread
    import threading

    _stdin_fd = sys.stdin.fileno()
    is_tty = os.isatty(_stdin_fd)
    _original_term = None
    if is_tty:
        import tty, termios as _termios

        _original_term = _termios.tcgetattr(_stdin_fd)
    else:
        print("[Sim2Sim] No TTY detected — running without keyboard controls")

    cmd_vx = args.vx
    cmd_vy = 0.0
    cmd_wz = 0.0
    cmd_step = 0.05
    speed_mult = 1.0
    _stop_kb = threading.Event()

    commands = np.array([cmd_vx, 0.0, 0.0, 0.0], dtype=np.float32)

    def _print_cmd():
        print(
            f"\r[Cmd] vx={commands[0]:+.2f}  vy={commands[1]:+.2f}"
            f"  wz={commands[2]:+.2f}  speed={speed_mult:.1f}x   ",
            end="",
            flush=True,
        )

    def _keyboard_thread():
        """Read raw keypresses from the terminal in a background thread."""
        if not is_tty:
            return
        print("[Sim2Sim] Keyboard thread started")
        import tty, termios as _termios

        nonlocal cmd_vx, cmd_vy, cmd_wz, speed_mult, commands
        fd = sys.stdin.fileno()
        old = _termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while not _stop_kb.is_set():
                ch = sys.stdin.read(1)
                updated = True
                cu = ch.upper()
                if cu == "W":
                    cmd_vx = min(1.0, cmd_vx + cmd_step)
                    print(f"\n[KBD] W -> vx={cmd_vx:.2f}")
                elif cu == "O":
                    nonlocal show_debug_obs
                    show_debug_obs = not show_debug_obs
                    print(f"\n[DEBUG] Show obs: {show_debug_obs}")
                elif cu == "S":
                    cmd_vx = max(-1.0, cmd_vx - cmd_step)
                    print(f"\n[KBD] S -> vx={cmd_vx:.2f}")
                elif cu == "A":
                    cmd_vy = min(1.0, cmd_vy + cmd_step)
                    print(f"\n[KBD] A -> vy={cmd_vy:.2f}")
                elif cu == "D":
                    cmd_vy = max(-1.0, cmd_vy - cmd_step)
                    print(f"\n[KBD] D -> vy={cmd_vy:.2f}")
                elif cu == "Q":
                    cmd_wz = min(1.0, cmd_wz + cmd_step)
                    print(f"\n[KBD] Q -> wz={cmd_wz:.2f}")
                elif cu == "E":
                    cmd_wz = max(-1.0, cmd_wz - cmd_step)
                    print(f"\n[KBD] E -> wz={cmd_wz:.2f}")
                elif cu == "R":
                    cmd_vx = cmd_vy = cmd_wz = 0.0
                    print(f"\n[KBD] R -> RESET")
                elif ch == "=":
                    speed_mult = round(min(3.0, speed_mult + 0.1), 1)
                elif ch == "-":
                    speed_mult = round(max(0.1, speed_mult - 0.1), 1)
                elif ch == "\x03":  # Ctrl+C
                    _stop_kb.set()
                    break
                else:
                    updated = False
                if updated:
                    commands[:] = [
                        cmd_vx * speed_mult,
                        cmd_vy * speed_mult,
                        cmd_wz * speed_mult,
                        0.0,
                    ]
                    _print_cmd()
        finally:
            _termios.tcsetattr(fd, _termios.TCSADRAIN, old)

    kb_thread = threading.Thread(target=_keyboard_thread, daemon=True)
    kb_thread.start()

    # 6. Simulation state
    last_actions = np.zeros(ACTION_DIM, dtype=np.float32)
    step_count = 0
    total_reward = 0.0

    print(
        "\n[Sim2Sim] Controls (type in THIS terminal):\n"
        "  W/S=fwd  A/D=strafe  Q/E=turn  R=stop  =/- speed  O=debug obs"
    )
    print(f"[Sim2Sim] Robot: {args.robot}  |  dt={STEP_DT*1000:.1f}ms/policy step\n")
    _print_cmd()

    show_debug_obs = False

    def run_step(viewer=None):
        nonlocal last_actions, step_count, total_reward, pos_err_hist, vel_hist

        # Build observation
        obs_np = build_obs(
            data, mj_to_isaac, last_actions, commands, desired_qpos_isaac, num_joints
        )
        obs_t = torch.from_numpy(obs_np).unsqueeze(0)  # [1, 49]

        if scaler is not None:
            obs_t = scaler(obs_t)

        if show_debug_obs and step_count % 50 == 0:
            o_norm = obs_t.squeeze(0).cpu().numpy()
            o_raw = obs_np
            print(f"\n[OBS] Step {step_count}")
            print(f"  Gravity (Raw):  {o_raw[6:9]}")
            print(f"  Gravity (Norm): {o_norm[6:9]}  (level would be ~0,0,-1 raw)")

            # Print scaler params for Gravity
            if scaler is not None:
                gm = scaler.mean[6:9].cpu().numpy()
                gs = torch.sqrt(scaler.var[6:9] + 1e-8).cpu().numpy()
                print(f"  [Scaler] Grav Mean: {gm}, Std: {gs}")

            print(f"  LinVel (Raw):   {o_raw[0:3]}")
            print(f"  LinVel (Norm):  {o_norm[0:3]}")
            print(f"  PosErr (Raw):   {o_raw[13:25]}")
            print(f"  PosErr (Norm):  {o_norm[13:25]}")

            # Print raw data for sanity check
            root_quat = data.qpos[3:7]  # w,x,y,z
            print(f"  [RAW] Quat (w,x,y,z): {root_quat}")
            print(f"  [RAW] qpos[2] (height): {data.qpos[2]:.3f}")

        # Settle time: hold home pose for first 3.0s (150 steps)
        # Policy forward pass (deterministic mean)
        with torch.no_grad():
            actions_t = policy(obs_t)  # [1, 12]
        actions_isaac = actions_t.squeeze(0).numpy()

        targets_isaac = actions_isaac * ACTION_SCALE + desired_qpos_isaac

        if step_count == 100:
            print(f"STEP 100 TARGETS (Isaac order): {targets_isaac.tolist()}")
            qpos_mj_actuated = data.qpos[7:19]
            print(
                f"STEP 100 ACTUAL  (Isaac order): {qpos_mj_actuated[mj_to_isaac].tolist()}"
            )

        # ── Actuator Loop ────────────────────────────────────────────────────────
        # Isaac Lab training (cp7) uses sim_dt=0.005 (200Hz).
        # MuJoCo SIM_DT=0.001 (1000Hz).
        # We evaluate the ActuatorNet every ACT_DECIMATION steps.
        ACT_DECIMATION = 5  # Evaluate actuator every 5ms (200Hz)

        for i in range(DECIMATION):
            if i % ACT_DECIMATION == 0:
                # ── Evaluate ActuatorNet ──
                mj_qpos_i = data.qpos[7 : 7 + num_joints]
                mj_qvel_i = data.qvel[6 : 6 + num_joints]
                jpos_isaac_i = mj_qpos_i[mj_to_isaac]
                jvel_isaac_i = mj_qvel_i[mj_to_isaac]

                # pos_err = current - target (to match pos_scale=-1.0 in Isaac Lab training)
                pos_err = jpos_isaac_i - targets_isaac

                pos_err_hist = np.roll(pos_err_hist, shift=1, axis=0)
                pos_err_hist[0, :] = pos_err

                vel_hist = np.roll(vel_hist, shift=1, axis=0)
                vel_hist[0, :] = jvel_isaac_i

                # Build input vector [12 joints, 6 features]
                net_in = np.zeros((num_joints, 6), dtype=np.float32)
                net_in[:, 0:3] = pos_err_hist.T
                net_in[:, 3:6] = vel_hist.T

                with torch.no_grad():
                    net_in_t = torch.from_numpy(net_in)
                    torques_isaac = act_net(net_in_t).squeeze().numpy()  # [12]

                # Clamp output to motor capability limit (23.7 N*m for Quadruped)
                torques_isaac = np.clip(torques_isaac, -23.7, 23.7)

                # Apply to MuJoCo
                torques_mj = np.zeros(num_joints, dtype=np.float64)
                for j, mj_idx in enumerate(isaac_to_mj):
                    torques_mj[mj_idx] = torques_isaac[j]

                data.ctrl[:num_joints] = torques_mj

            # Step physics
            mujoco.mj_step(model, data)

        # Store actions for next obs
        last_actions[:] = actions_isaac

        # Simple reward: match commanded velocity (for printout only)
        base_quat = data.qpos[3:7]
        R = quat_to_rot_matrix(base_quat)
        lin_vel_b = R.T @ data.qvel[0:3]
        vel_err = np.sum((lin_vel_b[:2] - commands[:2]) ** 2)
        reward = float(np.exp(-vel_err / 0.25))  # simplified tracking reward
        total_reward += reward
        step_count += 1

        if step_count < 10:
            print(f"[Step {step_count}] policy_out: {actions_isaac[:3].tolist()}")
            if step_count <= 2:
                print(f"[Step {step_count}] torques: {torques_isaac.tolist()}")

        if step_count % 20 == 0:
            base_z = data.qpos[2]
            dt_wall = max(
                1e-6, time.time() - (start_wall_time + (step_count - 20) * STEP_DT)
            )
            hz_actual = 20.0 / dt_wall
            print(
                f"\r[Step {step_count:6d}] "
                f"vx={lin_vel_b[0]:+.2f}  vy={lin_vel_b[1]:+.2f}  "
                f"h={base_z:.3f}  rew={reward:.3f}  "
                f"cmd=[{commands[0]:+.2f},{commands[1]:+.2f},{commands[2]:+.2f}]  "
                f"Hz={hz_actual:.1f}   ",
                end="",
                flush=True,
            )

    # 7. Run
    try:
        if args.no_render:
            start_wall_time = time.time()
            while not _stop_kb.is_set():
                run_step()
                if args.duration > 0 and step_count * STEP_DT >= args.duration:
                    break

                # Robust wall-clock synchronization
                target_wall_time = start_wall_time + (step_count * STEP_DT)
                while time.time() < target_wall_time:
                    time.sleep(0.001)
        else:
            import mujoco.viewer

            def viewer_key_callback(keycode):
                nonlocal cmd_vx, cmd_vy, cmd_wz, speed_mult, commands
                updated = True
                try:
                    ch = chr(keycode).upper()
                    print(f"\r[Viewer KBD] Key registered: {keycode} ({ch})   ", end="")
                    if ch == "W":
                        cmd_vx = min(1.0, cmd_vx + cmd_step)
                        print(f"\r[KBD] W -> vx={cmd_vx:.2f}   ", end="")
                    elif ch == "S":
                        cmd_vx = max(-1.0, cmd_vx - cmd_step)
                        print(f"\r[KBD] S -> vx={cmd_vx:.2f}   ", end="")
                    elif ch == "A":
                        cmd_vy = min(1.0, cmd_vy + cmd_step)
                        print(f"\r[KBD] A -> vy={cmd_vy:.2f}   ", end="")
                    elif ch == "D":
                        cmd_vy = max(-1.0, cmd_vy - cmd_step)
                        print(f"\r[KBD] D -> vy={cmd_vy:.2f}   ", end="")
                    elif ch == "Q":
                        cmd_wz = min(1.0, cmd_wz + cmd_step)
                        print(f"\r[KBD] Q -> wz={cmd_wz:.2f}   ", end="")
                    elif ch == "E":
                        cmd_wz = max(-1.0, cmd_wz - cmd_step)
                        print(f"\r[KBD] E -> wz={cmd_wz:.2f}   ", end="")
                    elif ch == "R":
                        cmd_vx = cmd_vy = cmd_wz = 0.0
                        print(f"\r[KBD] R -> RESET             ", end="")
                    elif ch == "=" or keycode == 333:
                        speed_mult = round(min(3.0, speed_mult + 0.1), 1)
                    elif ch == "-" or keycode == 334:
                        speed_mult = round(max(0.1, speed_mult - 0.1), 1)
                    else:
                        updated = False
                except Exception as e:
                    print(
                        f"\r[Viewer KBD] Error processing keycode {keycode}: {e}   ",
                        end="",
                    )
                    updated = False

                if updated:
                    commands[:] = [
                        cmd_vx * speed_mult,
                        cmd_vy * speed_mult,
                        cmd_wz * speed_mult,
                        0.0,
                    ]

            with mujoco.viewer.launch_passive(
                model, data, key_callback=viewer_key_callback
            ) as viewer:
                # Track the base body (search for common names in Menagerie: trunk, base, base_link)
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                track_id = -1
                for body_name in ["trunk", "base", "base_link"]:
                    track_id = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_BODY, body_name
                    )
                    if track_id != -1:
                        break
                viewer.cam.trackbodyid = track_id

                start_wall_time = time.time()
                while viewer.is_running() and not _stop_kb.is_set():
                    run_step(viewer)
                    viewer.sync()
                    if args.duration > 0 and step_count * STEP_DT >= args.duration:
                        break

                    # Robust wall-clock synchronization
                    # We want: actual_elapsed >= total_sim_time
                    target_wall_time = start_wall_time + (step_count * STEP_DT)
                    while time.time() < target_wall_time:
                        # Sleep very briefly to avoid 100% CPU but maintain accuracy
                        time.sleep(0.001)
    finally:
        _stop_kb.set()
        # Always restore terminal settings (daemon thread's finally may not run)
        if is_tty and _original_term is not None:
            import termios as _termios

            try:
                _termios.tcsetattr(_stdin_fd, _termios.TCSADRAIN, _original_term)
            except Exception:
                pass

    print(
        f"\n[Sim2Sim] Done — {step_count} steps, "
        f"avg reward = {total_reward/max(1,step_count):.4f}"
    )


if __name__ == "__main__":
    # Prefer EGL for software/headless rendering if hardware GL is missing
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"
    main()
