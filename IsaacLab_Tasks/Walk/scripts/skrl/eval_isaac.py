# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Automated Evaluation Script for Quadruped Policies
Calculates foot lift, GRF, synchronization, and base oscillations across multiple speeds.
"""

import argparse
import sys
import os
import json

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Automated Evaluation of an RL agent from skrl."
)
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during evaluation."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument(
    "--checkpoint", type=str, default=None, help="Path to model checkpoint."
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import random
import time
import numpy as np

import gymnasium as gym
import skrl
import torch
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import Quadruped.tasks  # noqa: F401

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = (
        "skrl_cfg_entry_point"
        if algorithm in ["ppo"]
        else f"skrl_{algorithm}_cfg_entry_point"
    )
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    experiment_cfg: dict,
):
    """Play with skrl agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    experiment_cfg["seed"] = (
        args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    )
    env_cfg.seed = experiment_cfg["seed"]

    log_root_path = os.path.join(
        "logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"]
    )
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path,
            run_dir=f".*_{algorithm}_{args_cli.ml_framework}",
            other_dirs=["checkpoints"],
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(
        env, ml_framework=args_cli.ml_framework
    )

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.set_running_mode("eval")

    # Evaluation Config
    speeds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.00]
    steps_per_speed = 250
    warmup_steps = 50
    
    results = {}
    
    # Internal variables mapping
    unwrapped_env = env.unwrapped.unwrapped
    device = unwrapped_env.device
    num_envs = unwrapped_env.num_envs
    
    # Check if heterogeneous
    is_heterogeneous = getattr(unwrapped_env, "is_heterogeneous", False)

    print("\n" + "="*60)
    print("🚀 STARTING AUTOMATED POLICY EVALUATION")
    print("="*60)

    for speed in speeds:
        print(f"\n[EVAL] Testing forward velocity: {speed} m/s")
        
        # Reset environment
        obs, _ = env.reset()
        
        # Metric buffers
        foot_heights_max = torch.zeros((num_envs, 4), device=device)
        foot_swing_times = []
        grf_stance_sum = torch.zeros((num_envs, 4), device=device)
        grf_stance_counts = torch.zeros((num_envs, 4), device=device)
        
        sync_match_count = torch.zeros((num_envs,), device=device)
        total_eval_steps = 0
        
        base_lin_vel_z = []
        base_ang_vel_roll = []
        base_ang_vel_pitch = []
        
        actual_forward_vel = []

        # Force command zero during warmup (let it stand and stabilize)
        unwrapped_env.target_commands[:, :3] = 0.0
        unwrapped_env.commands[:, :3] = 0.0
        
        for step in range(steps_per_speed):
            # Apply command
            if step == warmup_steps:
                print("   -> Warmup finished, applying velocity command...")
                
            if step >= warmup_steps:
                unwrapped_env.target_commands[:, 0] = speed
                unwrapped_env.target_commands[:, 1:3] = 0.0
                unwrapped_env.commands[:, 0] = speed
                unwrapped_env.commands[:, 1:3] = 0.0
                
                # Prevent resampling
                unwrapped_env.command_timer[:] = 0.0

            with torch.inference_mode():
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
                if hasattr(env, "possible_agents"):
                    actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
                else:
                    actions = outputs[-1].get("mean_actions", outputs[0])
                obs, _, _, _, _ = env.step(actions)
                
            # Collect Metrics (only after warmup)
            if step >= warmup_steps:
                total_eval_steps += 1
                
                # Extract physics data
                if is_heterogeneous:
                    # Approximation: just grab the first robot type's feet ids.
                    # Or use all_feet_heights logic. To be safe, we will just assume homogeneous for now
                    # or safely get the height if we can.
                    feet_pos = torch.zeros((num_envs, 4, 3), device=device)
                    for i, view in enumerate(unwrapped_env.robot_views):
                        indices = unwrapped_env.robot_view_indices[i]
                        f_ids = unwrapped_env.robot_feet_ids[i]
                        feet_pos[indices] = view.data.body_pos_w[:, f_ids, :]
                else:
                    feet_pos = unwrapped_env.body_pos_w[:, unwrapped_env._feet_ids_articulation, :]
                    
                feet_z = feet_pos[:, :, 2]
                contact = unwrapped_env.last_feet_contact # shape: (num_envs, 4)
                
                # Foot lift height (update max during swing)
                swinging = ~contact
                foot_heights_max = torch.max(foot_heights_max, feet_z * swinging.float())
                
                # Foot swing time (captured when foot lands)
                just_landed = contact & (unwrapped_env.feet_air_time > 0)
                if just_landed.any():
                    # Average over the envs that just landed
                    landed_times = unwrapped_env.feet_air_time[just_landed].cpu().numpy()
                    foot_swing_times.extend(landed_times.tolist())
                    
                # GRF during stance
                grf_z = unwrapped_env.net_contact_forces[:, unwrapped_env._feet_ids, 2].abs()
                stance = contact
                grf_stance_sum += grf_z * stance.float()
                grf_stance_counts += stance.float()
                
                # Sync: Check diagonal pairs (FL=0, FR=1, RL=2, RR=3)
                # Trot diagonals: (0,3) and (1,2)
                pair1_sync = (contact[:, 0] == contact[:, 3])
                pair2_sync = (contact[:, 1] == contact[:, 2])
                # We consider it "in sync" if both pairs are synchronized (i.e. one pair swinging, other stance)
                # Or simply measure the time fraction where diagonals match.
                sync_match_count += (pair1_sync & pair2_sync).float()
                
                # Base oscillations
                base_lin_vel_z.append(unwrapped_env.base_lin_vel[:, 2].cpu().numpy())
                base_ang_vel_roll.append(unwrapped_env.base_ang_vel[:, 0].cpu().numpy())
                base_ang_vel_pitch.append(unwrapped_env.base_ang_vel[:, 1].cpu().numpy())
                
                # Actual forward velocity
                actual_forward_vel.append(unwrapped_env.base_lin_vel[:, 0].cpu().numpy())

        # Process metrics for this speed
        # Averages across all envs
        avg_foot_height_max = (foot_heights_max.sum(dim=0) / num_envs).cpu().numpy().tolist() # [FL, FR, RL, RR]
        avg_swing_time = np.mean(foot_swing_times) if len(foot_swing_times) > 0 else 0.0
        
        # Average GRF per foot when in stance
        avg_grf = (grf_stance_sum / grf_stance_counts.clamp(min=1.0)).sum(dim=0) / num_envs
        avg_grf = avg_grf.cpu().numpy().tolist()
        
        sync_percentage = (sync_match_count.sum() / (num_envs * total_eval_steps)).item() * 100.0
        
        base_lin_vel_z = np.array(base_lin_vel_z) # shape (steps, envs)
        base_ang_vel_roll = np.array(base_ang_vel_roll)
        base_ang_vel_pitch = np.array(base_ang_vel_pitch)
        actual_forward_vel = np.array(actual_forward_vel)
        
        std_z = np.mean(np.std(base_lin_vel_z, axis=0))
        std_roll = np.mean(np.std(base_ang_vel_roll, axis=0))
        std_pitch = np.mean(np.std(base_ang_vel_pitch, axis=0))
        
        avg_actual_vel = np.mean(actual_forward_vel)

        results[str(speed)] = {
            "commanded_speed": speed,
            "actual_speed": float(avg_actual_vel),
            "foot_lift_height_m": avg_foot_height_max,
            "foot_swing_time_s": float(avg_swing_time),
            "grf_stance_N": avg_grf,
            "trot_sync_percent": float(sync_percentage),
            "base_oscillation": {
                "std_z_vel": float(std_z),
                "std_roll_vel": float(std_roll),
                "std_pitch_vel": float(std_pitch)
            }
        }
        
        print(f"   => Actual Fwd Vel: {avg_actual_vel:.3f} m/s")
        print(f"   => Foot Lift Height (FL,FR,RL,RR): {[round(v, 4) for v in avg_foot_height_max]} m")
        print(f"   => Average Swing Time: {avg_swing_time:.3f} s")
        print(f"   => Stance GRF (FL,FR,RL,RR): {[round(v, 1) for v in avg_grf]} N")
        print(f"   => Trot Synchronization: {sync_percentage:.1f}%")
        print(f"   => Base Oscillation (Z_vel, Roll, Pitch): {std_z:.3f} m/s, {std_roll:.3f} rad/s, {std_pitch:.3f} rad/s")

    # Output JSON Report
    report_path = os.path.join(log_dir, "eval_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n" + "="*60)
    print(f"✅ EVALUATION COMPLETE. Report saved to: {report_path}")
    print("="*60 + "\n")

    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
