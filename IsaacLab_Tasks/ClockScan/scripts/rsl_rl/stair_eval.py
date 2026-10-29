"""Stair test against step height, measured in Isaac Sim, as a chart.

Every robot gets a tile of the top block of step counts (6 steps over the training height range), down
(platform on top) or up (pit), or with --heights a tile of 6 steps at exactly one of those heights, and walks at one pinned command from a standing start on the platform, heading +x, for --seconds.
Only its first episode counts. Per tile it records whether the base got --distance m from the tile centre
(past the first edge at 1.5 m) and whether the episode ended by a fall. The step height of each tile is read
from the terrain.

Writes stair_eval_<checkpoint>.png and .json next to the first checkpoint (or --out), one line per checkpoint
and sideways speed:

    python scripts/rsl_rl/stair_eval.py --task=Go2-Clock-Scan --headless \\
        --checkpoint logs/rsl_rl/go2_clock_scan/<run>/model_<n>.pt [--compare <other .pt> ...] [--vy 0,-0.2,0.2]
"""

import argparse
import os

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Stair test against step height for RSL-RL checkpoints.")
parser.add_argument("--task", type=str, default="Go2-Clock-Scan")
parser.add_argument("--compare", type=str, nargs="*", default=[], help="More checkpoints drawn on the same chart.")
parser.add_argument("--speed", type=float, default=0.5, help="Forward command, m/s.")
parser.add_argument("--vy", type=str, default="0", help="Comma-separated sideways commands, m/s; one line each.")
parser.add_argument("--rows", type=int, default=16, help="Step heights tested (terrain rows); tiles per height with --heights.")
parser.add_argument("--heights", type=str, default=None, help="Comma-separated exact step heights, cm, instead of the sweep.")
parser.add_argument("--per_tile", type=int, default=8, help="Robots per tile.")
parser.add_argument("--seconds", type=float, default=12.0)
parser.add_argument("--distance", type=float, default=2.0, help="Distance from the tile centre that counts as across.")
parser.add_argument("--out", type=str, default=None, help="Output path without extension.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, unknown = parser.parse_known_args()
if "--headless" in unknown:
    unknown.remove("--headless")
    args_cli.headless = True
if unknown:
    parser.error(f"unrecognized arguments: {' '.join(unknown)}")
os.environ["CLOCK_SCAN_PLAY_CMD_X"] = str(args_cli.speed)
os.environ["CLOCK_SCAN_PLAY_YAW"] = "0"

simulation_app = AppLauncher(args_cli).app

import gymnasium as gym
import json
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import clock_scan.tasks  # noqa: F401
from clock_scan.tasks.locomotion.robots.go2.clock_scan_env_cfg import ROUGH_TERRAINS_CFG, STEP_COUNTS
from clock_scan.utils.parser_cfg import parse_env_cfg

TYPES = {"down": "pyramid_stairs", "up": "pyramid_stairs_inv"}


@torch.inference_mode()
def rollout(env, policy, level, col):
    """One run of every robot from its tile's platform: the farthest the base got from the tile centre in its
    first episode, and whether that episode ended by a fall."""
    base = env.unwrapped
    robot = base.scene["robot"]
    terrain = base.scene.terrain
    terrain.terrain_levels[:] = level
    terrain.terrain_types[:] = col
    terrain.env_origins[:] = terrain.terrain_origins[level, col]
    base.reset()
    obs = env.get_observations()
    obs = obs[0] if isinstance(obs, tuple) else obs
    n = base.num_envs
    origin = terrain.env_origins[:, :2].clone()
    active = torch.ones(n, dtype=torch.bool, device=base.device)
    far = torch.zeros(n, device=base.device)
    fell = torch.zeros(n, dtype=torch.bool, device=base.device)
    for _ in range(int(args_cli.seconds / base.step_dt)):
        d = torch.linalg.norm(robot.data.root_pos_w.torch[:, :2] - origin, dim=1)
        far = torch.where(active, torch.maximum(far, d), far)
        obs, _, _, _ = env.step(policy(obs))
        fell |= base.termination_manager.terminated & active
        active &= ~(base.termination_manager.terminated | base.termination_manager.time_outs)
    return far, fell


def main():
    checkpoints = [args_cli.checkpoint] + list(args_cli.compare)
    vys = [float(v) for v in args_cli.vy.split(",")]
    steps = max(STEP_COUNTS)
    parts = len(STEP_COUNTS)
    if args_cli.heights:
        heights = [float(h) for h in args_cli.heights.split(",")]
        subs = {}
        for tag, name in TYPES.items():
            for h in heights:
                subs[f"{tag}_{h:g}"] = ROUGH_TERRAINS_CFG.sub_terrains[name].replace(
                    step_counts=(steps,), step_height_range=(h / 100.0, h / 100.0)
                )
        difficulty = (0.0, 1.0)
    else:
        subs = {name: ROUGH_TERRAINS_CFG.sub_terrains[name] for name in TYPES.values()}
        difficulty = ((parts - 1) / parts, 1.0)
    rows, cols = args_cli.rows, len(subs)
    subs = {k: v.replace(proportion=1.0 / cols) for k, v in subs.items()}
    num_envs = rows * cols * args_cli.per_tile
    cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=num_envs, entry_point_key="play_env_cfg_entry_point")
    cfg.scene.terrain.terrain_generator = cfg.scene.terrain.terrain_generator.replace(
        num_rows=rows, num_cols=cols, sub_terrains=subs, difficulty_range=difficulty, curriculum=True
    )
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.episode_length_s = max(cfg.episode_length_s, args_cli.seconds + 1.0)
    agent = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=cfg), clip_actions=agent.clip_actions)
    agent = cli_args.migrate_rsl_rl_cfg(agent)
    base = env.unwrapped
    terrain = base.scene.terrain
    dev = base.device
    cmd = base.command_manager.get_term("base_velocity")

    ids = torch.arange(num_envs, device=dev)
    level = ids % rows
    col = (ids // rows) % cols
    names = list(subs.keys())
    col_tag = ["up" if subs[n].inverted else "down" for n in names]
    height_cm = (100.0 * terrain.terrain_origins[:, :, 2].abs() / steps).cpu()
    env_height = torch.round(height_cm[level.cpu(), col.cpu()] * 10.0) / 10.0
    env_tag = [col_tag[c] for c in col.tolist()]

    results = []
    for ck in checkpoints:
        runner = OnPolicyRunner(env, cli_args.runner_cfg_for_installed_rsl_rl(agent), log_dir=None, device="cuda:0")
        cli_args.load_checkpoint(runner, ck)
        policy = runner.get_inference_policy(device="cuda:0")
        for vy in vys:
            cmd.cfg.ranges.lin_vel_y = (vy, vy)
            far, fell = rollout(env, policy, level, col)
            across = (far > args_cli.distance).cpu()
            fell = fell.cpu()
            line = {"checkpoint": ck, "vy": vy, "speed": args_cli.speed}
            for tag in TYPES:
                is_tag = torch.tensor([t == tag for t in env_tag])
                pts = []
                for h in sorted(set(env_height[is_tag].tolist())):
                    sel = is_tag & (env_height == h)
                    pts.append({
                        "height_cm": h,
                        "robots": int(sel.sum()),
                        "across": round(float(across[sel].float().mean()), 3),
                        "fell": round(float(fell[sel].float().mean()), 3),
                    })
                line[tag] = pts
            results.append(line)
            print(f"[stair_eval] {os.path.basename(ck)} vy={vy:+.2f}: " + "  ".join(
                f"{tag} across {sum(p['across'] for p in line[tag]) / len(line[tag]):.2f} fell {sum(p['fell'] for p in line[tag]) / len(line[tag]):.2f}"
                for tag in TYPES))

    out = args_cli.out or os.path.join(
        os.path.dirname(os.path.abspath(checkpoints[0])), "stair_eval_" + os.path.splitext(os.path.basename(checkpoints[0]))[0]
    )
    with open(out + ".json", "w") as f:
        json.dump({"speed": args_cli.speed, "seconds": args_cli.seconds, "distance": args_cli.distance,
                   "per_tile": args_cli.per_tile, "results": results}, f, indent=1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for k, line in enumerate(results):
        label = os.path.basename(os.path.dirname(line["checkpoint"]))[-8:] + "/" + os.path.basename(line["checkpoint"])
        if len(vys) > 1:
            label += f" vy={line['vy']:+.2f}"
        for ax, tag in zip(axes, TYPES):
            pts = sorted(line[tag], key=lambda p: p["height_cm"])
            h = [p["height_cm"] for p in pts]
            ax.plot(h, [p["across"] for p in pts], "-o", ms=3, color=colors[k % len(colors)], label=label + " across")
            ax.plot(h, [p["fell"] for p in pts], "--x", ms=4, color=colors[k % len(colors)], alpha=0.7, label=label + " fell")
    for ax, tag in zip(axes, TYPES):
        ax.set_title(f"{tag}: {steps} steps, {args_cli.speed:.2f} m/s forward, {results[0][tag][0]['robots']} robots per point")
        ax.set_xlabel("step height (cm)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"share (across = got {args_cli.distance:.1f} m out)")
    axes[1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=110)
    print(f"[stair_eval] wrote {out}.png and .json")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
