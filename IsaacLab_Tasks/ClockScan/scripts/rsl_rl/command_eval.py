"""Command safety test on stairs, measured in Isaac Sim.

Every robot gets a tile at one of --heights: pyramid stairs down (platform on top), up (pit), or the narrow flight
(--flight_width m, open sides) from the upper floor (down) or the lower floor (up), starting as in training. It
first walks forward at --approach m/s (with the operator, if the config has one), and at a random time from 0 to
--switch_max s gets the test command: on the platform, on the steps or at the top. Each command of --commands is
then held for --seconds with no operator and no forward-only limit. Only the first episode counts. Per terrain,
height and command it records the share that fell after the switch (any termination but the time limit) and how
much of the command the robot followed after it (actual velocity along the command over the command; 1 =
followed, 0 = stood still), and the share that fell before it (approach).

Writes command_eval_<checkpoint>.json and .md next to the checkpoint (or --out):

    python scripts/rsl_rl/command_eval.py --task=Go2-Clock-Scan --headless \\
        --checkpoint logs/rsl_rl/go2_clock_scan/<run>/model_<n>.pt
"""

import argparse
import os

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Command safety test on stairs for an RSL-RL checkpoint.")
parser.add_argument("--task", type=str, default="Go2-Clock-Scan")
parser.add_argument(
    "--commands",
    type=str,
    default="x=-0.2;x=-0.4;y=0.3;y=-0.3;yaw=0.5;x=0.3,yaw=0.5;x=0.3,y=0.3;x=-0.3,y=0.2",
    help="Semicolon-separated commands, each comma-separated x, y (m/s) and yaw (rad/s).",
)
parser.add_argument("--heights", type=str, default="15,18", help="Comma-separated step heights, cm.")
parser.add_argument("--rows", type=int, default=4, help="Tiles per terrain and height.")
parser.add_argument("--per_tile", type=int, default=8, help="Robots per tile.")
parser.add_argument("--seconds", type=float, default=15.0, help="Test command time after the switch, s.")
parser.add_argument("--approach", type=float, default=0.4, help="Forward speed before the switch, m/s.")
parser.add_argument(
    "--switch_max", type=float, default=10.0, help="Latest switch time, s (0: test command from the start)."
)
parser.add_argument("--flight_width", type=float, default=0.8, help="Narrow flight width, m.")
parser.add_argument("--out", type=str, default=None, help="Output path without extension.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, unknown = parser.parse_known_args()
if "--headless" in unknown:
    unknown.remove("--headless")
    args_cli.headless = True
if unknown:
    parser.error(f"unrecognized arguments: {' '.join(unknown)}")
os.environ["CLOCK_SCAN_PLAY_CMD_X"] = "0"
os.environ["CLOCK_SCAN_PLAY_YAW"] = "0"

simulation_app = AppLauncher(args_cli).app

import gymnasium as gym
import json
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import clock_scan.tasks  # noqa: F401
from clock_scan.tasks.locomotion.robots.go2.clock_scan_env_cfg import ROUGH_TERRAINS_CFG, STEP_COUNTS
from clock_scan.utils.parser_cfg import parse_env_cfg

TYPES = {
    "pyramid down": "pyramid_stairs",
    "pyramid up": "pyramid_stairs_inv",
    "narrow down": "narrow_stairs",
    "narrow up": "narrow_stairs_inv",
}


def parse_command(text: str) -> tuple[float, float, float]:
    values = {"x": 0.0, "y": 0.0, "yaw": 0.0}
    for part in text.split(","):
        key, value = part.split("=")
        values[key.strip()] = float(value)
    return values["x"], values["y"], values["yaw"]


@torch.inference_mode()
def rollout(env, policy, level, col, test_cmd):
    """One run of every robot: whether it fell before and after its switch, and the sums for the share of the
    test command it followed."""
    base = env.unwrapped
    robot = base.scene["robot"]
    terrain = base.scene.terrain
    cmd = base.command_manager.get_term("base_velocity")
    terrain.terrain_levels[:] = level
    terrain.terrain_types[:] = col
    terrain.env_origins[:] = terrain.terrain_origins[level, col]
    cmd.cfg.ranges.lin_vel_x = (args_cli.approach, args_cli.approach)
    cmd.cfg.ranges.lin_vel_y = (0.0, 0.0)
    cmd.cfg.ranges.ang_vel_z = (0.0, 0.0)
    cmd.forward_only[:] = True
    base.reset()
    obs = env.get_observations()
    obs = obs[0] if isinstance(obs, tuple) else obs
    n = base.num_envs
    dev = base.device
    switch = torch.rand(n, device=dev) * args_cli.switch_max
    switched = torch.zeros(n, dtype=torch.bool, device=dev)
    target = torch.tensor(test_cmd, device=dev)
    active = torch.ones(n, dtype=torch.bool, device=dev)
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    fell_before = torch.zeros(n, dtype=torch.bool, device=dev)
    along = torch.zeros(n, device=dev)
    asked = torch.zeros(n, device=dev)
    steps = int((args_cli.switch_max + args_cli.seconds) / base.step_dt)
    for _ in range(steps):
        t = base.episode_length_buf * base.step_dt
        now = active & ~switched & (t >= switch)
        if bool(now.any()):
            cmd.vel_command_b[now] = target
            cmd.forward_only[now] = False
            cmd.steer_heading[now] = float("nan")
            switched |= now
        done = switched & (t >= switch + args_cli.seconds)
        active &= ~done
        c = cmd.command
        v = torch.cat([robot.data.root_lin_vel_b.torch[:, :2], robot.data.root_ang_vel_b.torch[:, 2:3]], dim=1)
        on = active & switched & (c.abs().sum(dim=1) > 0)
        along += torch.where(on, (v * c).sum(dim=1), torch.zeros_like(along))
        asked += torch.where(on, (c * c).sum(dim=1), torch.zeros_like(asked))
        obs, _, _, _ = env.step(policy(obs))
        ended = base.termination_manager.terminated & active
        fell |= ended & switched
        fell_before |= ended & ~switched
        active &= ~(base.termination_manager.terminated | base.termination_manager.time_outs)
    return fell, fell_before, along, asked


def main():
    heights = [float(h) for h in args_cli.heights.split(",")]
    commands = [c.strip() for c in args_cli.commands.split(";") if c.strip()]
    subs = {}
    for tag, name in TYPES.items():
        for h in heights:
            sub = ROUGH_TERRAINS_CFG.sub_terrains[name]
            if name.startswith("narrow"):
                block = {"widths": (args_cli.flight_width,), "side_probs": (1.0, 0.0, 0.0)}
            else:
                block = {"step_counts": (max(STEP_COUNTS),)}
            subs[f"{tag}_{h:g}"] = sub.replace(step_height_range=(h / 100.0, h / 100.0), **block)
    rows, cols = args_cli.rows, len(subs)
    subs = {k: v.replace(proportion=1.0 / cols) for k, v in subs.items()}
    num_envs = rows * cols * args_cli.per_tile
    cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=num_envs, entry_point_key="play_env_cfg_entry_point")
    cfg.scene.terrain.terrain_generator = cfg.scene.terrain.terrain_generator.replace(
        num_rows=rows, num_cols=cols, sub_terrains=subs, difficulty_range=(0.0, 1.0), curriculum=True
    )
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.episode_length_s = max(cfg.episode_length_s, args_cli.switch_max + args_cli.seconds + 1.0)
    cfg.commands.base_velocity.resampling_time_range = (1e6, 1e6)
    agent = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=cfg), clip_actions=agent.clip_actions)
    agent = cli_args.migrate_rsl_rl_cfg(agent)
    base = env.unwrapped
    cmd = base.command_manager.get_term("base_velocity")

    ids = torch.arange(num_envs, device=base.device)
    level = ids % rows
    col = (ids // rows) % cols
    names = list(subs.keys())
    env_name = [names[c] for c in col.tolist()]

    runner = OnPolicyRunner(env, cli_args.runner_cfg_for_installed_rsl_rl(agent), log_dir=None, device="cuda:0")
    cli_args.load_checkpoint(runner, args_cli.checkpoint)
    policy = runner.get_inference_policy(device="cuda:0")

    results = []
    for text in commands:
        fell, fell_before, along, asked = (t.cpu() for t in rollout(env, policy, level, col, parse_command(text)))
        for name in names:
            sel = torch.tensor([e == name for e in env_name])
            tag, h = name.rsplit("_", 1)
            results.append({
                "command": text, "terrain": tag, "height_cm": float(h), "robots": int(sel.sum()),
                "fell": round(float(fell[sel].float().mean()), 3),
                "fell_before": round(float(fell_before[sel].float().mean()), 3),
                "followed": round(float(along[sel].sum() / asked[sel].sum().clamp(min=1e-9)), 3),
            })
        print(f"[command_eval] {text}: fell {float(fell.float().mean()):.2f}")

    out = args_cli.out or os.path.join(
        os.path.dirname(os.path.abspath(args_cli.checkpoint)),
        "command_eval_" + os.path.splitext(os.path.basename(args_cli.checkpoint))[0],
    )
    with open(out + ".json", "w") as f:
        json.dump({"checkpoint": args_cli.checkpoint, "seconds": args_cli.seconds, "approach": args_cli.approach,
                   "switch_max": args_cli.switch_max, "flight_width": args_cli.flight_width,
                   "results": results}, f, indent=1)
    cells = [(tag, h) for tag in TYPES for h in heights]
    before = max(r["fell_before"] for r in results)
    lines = [f"Fell / followed after the switch, {args_cli.checkpoint}: forward at {args_cli.approach:g} m/s, the"
             f" command at 0-{args_cli.switch_max:g} s, held {args_cli.seconds:g} s with no operator;"
             f" {rows * args_cli.per_tile} robots per cell; narrow {args_cli.flight_width:g} m with open sides."
             f" Fell before the switch: at most {before:.2f}.", "",
             "| command | " + " | ".join(f"{t} {h:g}" for t, h in cells) + " |", "|" + "---|" * (len(cells) + 1)]
    for text in commands:
        row = {(r["terrain"], r["height_cm"]): r for r in results if r["command"] == text}
        lines.append(f"| {text} | " + " | ".join(f"{row[c]['fell']:.2f}/{row[c]['followed']:.2f}" for c in cells) + " |")
    with open(out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
