"""Per-start test of a Go2-LieDown checkpoint: how many end lying, how fast, and what hits the ground how hard.

Each robot goes through every start the task uses, one 10 s episode each, no pushes. A hit is a jump in a
body's contact force from one control step to the next; the feet are counted separately. The first 0.6 s
is reported apart from the rest, since it is mostly the start's own fall. At the end of each episode it
reports how much weight is on the belly (pads) and how many robots have the head on the ground.

    PYTHONPATH=source/recovery python scripts/rsl_rl/eval_liedown.py --checkpoint logs/rsl_rl/go2_liedown/<run>/model_<n>.pt
"""

import argparse
import os
import sys

os.environ.setdefault("RECOVERY_PLAY_CYCLE", "1")
os.environ.setdefault("RECOVERY_PLAY_PUSHES", "0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cli_args  # isort: skip

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--task", default="Go2-LieDown")
parser.add_argument("--num_envs", type=int, default=256)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import recovery.tasks  # noqa: F401
from recovery.tasks.locomotion import mdp
from recovery.tasks.locomotion.mdp.events import START_MODES
from recovery.tasks.locomotion.robots.go2 import recovery_env_cfg as rc
from recovery.utils.parser_cfg import parse_env_cfg

EARLY_S = 0.6
HARD = 300.0

cfg = parse_env_cfg(args.task, num_envs=args.num_envs, entry_point_key="play_env_cfg_entry_point")
cfg.scene.terrain.terrain_generator.num_rows = 10
cfg.scene.terrain.terrain_generator.num_cols = 20
agent = cli_args.parse_rsl_rl_cfg(args.task, args)
env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg), clip_actions=agent.clip_actions)
agent = cli_args.migrate_rsl_rl_cfg(agent)
runner = OnPolicyRunner(env, cli_args.runner_cfg_for_installed_rsl_rl(agent), log_dir=None, device=agent.device)
cli_args.load_checkpoint(runner, args.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

u = env.unwrapped
scan = mdp.SceneEntityCfg("height_scanner")
sensor = u.scene.sensors["contact_forces"]
body_ids = sensor.find_bodies(["base", "Head_.*", ".*_hip", ".*_thigh", ".*_calf"])[0]
foot_ids = sensor.find_bodies(".*_foot")[0]
head_ids = sensor.find_bodies("Head_.*")[0]
base_id = sensor.find_bodies("base")[0]
start = cfg.events.start.params
active = [m for m in START_MODES if start["fractions"].get(m, 0) or (start.get("easy_fractions") or {}).get(m, 0)]
steps = int(round(u.max_episode_length))

obs = env.get_observations()
print(f"[eval] {args.checkpoint}")
for _ in active:
    n, dev = u.num_envs, u.device
    first = torch.full((n,), float("nan"), device=dev)
    late_max = torch.zeros(n, device=dev)
    early_max = torch.zeros(n, device=dev)
    foot_max = torch.zeros(n, device=dev)
    last = None
    mode = None
    for t in range(steps - 1):
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        if mode is None:
            mode = active[int(u.recovery_start_count[0] - 1) % len(active)]
        f = sensor.data.net_normal_forces_w_history.torch.norm(dim=-1).amax(dim=1)
        if last is not None:
            jump = (f - last).clamp(min=0.0)
            body = jump[:, body_ids].amax(dim=1)
            if (t + 1) * u.step_dt > EARLY_S:
                late_max = torch.maximum(late_max, body)
            else:
                early_max = torch.maximum(early_max, body)
            foot_max = torch.maximum(foot_max, jump[:, foot_ids].amax(dim=1))
        last = f
        done = mdp.lying_gate(u, rc.LIE_POSE, rc.LIE_MAX_HEIGHT, scan) > 0.5
        first[done & torch.isnan(first)] = (t + 1) * u.step_dt
    times = first[~torch.isnan(first)]
    end = sensor.data.net_normal_forces_w_history.torch.norm(dim=-1).amax(dim=1)
    head_end, belly_end = end[:, head_ids].sum(dim=1), end[:, base_id].sum(dim=1)
    t_med = times.median().item() if len(times) else float("nan")
    print(
        f"[eval] {mode:9s} lying {done.float().mean() * 100:5.1f}%, median {t_med:.1f} s |"
        f" biggest non-foot hit after {EARLY_S} s: median {late_max.median():5.0f} N, over {HARD:.0f} N in"
        f" {(late_max > HARD).float().mean() * 100:3.0f}% | in the first {EARLY_S} s: median {early_max.median():5.0f} N |"
        f" biggest foot hit median {foot_max.median():5.0f} N | resting: belly {belly_end.mean():4.0f} N,"
        f" head touching in {(head_end > 1.0).float().mean() * 100:3.0f}%",
        flush=True,
    )
    with torch.inference_mode():
        obs, _, _, _ = env.step(policy(obs))
app.close()
