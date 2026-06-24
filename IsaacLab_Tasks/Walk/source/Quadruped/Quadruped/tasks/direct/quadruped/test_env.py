import os
import torch
import sys

# Add IsaacLab to path or assume it works
from isaaclab.app import AppLauncher

launcher = AppLauncher(headless=True)
app = launcher.app

import isaaclab.envs.mdp as mdp
from isaaclab.envs import DirectRLEnvCfg

sys.path.append("/home/05680435969@corp.udesc.br/Quadruped/IsaacLab_Tasks/Walk/source/Quadruped")

from Quadruped.tasks.direct.quadruped.quadruped_env import QuadrupedEnv
from Quadruped.tasks.direct.quadruped.quadruped_env_cfg import QuadrupedEnvCfg

env_cfg = QuadrupedEnvCfg()
env_cfg.scene.num_envs = 2  # small test
env = QuadrupedEnv(env_cfg)

print("Environment created.")

# Step with 0 action
actions = torch.zeros((2, 12), device=env.device)
obs, rew, reset, _ = env.step(actions)

print(f"Base Height initially: {env.root_pos_w[:, 2]}")

for _ in range(50):
    # Action of 0.5 (should move joints significantly)
    actions = torch.ones((2, 12), device=env.device) * 0.5
    obs, rew, reset, _ = env.step(actions)

print(f"Base Height after 50 steps: {env.root_pos_w[:, 2]}")
print(f"Joint positions: {env.joint_pos[0]}")
print(f"Desired Joint Pos: {env.desired_joint_pos[0]}")
print(f"Backlash state: {env.backlash_state[0]}")
print(f"Last targets: {env.last_targets[0]}")

app.close()
