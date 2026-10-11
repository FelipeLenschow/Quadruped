from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .rewards import lying_gate, standing_gate

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import CurriculumTermCfg


def difficulty(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> torch.Tensor:
    """Each env's level in [0, 1]; 1 everywhere when the curriculum is off."""
    level = getattr(env, "recovery_level", None)
    if level is None:
        return torch.ones(len(env_ids), device=env.device)
    return level[env_ids]


class recovery_level(ManagerTermBase):
    """Per-env difficulty: up when the robot ends the episode at its goal, down when it does not.
    The goal is standing, or with lie_joint_pos given, lying in that pose below max_height.

    Runs before the reset events, so it sees the final state and the start event reads the new level.
    Episodes shorter than min_s (the first ones, which rsl_rl starts at a random length) are not judged.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        env.recovery_level = torch.full((env.num_envs,), float(cfg.params.get("initial", 0.0)), device=env.device)
        self._start = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids,
        target_height: float,
        sensor_cfg: SceneEntityCfg,
        initial: float = 0.0,
        up: float = 0.05,
        down: float = 0.1,
        min_s: float = 8.0,
        lie_joint_pos: dict[str, float] | None = None,
        max_height: float = 0.1,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        if isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)
        judged = (env.common_step_counter - self._start[env_ids]) * env.step_dt >= min_s
        ids = env_ids[judged]
        if len(ids):
            if lie_joint_pos:
                gate = lying_gate(env, lie_joint_pos, max_height, sensor_cfg, asset_cfg)
            else:
                gate = standing_gate(env, target_height, sensor_cfg, asset_cfg)
            step = torch.where(gate[ids] > 0.5, up, -down)
            env.recovery_level[ids] = (env.recovery_level[ids] + step).clamp(0.0, 1.0)
        self._start[env_ids] = env.common_step_counter
        return env.recovery_level.mean()
