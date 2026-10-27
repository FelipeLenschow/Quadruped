from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def physics_blowup(
    env: ManagerBasedRLEnv, max_joint_vel: float = 100.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Joint speeds far beyond any motor, or a non-finite state: the solver exploded, so reset the env."""
    data = env.scene[asset_cfg.name].data
    joint_vel = data.joint_vel.torch
    root = data.root_state_w.torch
    return ~(joint_vel.abs() <= max_joint_vel).all(dim=1) | ~torch.isfinite(root).all(dim=1)


def tipped_standing(
    env: ManagerBasedRLEnv, limit_angle: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Tilted past limit_angle rad from level, outside the flip window."""
    g = env.scene[asset_cfg.name].data.projected_gravity_b.torch
    tilt = torch.acos((-g[:, 2]).clamp(-1.0, 1.0))
    return (tilt > limit_angle) & (env.command_manager.get_term(command_name).flipping < 0.5)
