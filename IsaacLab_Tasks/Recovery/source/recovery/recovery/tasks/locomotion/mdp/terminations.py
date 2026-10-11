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
