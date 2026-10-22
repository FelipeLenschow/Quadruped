from __future__ import annotations

import math
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


def tipped_on_legs(
    env: ManagerBasedRLEnv, side_limit: float, tilt_limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Fallen over: leaning sideways past side_limit rad, or tilted past tilt_limit rad from level in any
    direction. Pitching up onto two legs is allowed up to tilt_limit."""
    g = env.scene[asset_cfg.name].data.projected_gravity_b.torch
    tilt = torch.acos((-g[:, 2]).clamp(-1.0, 1.0))
    return (g[:, 1].abs() > math.sin(side_limit)) | (tilt > tilt_limit)
