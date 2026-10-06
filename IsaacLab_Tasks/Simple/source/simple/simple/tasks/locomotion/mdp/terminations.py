from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def illegal_contact_after(
    env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg, grace_s: float = 0.0
) -> torch.Tensor:
    """illegal_contact, ignored for the first grace_s of an episode.

    A robot that starts lying down, or lands a drop on its belly, is in base contact by design;
    terminating it there would teach nothing about getting up.
    """
    forces = env.scene.sensors[sensor_cfg.name].data.net_normal_forces_w_history.torch
    hit = torch.any(torch.max(torch.linalg.norm(forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1)
    return hit & (env.episode_length_buf * env.step_dt >= grace_s)


def physics_blowup(
    env: ManagerBasedRLEnv, max_joint_vel: float = 100.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Joint speeds far beyond any motor, or a non-finite state: the solver exploded, so reset the env."""
    data = env.scene[asset_cfg.name].data
    joint_vel = data.joint_vel.torch
    root = data.root_state_w.torch
    return ~(joint_vel.abs() <= max_joint_vel).all(dim=1) | ~torch.isfinite(root).all(dim=1)
