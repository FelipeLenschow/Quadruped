from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_plant(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    armature_range: tuple[float, float] | None = None,
    viscous_friction_range: tuple[float, float] | None = None,
) -> None:
    """Sample joint armature and viscous damping, once per environment.

    Isaac Lab's randomize_joint_parameters covers static friction and armature but not viscous
    damping, and the Go2 USD ships armature and damping at zero on every joint.
    """
    asset = env.scene[asset_cfg.name]
    device = asset.device
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device)
    joint_ids = asset_cfg.joint_ids
    num_joints = asset.num_joints if isinstance(joint_ids, slice) else len(joint_ids)
    shape = (len(env_ids), num_joints)

    if armature_range is not None:
        armature = torch.empty(shape, device=device).uniform_(*armature_range)
        asset.write_joint_armature_to_sim_index(armature=armature, joint_ids=joint_ids, env_ids=env_ids)

    if viscous_friction_range is not None:
        friction = asset.data.joint_friction_coeff
        friction = (friction.torch if hasattr(friction, "torch") else friction)[env_ids][:, joint_ids]
        viscous = torch.empty(shape, device=device).uniform_(*viscous_friction_range)
        asset.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=friction,
            joint_viscous_friction_coeff=viscous,
            joint_ids=joint_ids,
            env_ids=env_ids,
        )


def reset_trick_start(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_noise: float = 0.15,
    lin_vel: float = 0.3,
    ang_vel: float = 0.3,
    xy_range: float = 0.5,
    yaw_range: float = torch.pi,
) -> None:
    """Standing in the nominal pose, legs off it by up to joint_noise, still drifting at up to lin_vel and
    ang_vel: how the walking policy hands over when it stops."""
    from isaaclab.utils import math as math_utils

    asset = env.scene[asset_cfg.name]
    device = asset.device
    k = len(env_ids)
    if k == 0:
        return

    pose = asset.data.default_root_pose.torch[env_ids].clone()
    pose[:, 0:3] += env.scene.env_origins[env_ids]
    pose[:, 0:2] += math_utils.sample_uniform(-xy_range, xy_range, (k, 2), device)
    zeros = torch.zeros(k, device=device)
    yaw = math_utils.sample_uniform(-yaw_range, yaw_range, (k,), device)
    pose[:, 3:7] = math_utils.quat_mul(pose[:, 3:7], math_utils.quat_from_euler_xyz(zeros, zeros, yaw))
    velocity = torch.zeros(k, 6, device=device)
    velocity[:, 0:2] = math_utils.sample_uniform(-lin_vel, lin_vel, (k, 2), device)
    velocity[:, 5] = math_utils.sample_uniform(-ang_vel, ang_vel, (k,), device)
    asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)

    limits = asset.data.soft_joint_pos_limits.torch[env_ids]
    joint_pos = asset.data.default_joint_pos.torch[env_ids].clone()
    joint_pos += math_utils.sample_uniform(-joint_noise, joint_noise, joint_pos.shape, device)
    joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
    asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos), env_ids=env_ids)
