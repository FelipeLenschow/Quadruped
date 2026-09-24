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


def reset_start_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lying_fraction: float = 0.0,
    lying_joint_pos: dict[str, float] | None = None,
    lying_height: float = 0.16,
    drop_fraction: float = 0.0,
    drop_height_range: tuple[float, float] = (0.1, 0.3),
    drop_tilt: float = 0.3,
    joint_noise: float = 0.1,
    xy_range: float = 0.5,
) -> None:
    """Start a share of episodes lying on folded legs, and a share dropped from above standing height.

    Runs after reset_base / reset_robot_joints and overwrites only the environments it picks, so
    the rest keep the nominal start. Root pose is rebuilt here from the default rather than read
    back from the sim: the write reset_base just made is not guaranteed to be visible in the data
    buffers before the next physics step.
    """
    from isaaclab.utils import math as math_utils

    asset = env.scene[asset_cfg.name]
    device = asset.device
    n = len(env_ids)
    if n == 0 or (lying_fraction <= 0.0 and drop_fraction <= 0.0):
        return
    draw = torch.rand(n, device=device)
    lying = draw < lying_fraction
    drop = (draw >= lying_fraction) & (draw < lying_fraction + drop_fraction)
    picked = lying | drop
    if not bool(picked.any()):
        return
    ids = env_ids[picked]
    lying = lying[picked]
    k = len(ids)

    default_pose = asset.data.default_root_pose.torch[ids]
    origins = env.scene.env_origins[ids]
    pos = default_pose[:, 0:3] + origins
    pos[:, 0:2] += math_utils.sample_uniform(-xy_range, xy_range, (k, 2), device)
    drop_z = math_utils.sample_uniform(*drop_height_range, (k,), device)
    pos[:, 2] = torch.where(lying, origins[:, 2] + lying_height, pos[:, 2] + drop_z)

    tilt = torch.where(lying, torch.zeros(k, device=device), torch.full((k,), drop_tilt, device=device))
    roll = math_utils.sample_uniform(-1.0, 1.0, (k,), device) * tilt
    pitch = math_utils.sample_uniform(-1.0, 1.0, (k,), device) * tilt
    yaw = math_utils.sample_uniform(-torch.pi, torch.pi, (k,), device)
    quat = math_utils.quat_mul(default_pose[:, 3:7], math_utils.quat_from_euler_xyz(roll, pitch, yaw))
    asset.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], dim=-1), env_ids=ids)
    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(k, 6, device=device), env_ids=ids)

    joint_pos = asset.data.default_joint_pos.torch[ids].clone()
    if lying_joint_pos:
        lying_row = joint_pos[0].clone()
        for pattern, value in lying_joint_pos.items():
            lying_row[asset.find_joints(pattern)[0]] = value
        joint_pos[lying] = lying_row
    joint_pos += math_utils.sample_uniform(-joint_noise, joint_noise, joint_pos.shape, device)
    limits = asset.data.soft_joint_pos_limits.torch[ids]
    joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
    asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=ids)
    asset.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos), env_ids=ids)
