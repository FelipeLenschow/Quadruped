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
    on_legs_fraction: float = 0.0,
    on_legs_tilt: tuple[float, float] = (1.0, 1.5),
) -> None:
    """Standing in the nominal pose, legs off it by up to joint_noise, still drifting at up to lin_vel and
    ang_vel: how the walking policy hands over when it stops.

    A share on_legs_fraction starts as LieDown's handstand and hindstand do instead, at rest: nose down on
    straight front legs (switch 0) or tail down on straight rear legs (switch 1), on_legs_tilt rad from
    level. env.trick_start_switch keeps which, -1 for standing, for the trick command's first segment."""
    from isaaclab.utils import math as math_utils

    asset = env.scene[asset_cfg.name]
    device = asset.device
    k = len(env_ids)
    if k == 0:
        return

    on_legs = torch.rand(k, device=device) < on_legs_fraction
    hind = on_legs & (torch.rand(k, device=device) < 0.5)
    hand = on_legs & ~hind
    if not hasattr(env, "trick_start_switch"):
        env.trick_start_switch = torch.full((env.num_envs,), -1, dtype=torch.long, device=device)
    env.trick_start_switch[env_ids] = torch.where(hand, 0, torch.where(hind, 1, -1))

    pose = asset.data.default_root_pose.torch[env_ids].clone()
    pose[:, 0:3] += env.scene.env_origins[env_ids]
    pose[:, 0:2] += math_utils.sample_uniform(-xy_range, xy_range, (k, 2), device)
    # Hip 0.19 m from the base along the body, then a straight leg of ~0.375 m along it too.
    tilt = math_utils.sample_uniform(*on_legs_tilt, (k,), device)
    ground = env.scene.env_origins[env_ids, 2]
    pose[:, 2] = torch.where(on_legs, ground + 0.566 * torch.sin(tilt).clamp(min=0.3) + 0.03, pose[:, 2])
    zeros = torch.zeros(k, device=device)
    pitch = torch.where(hand, tilt, torch.where(hind, -tilt, zeros))
    yaw = math_utils.sample_uniform(-yaw_range, yaw_range, (k,), device)
    pose[:, 3:7] = math_utils.quat_mul(pose[:, 3:7], math_utils.quat_from_euler_xyz(zeros, pitch, yaw))
    velocity = torch.zeros(k, 6, device=device)
    velocity[:, 0:2] = math_utils.sample_uniform(-lin_vel, lin_vel, (k, 2), device)
    velocity[:, 5] = math_utils.sample_uniform(-ang_vel, ang_vel, (k,), device)
    velocity[on_legs] = 0.0
    asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)

    limits = asset.data.soft_joint_pos_limits.torch[env_ids]
    joint_pos = asset.data.default_joint_pos.torch[env_ids].clone()
    front, rear = asset.find_joints("F[L,R]_.*_joint")[0], asset.find_joints("R[L,R]_.*_joint")[0]
    for ids, leg, rows in ((front, (0.0, -1.3, -0.95), hand), (rear, (0.0, 2.2, -0.95), hind)):
        target = torch.tensor(
            [leg[0 if "hip" in n else 1 if "thigh" in n else 2] for n in (asset.joint_names[i] for i in ids)], device=device
        )
        joint_pos[rows.nonzero().flatten().unsqueeze(1), torch.tensor(ids, device=device)] = target
    joint_pos += math_utils.sample_uniform(-joint_noise, joint_noise, joint_pos.shape, device)
    joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
    asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos), env_ids=env_ids)
