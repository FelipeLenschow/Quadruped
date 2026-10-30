from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs.mdp import reset_root_state_uniform
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

from .terrain_info import tile_heading

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
    lying_joint_pos: dict[str, float] | list[dict[str, float]] | None = None,
    lying_blend: bool = False,
    lying_height: float = 0.16,
    drop_fraction: float = 0.0,
    drop_height_range: tuple[float, float] = (0.1, 0.3),
    drop_tilt: float = 0.3,
    joint_noise: float = 0.1,
    xy_range: float = 0.5,
    aligned_range: tuple[float, float, float] = (0.3, 0.15, 0.3),
) -> None:
    """Start a share of episodes lying on folded legs, and a share dropped from above standing height.
    lying_joint_pos may be a list of poses; each lying start picks one, and with lying_blend lands a random
    share of the way from the first pose to it.

    Runs after reset_base / reset_robot_joints and overwrites only the environments it picks, so
    the rest keep the nominal start. Root pose is rebuilt here from the default rather than read
    back from the sim: the write reset_base just made is not guaranteed to be visible in the data
    buffers before the next physics step.
    """

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
    heading = tile_heading(env, ids)
    aligned = ~torch.isnan(heading)
    if bool(aligned.any()):
        half = torch.tensor(aligned_range, device=device)
        noise = math_utils.sample_uniform(-half, half, (k, 3), device)
        pos[aligned, 0:2] = default_pose[aligned, 0:2] + origins[aligned, 0:2] + noise[aligned, 0:2]
        yaw[aligned] = heading[aligned] + noise[aligned, 2]
    quat = math_utils.quat_mul(default_pose[:, 3:7], math_utils.quat_from_euler_xyz(roll, pitch, yaw))
    asset.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], dim=-1), env_ids=ids)
    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(k, 6, device=device), env_ids=ids)

    joint_pos = asset.data.default_joint_pos.torch[ids].clone()
    if lying_joint_pos and bool(lying.any()):
        poses = lying_joint_pos if isinstance(lying_joint_pos, (list, tuple)) else [lying_joint_pos]
        rows = []
        for pose in poses:
            row = joint_pos[0].clone()
            for pattern, value in pose.items():
                row[asset.find_joints(pattern)[0]] = value
            rows.append(row)
        rows = torch.stack(rows)
        pick = torch.randint(len(rows), (int(lying.sum()),), device=device)
        lying_pos = rows[pick]
        if lying_blend:
            blend = torch.rand(len(pick), 1, device=device)
            lying_pos = rows[0] + blend * (lying_pos - rows[0])
        joint_pos[lying] = lying_pos
    joint_pos += math_utils.sample_uniform(-joint_noise, joint_noise, joint_pos.shape, device)
    limits = asset.data.soft_joint_pos_limits.torch[ids]
    joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
    asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=ids)
    asset.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos), env_ids=ids)


class reset_root_state_aligned(reset_root_state_uniform):
    """reset_root_state_uniform, but on tiles with a start heading (mdp.tile_heading) the yaw is that heading
    +- aligned_range[2] and the position the origin +- aligned_range[0:2] (x, y)."""

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        pose_range: dict[str, tuple[float, float]],
        velocity_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        aligned_range: tuple[float, float, float] = (0.3, 0.15, 0.3),
    ):
        asset = env.scene[asset_cfg.name]
        n = len(env_ids)
        default_root_pose = asset.data.default_root_pose.torch[env_ids]
        ranges = self._pose_ranges
        pose = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (n, 6), device=asset.device)
        heading = tile_heading(env, env_ids)
        aligned = ~torch.isnan(heading)
        if bool(aligned.any()):
            half = torch.tensor(aligned_range, device=asset.device)
            noise = math_utils.sample_uniform(-half, half, (n, 3), device=asset.device)
            pose[aligned, 0:2] = noise[aligned, 0:2]
            pose[aligned, 5] = heading[aligned] + noise[aligned, 2]
        positions = default_root_pose[:, 0:3] + env.scene.env_origins[env_ids] + pose[:, 0:3]
        orientations = math_utils.quat_mul(
            default_root_pose[:, 3:7], math_utils.quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5])
        )
        ranges = self._velocity_ranges
        velocities = asset.data.default_root_vel.torch[env_ids] + math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (n, 6), device=asset.device
        )
        asset.write_root_pose_to_sim_index(root_pose=torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim_index(root_velocity=velocities, env_ids=env_ids)
