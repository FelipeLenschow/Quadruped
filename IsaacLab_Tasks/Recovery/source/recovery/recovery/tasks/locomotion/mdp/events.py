from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

from .curriculums import difficulty

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

START_MODES = ("stand", "moving", "drop", "lying", "flipped", "tumble", "handstand", "hindstand", "tipping")


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


def reset_recovery_start(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    fractions: dict[str, float],
    lying_poses: list[dict[str, float]],
    easy_fractions: dict[str, float] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lying_height: float = 0.16,
    drop_height_range: tuple[float, float] = (0.1, 0.5),
    drop_tilt: float = 0.6,
    flipped_height: float = 0.5,
    tumble_height_range: tuple[float, float] = (0.6, 1.0),
    tumble_lin_vel: float = 1.0,
    tumble_ang_vel: float = 4.0,
    moving_lin_vel: float = 1.0,
    moving_ang_vel: float = 1.0,
    moving_joint_noise: float = 0.25,
    stand_on_legs_tilt: tuple[float, float] = (1.05, 1.75),
    tipping_ang_vel: tuple[float, float] = (1.5, 4.0),
    joint_noise: float = 0.1,
    xy_range: float = 0.5,
    cycle: bool = False,
) -> None:
    """Every episode starts in one of START_MODES, drawn with the given weights:

    stand    nominal pose on the ground
    moving   nominal pose, walking and turning (up to the moving_* speeds), legs off nominal
    drop     nominal joints, dropped from above standing height with roll and pitch
    lying    on its belly in one of lying_poses
    flipped  on its side or back with random joints, a little above the ground
    tumble   any orientation, random joints, thrown from up high with linear and angular velocity
    handstand  nose down on straight front legs, tilted stand_on_legs_tilt (rad) from level
    hindstand  tail down on straight rear legs, same tilt
    tipping  nominal pose, moving like `moving`, and already rolling or pitching over at tipping_ang_vel

    With the curriculum on, each env's level moves the weights from easy_fractions to fractions and
    scales the drop height and tilt and the throw height and speed from their lower bound up to full.
    With cycle, each env goes through the START_MODES that have a weight, in order, one per episode.
    """
    from isaaclab.utils import math as math_utils

    asset = env.scene[asset_cfg.name]
    device = asset.device
    k = len(env_ids)
    if k == 0:
        return

    level = difficulty(env, env_ids)
    hard = torch.tensor([fractions.get(m, 0.0) for m in START_MODES], device=device)
    easy = hard if easy_fractions is None else torch.tensor([easy_fractions.get(m, 0.0) for m in START_MODES], device=device)
    weights = torch.lerp(easy / easy.sum(), hard / hard.sum(), level.unsqueeze(1))
    mode = torch.multinomial(weights, 1).squeeze(1)
    if cycle:
        if not hasattr(env, "recovery_start_count"):
            env.recovery_start_count = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        active = ((hard > 0) | (easy > 0)).nonzero().flatten()
        mode = active[env.recovery_start_count[env_ids] % len(active)]
        env.recovery_start_count[env_ids] += 1
    stand, moving, drop, lying, flipped, tumble, handstand, hindstand, tipping = (mode == i for i in range(len(START_MODES)))
    on_legs = handstand | hindstand

    default_pose = asset.data.default_root_pose.torch[env_ids]
    origins = env.scene.env_origins[env_ids]
    pos = default_pose[:, 0:3] + origins
    pos[:, 0:2] += math_utils.sample_uniform(-xy_range, xy_range, (k, 2), device)
    ground = origins[:, 2]
    def scaled(lo, hi):
        return lo + (hi - lo) * level * torch.rand(k, device=device)

    pos[:, 2] = torch.where(drop, pos[:, 2] + scaled(*drop_height_range), pos[:, 2])
    pos[:, 2] = torch.where(lying, ground + lying_height, pos[:, 2])
    pos[:, 2] = torch.where(flipped, ground + flipped_height, pos[:, 2])
    pos[:, 2] = torch.where(tumble, ground + scaled(*tumble_height_range), pos[:, 2])
    # Hip 0.19 m from the base along the body, then a straight leg of ~0.375 m along it too.
    tilt = scaled(*stand_on_legs_tilt)
    pos[:, 2] = torch.where(on_legs, ground + 0.566 * torch.sin(tilt).clamp(min=0.3) + 0.03, pos[:, 2])

    zeros = torch.zeros(k, device=device)
    roll = torch.where(drop, math_utils.sample_uniform(-1.0, 1.0, (k,), device) * drop_tilt * level, zeros)
    pitch = torch.where(drop, math_utils.sample_uniform(-1.0, 1.0, (k,), device) * drop_tilt * level, zeros)
    side = torch.tensor([0.5 * torch.pi, -0.5 * torch.pi, torch.pi], device=device)[torch.randint(3, (k,), device=device)]
    roll = torch.where(flipped, side + math_utils.sample_uniform(-0.3, 0.3, (k,), device), roll)
    pitch = torch.where(flipped, math_utils.sample_uniform(-0.3, 0.3, (k,), device), pitch)
    pitch = torch.where(handstand, tilt, torch.where(hindstand, -tilt, pitch))
    yaw = math_utils.sample_uniform(-torch.pi, torch.pi, (k,), device)
    quat = math_utils.quat_mul(default_pose[:, 3:7], math_utils.quat_from_euler_xyz(roll, pitch, yaw))
    quat[tumble] = math_utils.random_orientation(int(tumble.sum()), device)

    velocity = math_utils.sample_uniform(-1.0, 1.0, (k, 6), device) * level.unsqueeze(1)
    velocity[:, 0:3] *= torch.where(tumble, tumble_lin_vel, moving_lin_vel).unsqueeze(1)
    velocity[:, 3:6] *= torch.where(tumble, tumble_ang_vel, moving_ang_vel).unsqueeze(1)
    velocity[moving, 2:5] = 0.0
    velocity[tipping, 0:2] *= moving_lin_vel
    velocity[tipping, 2:6] = 0.0
    heading = math_utils.sample_uniform(-torch.pi, torch.pi, (k,), device)
    spin = scaled(*tipping_ang_vel)
    velocity[tipping, 3] = (spin * torch.cos(heading))[tipping]
    velocity[tipping, 4] = (spin * torch.sin(heading))[tipping]
    velocity[~(tumble | moving | tipping)] = 0.0
    asset.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)

    limits = asset.data.soft_joint_pos_limits.torch[env_ids]
    joint_pos = asset.data.default_joint_pos.torch[env_ids].clone()
    if lying_poses and bool(lying.any()):
        rows = []
        for pose in lying_poses:
            row = joint_pos[0].clone()
            for pattern, value in pose.items():
                row[asset.find_joints(pattern)[0]] = value
            rows.append(row)
        joint_pos[lying] = torch.stack(rows)[torch.randint(len(rows), (int(lying.sum()),), device=device)]
    front, rear = asset.find_joints("F[L,R]_.*_joint")[0], asset.find_joints("R[L,R]_.*_joint")[0]
    straight_front = torch.tensor([0.0, -1.3, -0.95], device=device)
    straight_rear = torch.tensor([0.0, 2.2, -0.95], device=device)
    for ids, leg in ((front, straight_front), (rear, straight_rear)):
        names = [asset.joint_names[i] for i in ids]
        target = torch.stack([leg[0 if "hip" in n else 1 if "thigh" in n else 2] for n in names])
        rows = handstand if leg is straight_front else hindstand
        joint_pos[rows.nonzero().flatten().unsqueeze(1), torch.tensor(ids, device=device)] = target
    wild = flipped | tumble
    joint_pos[wild] = math_utils.sample_uniform(limits[wild][..., 0], limits[wild][..., 1], limits[wild][..., 0].shape, device)
    noise = torch.where(moving, moving_joint_noise, joint_noise).unsqueeze(1)
    joint_pos += math_utils.sample_uniform(-1.0, 1.0, joint_pos.shape, device) * noise
    joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
    asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos), env_ids=env_ids)
