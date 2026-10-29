from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque.torch[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


"""
Velocity tracking with a command-scaled kernel.
"""


def track_lin_vel_xy_exp_scaled(
    env: ManagerBasedRLEnv,
    std: float,
    sigma_exp: float,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    grace_s: float = 0.0,
    grace_floor: float = 0.25,
) -> torch.Tensor:
    """Linear-velocity tracking whose tolerance scales with the commanded speed.

    Upstream track_lin_vel_xy_exp uses a FIXED width: exp(-error / std^2). That width is what
    lets a stationary robot collect most of the reward under a slow command -- at std^2 = 0.25
    and a 0.2 m/s command, standing still scores exp(-0.04/0.25) = 85% of the maximum. The
    policy is not failing to track; not tracking is nearly free.

    Here the denominator is  std^2 * ||c_xy||^sigma_exp, so the robot is held to a tolerance
    proportional to what it was asked to do:

        sigma_exp = 0   identical to upstream (x^0 = 1), the fixed absolute tolerance.
        sigma_exp = 1   tolerance grows linearly with speed.
        sigma_exp = 2   EXACTLY scale-invariant: the reward depends only on the RELATIVE
                        error v/c, so a 0.05 m/s command is as demanding as a 1.0 m/s one.

    The clamp keeps a zero command finite. It bites below ||c_xy|| = sqrt(0.005/std^2), which
    at std^2 = 0.25 is 0.14 m/s for sigma_exp = 2 -- inside the band this reward is meant to
    fix, so it does soften the very slowest commands. Kept anyway, and kept at 0.005, because
    that is the value the Direct-env implementation this mirrors uses; the two must agree for
    their numbers to be comparable.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    lin_vel_error = torch.sum(torch.square(command[:, :2] - asset.data.root_lin_vel_b.torch[:, :2]), dim=1)
    command_speed = torch.norm(command[:, :2], dim=1)
    denominator = torch.clamp(std**2 * command_speed**sigma_exp, min=0.005)
    if grace_s > 0.0:
        fade = (1.0 - env.command_manager.get_term(command_name).transition_age / grace_s).clamp(0.0, 1.0)
        denominator = torch.maximum(denominator, fade * grace_floor)
    return torch.exp(-lin_vel_error / denominator)


def track_ang_vel_z_exp_scaled(
    env: ManagerBasedRLEnv,
    std: float,
    sigma_exp: float,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    grace_s: float = 0.0,
    grace_floor: float = 0.25,
) -> torch.Tensor:
    """Yaw-rate tracking with the same command-scaled kernel. See track_lin_vel_xy_exp_scaled.

    With grace_s set, both kernels are widened to at least grace_floor right after a switch between
    standing and moving, fading back over grace_s, so a start or stop may take a step instead of the
    robot braking hard to match a zero command at once."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    ang_vel_error = torch.square(command[:, 2] - asset.data.root_ang_vel_b.torch[:, 2])
    command_rate = torch.abs(command[:, 2])
    denominator = torch.clamp(std**2 * command_rate**sigma_exp, min=0.005)
    if grace_s > 0.0:
        fade = (1.0 - env.command_manager.get_term(command_name).transition_age / grace_s).clamp(0.0, 1.0)
        denominator = torch.maximum(denominator, fade * grace_floor)
    return torch.exp(-ang_vel_error / denominator)


"""
Robot.
"""


def joint_position_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    moving_scale: float = 1.0,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b.torch[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos.torch - asset.data.default_joint_pos.torch), dim=1)
    return torch.where(
        torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), moving_scale * reward, stand_still_scale * reward
    )


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, ratio: float = 4.0) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > ratio * forces_z, dim=1).float()
    return reward


"""
Clock rewards. Feet in FL, FR, RL, RR order, the clock's, so the cfgs use preserve_order=True.
"""


def clock_contact_force(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, sigma: float = 100.0, command_name: str = "clock"
) -> torch.Tensor:
    """Walk These Ways' contact force term: force on feet the clock has in swing. Negative."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = torch.norm(contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids], dim=-1)
    desired = env.command_manager.get_term(command_name).desired_contact
    return -torch.mean((1 - desired) * (1 - torch.exp(-torch.square(force) / sigma)), dim=1)


def clock_contact_vel(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sigma: float = 10.0, command_name: str = "clock"
) -> torch.Tensor:
    """Walk These Ways' contact velocity term: speed of feet the clock has in stance. Negative."""
    asset: Articulation = env.scene[asset_cfg.name]
    speed = torch.norm(asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids], dim=-1)
    desired = env.command_manager.get_term(command_name).desired_contact
    return -torch.mean(desired * (1 - torch.exp(-torch.square(speed) / sigma)), dim=1)


def clock_swing_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    swing_height: float,
    foot_radius: float = 0.02,
    max_error: float = 0.1,
    command_name: str = "clock",
    sensor_cfg: SceneEntityCfg | None = None,
    above_scale: float = 1.0,
    ground_sensors: list[str] | None = None,
) -> torch.Tensor:
    """Squared error to a sin^2 swing profile, zero vertical speed at lift-off and touchdown, peaking at
    swing_height mid-swing. Clipped to max_error so a fallen robot is not paid to end the episode.
    Measured above the env origin, with sensor_cfg above where each foot last stood, or with ground_sensors
    (one downward ray caster per foot, in asset_cfg's order) above the ground right under each foot, so the
    profile follows steps up and down. Feet above the profile cost above_scale times as much, so below 1 a
    foot may lift higher over an obstacle."""
    asset: Articulation = env.scene[asset_cfg.name]
    term = env.command_manager.get_term(command_name)
    swing = torch.square(torch.sin(math.pi * torch.clip(term.foot_phase * 2.0 - 1.0, 0.0, 1.0)))
    target = swing_height * swing + foot_radius
    foot_z = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
    ground = env.scene.env_origins[:, 2].unsqueeze(1).expand_as(foot_z)
    if sensor_cfg is not None:
        stood = getattr(env, "_foot_ground_z", None)
        if stood is None or stood.shape != foot_z.shape:
            stood = ground.clone()
        stood = torch.where((env.episode_length_buf <= 1).unsqueeze(1), ground, stood)
        forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w.torch[:, sensor_cfg.body_ids]
        stood = torch.where(torch.norm(forces, dim=-1) > 1.0, foot_z - foot_radius, stood)
        env._foot_ground_z = stood
        ground = stood
    if ground_sensors is not None:
        hits = torch.stack([env.scene.sensors[name].data.ray_hits_w.torch[:, 0, 2] for name in ground_sensors], dim=1)
        ground = torch.where(torch.isfinite(hits), hits, ground)
    error = (target - (foot_z - ground)).clamp(-max_error, max_error)
    error = torch.where(error < 0, error * math.sqrt(above_scale), error)
    return torch.sum(torch.square(error) * (1 - term.desired_contact), dim=1)


def foot_landing_velocity(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, max_speed: float = 3.0
) -> torch.Tensor:
    """Squared downward speed of each foot on the step before its first contact, summed over the feet
    that touched down this step."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    vz = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, 2]
    prev = getattr(env, "_foot_landing_vz", None)
    if prev is None or prev.shape != vz.shape:
        prev = torch.zeros_like(vz)
    landed = contact_sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids]
    env._foot_landing_vz = vz.clone()
    return torch.sum(torch.square((-prev).clamp(0.0, max_speed)) * landed, dim=1)

def base_height_moving(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    max_error: float = 0.05,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared base height error above the scan, clipped to max_error and zero while the command is zero,
    so a robot lying down or getting up is not paid to end the episode instead."""
    asset: RigidObject = env.scene[asset_cfg.name]
    ground = env.scene.sensors[sensor_cfg.name].data.ray_hits_w.torch[..., 2].mean(dim=1)
    error = (asset.data.root_pos_w.torch[:, 2] - ground - target_height).clamp(-max_error, max_error)
    moving = torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.0
    return torch.square(error) * moving


def terminated_by_terrain(env: ManagerBasedRLEnv, scales: dict[str, float]) -> torch.Tensor:
    """is_terminated times a scale per terrain type (env._terrain_kind, set by mdp.terrain_stages); 1 elsewhere."""
    terminated = env.termination_manager.terminated.float()
    kind = getattr(env, "_terrain_kind", None)
    if kind is None:
        return terminated
    table = getattr(env, "_fall_scale_table", None)
    if table is None:
        table = torch.tensor([scales.get(name, 1.0) for name in env._terrain_kind_names], device=env.device)
        env._fall_scale_table = table
    return terminated * table[kind]


def leash_position_l1(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """Distance from the base to the command's leash reference, m, not capped."""
    return torch.linalg.norm(env.command_manager.get_term(command_name).leash_pos_error_w, dim=1)


def leash_yaw_l1(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """Heading error to the leash reference, rad, in [0, pi]."""
    return env.command_manager.get_term(command_name).leash_yaw_error.abs()


def terrain_scaled(env: ManagerBasedRLEnv, term, term_params: dict, scales: dict[str, float]) -> torch.Tensor:
    """term(env, **term_params) times a scale per terrain type (env._terrain_kind, set by mdp.terrain_stages)."""
    value = term(env, **term_params)
    kind = getattr(env, "_terrain_kind", None)
    if kind is None:
        return value
    table = torch.tensor([scales.get(name, 1.0) for name in env._terrain_kind_names], device=env.device)
    return value * table[kind]


def ground_normal_b(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_slope: float = 0.8, asset_cfg=None) -> torch.Tensor:
    """Unit normal, in the base frame, of a plane fitted to the yaw-turned vertical height grid of sensor_cfg, its
    slope clipped to max_slope. Rays with no hit take the mean of the others."""
    asset: RigidObject = env.scene["robot" if asset_cfg is None else asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    cache = env.__dict__.setdefault("_ground_plane_fit", {})
    if sensor_cfg.name not in cache:
        pattern = sensor.cfg.pattern_cfg
        xy = pattern.func(pattern, env.device)[0][:, :2] + torch.tensor(sensor.cfg.offset.pos[:2], device=env.device)
        cache[sensor_cfg.name] = torch.linalg.pinv(torch.cat([torch.ones(len(xy), 1, device=env.device), xy], dim=1))
    z = sensor.data.ray_hits_w.torch[..., 2]
    finite = torch.isfinite(z)
    fill = torch.where(finite, z, torch.zeros_like(z)).sum(dim=1, keepdim=True) / finite.sum(dim=1, keepdim=True).clamp(min=1)
    z = torch.where(finite, z, fill)
    slope = (z @ cache[sensor_cfg.name].T)[:, 1:].clamp(-max_slope, max_slope)
    quat = asset.data.root_quat_w.torch
    normal = torch.cat([-slope, torch.ones_like(slope[:, :1])], dim=1)
    normal = quat_apply(yaw_quat(quat), normal / torch.linalg.norm(normal, dim=1, keepdim=True))
    return quat_apply_inverse(quat, normal)


def ground_orientation_l2(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_slope: float = 0.8) -> torch.Tensor:
    """flat_orientation_l2 against the ground instead of gravity: the squared x and y of the ground normal
    (ground_normal_b) in the base frame, so on stairs the body may pitch with the steps and on flat ground nothing
    changes."""
    return torch.sum(torch.square(ground_normal_b(env, sensor_cfg, max_slope)[:, :2]), dim=1)


def head_clearance_l1(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    min_clearance: float,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """How far the body of asset_cfg (the head) is below min_clearance above the ground straight under it (one
    downward ray, sensor_cfg), m, zero while the command is zero, so getting up from lying is not paid for."""
    from .observations import head_clearance

    gap = head_clearance(env, sensor_cfg, asset_cfg).squeeze(1)
    moving = torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.0
    return (min_clearance - gap).clamp(min=0.0) * moving
