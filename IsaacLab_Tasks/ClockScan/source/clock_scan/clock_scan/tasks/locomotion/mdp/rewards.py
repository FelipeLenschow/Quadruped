from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

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


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
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
) -> torch.Tensor:
    """Squared error to a sin^2 swing profile, zero vertical speed at lift-off and touchdown, peaking at
    swing_height mid-swing. Clipped to max_error so a fallen robot is not paid to end the episode.
    Measured above the env origin, or with sensor_cfg above where each foot last stood. Feet above the
    profile cost above_scale times as much, so below 1 a foot may lift higher over an obstacle."""
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
