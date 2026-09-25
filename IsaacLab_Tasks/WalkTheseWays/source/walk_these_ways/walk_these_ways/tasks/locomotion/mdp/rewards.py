from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
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
    return torch.exp(-lin_vel_error / denominator)


def track_ang_vel_z_exp_scaled(
    env: ManagerBasedRLEnv,
    std: float,
    sigma_exp: float,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Yaw-rate tracking with the same command-scaled kernel. See track_lin_vel_xy_exp_scaled."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    ang_vel_error = torch.square(command[:, 2] - asset.data.root_ang_vel_b.torch[:, 2])
    command_rate = torch.abs(command[:, 2])
    denominator = torch.clamp(std**2 * command_rate**sigma_exp, min=0.005)
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


def foot_apex_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_height: float,
    sigma: float,
    bias: float,
    command_name: str = "base_velocity",
    moving_threshold: float = 0.02,
) -> torch.Tensor:
    """Walk's foot_height term (Final7): each swing is scored once, at touchdown, on the highest
    point it reached above the env's terrain origin: exp(-(apex - target)^2 / sigma) - bias.
    Standing still earns nothing, and a foot flailing in the air earns nothing until it lands."""
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_z = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    if getattr(env, "_foot_apex", None) is None:
        env._foot_apex = torch.zeros_like(foot_z)
    env._foot_apex[env.episode_length_buf <= 1] = 0.0
    env._foot_apex = torch.maximum(env._foot_apex, foot_z)
    landed = contact_sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids]
    match = torch.exp(-torch.square(env._foot_apex - target_height) / sigma)
    moving = torch.norm(env.command_manager.get_command(command_name), dim=1) > moving_threshold
    reward = torch.sum((match - bias) * landed, dim=1) * moving
    in_contact = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] > 0
    env._foot_apex = torch.where(in_contact, foot_z, env._foot_apex)
    return reward


def feet_air_time_target(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    target_slow: float,
    target_fast: float,
    speed_lo: float,
    speed_hi: float,
    sigma: float,
    static_threshold: float = 0.02,
    static_ramp: float = 0.1,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Walk's feet_air_time term. The swing-time target ramps from target_slow at xy command speed
    speed_lo to target_fast at speed_hi. Potential-based: every airborne step pays the change of
    phi = exp(-(air_time - target)^2 / sigma), so a swing sums to phi(landing) - phi(0) and the
    signal pushes toward landing at the target."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    airborne = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] <= 0
    command = env.command_manager.get_command(command_name)
    speed_xy = torch.norm(command[:, :2], dim=1)
    slow_frac = 1.0 - ((speed_xy - speed_lo) / max(speed_hi - speed_lo, 1e-6)).clamp(0.0, 1.0)
    target = (target_fast + (target_slow - target_fast) * slow_frac).unsqueeze(1)
    err = air_time - target
    phi = torch.exp(-torch.square(err) / sigma)
    dphi_dt = -2.0 * err / sigma * phi
    moving = ((torch.norm(command, dim=1) - static_threshold) / max(static_ramp - static_threshold, 1e-6)).clamp(0.0, 1.0)
    return torch.sum(dphi_dt * env.step_dt * airborne, dim=1) * moving


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time.torch[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Walk These Ways gait rewards. Feet are FL, FR, RL, RR, the gait command's order, so every
asset_cfg / sensor_cfg here lists them explicitly with preserve_order=True.
"""


def gait_contact_force(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, sigma: float = 100.0, command_name: str = "gait"
) -> torch.Tensor:
    """WTW tracking_contacts_shaped_force: negative, used with a positive weight."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = torch.norm(contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids], dim=-1)
    desired = env.command_manager.get_term(command_name).desired_contact
    return -torch.mean((1 - desired) * (1 - torch.exp(-torch.square(force) / sigma)), dim=1)


def gait_contact_vel(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sigma: float = 10.0, command_name: str = "gait"
) -> torch.Tensor:
    """WTW tracking_contacts_shaped_vel: negative, used with a positive weight."""
    asset: Articulation = env.scene[asset_cfg.name]
    speed = torch.norm(asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids], dim=-1)
    desired = env.command_manager.get_term(command_name).desired_contact
    return -torch.mean(desired * (1 - torch.exp(-torch.square(speed) / sigma)), dim=1)


def gait_swing_height(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, foot_radius: float = 0.02, command_name: str = "gait"
) -> torch.Tensor:
    """Squared error to a swing-height profile peaking at the commanded height mid-swing."""
    from .commands.gait_command import SWING

    asset: Articulation = env.scene[asset_cfg.name]
    term = env.command_manager.get_term(command_name)
    swing = 1 - torch.abs(1.0 - torch.clip(term.foot_phase * 2.0 - 1.0, 0.0, 1.0) * 2.0)
    target = term.command[:, SWING].unsqueeze(1) * swing + foot_radius
    foot_z = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    return torch.sum(torch.square(target - foot_z) * (1 - term.desired_contact), dim=1)


def gait_body_height(
    env: ManagerBasedRLEnv,
    nominal_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "gait",
) -> torch.Tensor:
    """WTW jump: negative squared error of base height above the scanned ground to nominal + commanded
    offset, used with a positive weight."""
    from .commands.gait_command import HEIGHT

    asset: Articulation = env.scene[asset_cfg.name]
    ground = torch.mean(env.scene.sensors[sensor_cfg.name].data.ray_hits_w.torch[..., 2], dim=1)
    height = asset.data.root_pos_w.torch[:, 2] - ground
    target = nominal_height + env.command_manager.get_term(command_name).command[:, HEIGHT]
    return -torch.square(height - target)


def gait_body_pitch(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), command_name: str = "gait"
) -> torch.Tensor:
    """Squared error of projected gravity to the commanded pitch with zero roll. Positive pitch is nose down."""
    from .commands.gait_command import PITCH

    asset: Articulation = env.scene[asset_cfg.name]
    pitch = env.command_manager.get_term(command_name).command[:, PITCH]
    desired = torch.stack([torch.sin(pitch), torch.zeros_like(pitch)], dim=1)
    return torch.sum(torch.square(asset.data.projected_gravity_b.torch[:, :2] - desired), dim=1)


def gait_raibert(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    stance_length: float,
    velocity_command_name: str = "base_velocity",
    command_name: str = "gait",
) -> torch.Tensor:
    """Walk These Ways' Raibert heuristic, with lateral velocity added: squared error of each foot's
    xy position in the yaw frame to its nominal stance point, shifted along the swing by velocity."""
    from isaaclab.utils.math import yaw_quat

    from .commands.gait_command import FREQ, WIDTH

    asset: Articulation = env.scene[asset_cfg.name]
    term = env.command_manager.get_term(command_name)
    cmd = term.command
    vel = env.command_manager.get_command(velocity_command_name)
    feet = asset.data.body_pos_w.torch[:, asset_cfg.body_ids] - asset.data.root_pos_w.torch.unsqueeze(1)
    yaw = yaw_quat(asset.data.root_quat_w.torch).unsqueeze(1).expand(-1, feet.shape[1], -1)
    feet_b = quat_apply_inverse(yaw.reshape(-1, 4), feet.reshape(-1, 3)).view_as(feet)

    half_w = cmd[:, WIDTH:WIDTH + 1] / 2
    ys = torch.cat([half_w, -half_w, half_w, -half_w], dim=1)
    xs = torch.tensor([1.0, 1.0, -1.0, -1.0], device=env.device).expand_as(ys) * (stance_length / 2)
    phases = torch.abs(1.0 - term.foot_phase * 2.0) - 0.5
    half_period = 0.5 / cmd[:, FREQ:FREQ + 1]
    front = torch.tensor([1.0, 1.0, -1.0, -1.0], device=env.device)
    y_vel = vel[:, 1:2] + vel[:, 2:3] * (stance_length / 2) * front
    xs = xs + phases * vel[:, 0:1] * half_period
    ys = ys + phases * y_vel * half_period
    return torch.sum(torch.square(xs - feet_b[..., 0]) + torch.square(ys - feet_b[..., 1]), dim=1)


def feet_slip(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """WTW feet_slip: squared xy speed of feet in contact now or on the last step."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, 2] > 1.0
    last = getattr(env, "_last_foot_contact", None)
    if last is None or last.shape != contact.shape:
        last = contact
    env._last_foot_contact = contact
    speed = torch.sum(torch.square(asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2]), dim=2)
    return torch.sum((contact | last) * speed, dim=1)


def action_smoothness_2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """WTW action_smoothness_2 on raw actions: squared second difference, skipped on the first two steps."""
    action = env.action_manager.action
    prev = env.action_manager.prev_action
    prev2 = getattr(env, "_prev_prev_action", None)
    if prev2 is None or prev2.shape != action.shape:
        prev2 = prev.clone()
    started = (env.episode_length_buf > 2).float()
    value = torch.sum(torch.square(action - 2 * prev + prev2), dim=1) * started
    env._prev_prev_action = prev.clone()
    return value
