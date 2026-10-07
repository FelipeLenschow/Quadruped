from __future__ import annotations

import math
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


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos.torch - asset.data.default_joint_pos.torch), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


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


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b.torch * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b.torch[:, 2])
    return reward


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


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, :] - asset.data.root_pos_w.torch[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w.torch[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w.torch, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w.torch, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b.torch[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


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


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


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
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos.torch[:, joint_pair[0][0]] - asset.data.joint_pos.torch[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
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


def forward_progress(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Share of the commanded forward speed reached, clip(v_x / cmd_x, -1, 1): linear from standing to
    the target, where the exp tracking kernel is flat. Zero for commands under 0.1 m/s."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, 0]
    ratio = asset.data.root_lin_vel_b.torch[:, 0] / cmd.clamp(min=0.1)
    return ratio.clamp(-1.0, 1.0) * (cmd > 0.1)


def forward_speed(
    env: ManagerBasedRLEnv, max_speed: float = 15.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Forward speed along the robot's heading (yaw frame, so pitching does not change it), m/s,
    clipped to [-1, max_speed]."""
    from isaaclab.utils.math import yaw_quat

    asset: Articulation = env.scene[asset_cfg.name]
    vel = quat_apply_inverse(yaw_quat(asset.data.root_quat_w.torch), asset.data.root_lin_vel_w.torch)
    return vel[:, 0].clamp(-1.0, max_speed)


def sprint_speed(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    max_speed: float = 15.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """With a forward command: forward speed along the heading, clipped to [-1, max_speed]. Without
    one: minus the planar speed, so the robot is paid to stop and stay stopped."""
    from isaaclab.utils.math import yaw_quat

    asset: Articulation = env.scene[asset_cfg.name]
    vel = quat_apply_inverse(yaw_quat(asset.data.root_quat_w.torch), asset.data.root_lin_vel_w.torch)
    run = env.command_manager.get_command(command_name)[:, 0] > 0.0
    return torch.where(run, vel[:, 0].clamp(-1.0, max_speed), -torch.norm(vel[:, :2], dim=1))



def base_height_l2_ground(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    max_error: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared base height error above the mean of the scan rays that hit, clipped to max_error."""
    hits = env.scene.sensors[sensor_cfg.name].data.ray_hits_w.torch[..., 2]
    ground = torch.where(torch.isfinite(hits), hits, torch.nan).nanmean(dim=1).nan_to_num(0.0)
    error = env.scene[asset_cfg.name].data.root_pos_w.torch[:, 2] - ground - target_height
    return torch.square(error.clamp(-max_error, max_error))


def joint_vel_over(
    env: ManagerBasedRLEnv, limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Sum of squared joint speed above limit."""
    speed = env.scene[asset_cfg.name].data.joint_vel.torch[:, asset_cfg.joint_ids].abs()
    return torch.sum(torch.square((speed - limit).clamp(min=0.0)), dim=1)

def phase_gated(env: ManagerBasedRLEnv, term, getup_s: float, getup: bool, term_params: dict) -> torch.Tensor:
    """term(env, **term_params), kept only during the first getup_s of the episode (getup) or after it."""
    value = term(env, **term_params)
    during = env.episode_length_buf * env.step_dt < getup_s
    return value * (during if getup else ~during)
