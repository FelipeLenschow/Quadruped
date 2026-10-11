from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _base_height(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Base height above the mean of the rays that hit; the env origin if none did."""
    z = env.scene.sensors[sensor_cfg.name].data.ray_hits_w.torch[..., 2]
    hit = torch.isfinite(z)
    count = hit.sum(dim=1)
    ground = torch.where(hit, z, 0.0).sum(dim=1) / count.clamp(min=1)
    ground = torch.where(count > 0, ground, env.scene.env_origins[:, 2])
    return env.scene[asset_cfg.name].data.root_pos_w.torch[:, 2] - ground


def _up(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Cosine of the base tilt: 1 upright, 0 on its side, -1 on its back."""
    return -env.scene[asset_cfg.name].data.projected_gravity_b.torch[:, 2]


def standing_gate(
    env: ManagerBasedRLEnv, target_height: float, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """0 until the base is within ~37 deg of level and above 0.6 x target height, 1 once within ~25 deg and 0.9 x."""
    up = ((_up(env, asset_cfg) - 0.8) / 0.1).clamp(0.0, 1.0)
    h = ((_base_height(env, sensor_cfg, asset_cfg) / target_height - 0.6) / 0.3).clamp(0.0, 1.0)
    return up * h


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    qvel = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque.torch[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def upright(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """1 level, 0.25 on its side, 0 on its back."""
    return (0.5 * (1.0 + _up(env, asset_cfg))).square()


def base_height_up(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Base height as a share of the target, counted only while the belly faces the ground."""
    h = (_base_height(env, sensor_cfg, asset_cfg) / target_height).clamp(0.0, 1.0)
    return _up(env, asset_cfg).clamp(min=0.0) * h


def stand_pose(
    env: ManagerBasedRLEnv,
    std: float,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    err = (asset.data.joint_pos.torch - asset.data.default_joint_pos.torch).square().sum(dim=1)
    return standing_gate(env, target_height, sensor_cfg, asset_cfg) * torch.exp(-err / std**2)


def stand_still(
    env: ManagerBasedRLEnv,
    std: float,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    data = env.scene[asset_cfg.name].data
    motion = data.root_lin_vel_b.torch.square().sum(dim=1) + data.root_ang_vel_b.torch.square().sum(dim=1)
    return standing_gate(env, target_height, sensor_cfg, asset_cfg) * torch.exp(-motion / std**2)


def feet_on_ground(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    contact_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[contact_cfg.name]
    forces = sensor.data.net_normal_forces_w_history.torch[:, :, contact_cfg.body_ids]
    down = (forces.norm(dim=-1).amax(dim=1) > 1.0).float().mean(dim=1)
    return standing_gate(env, target_height, sensor_cfg, asset_cfg) * down


def joint_vel_over(env: ManagerBasedRLEnv, limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    qd = env.scene[asset_cfg.name].data.joint_vel.torch[:, asset_cfg.joint_ids]
    return (qd.abs() - limit).clamp(min=0.0).square().sum(dim=1)


def calm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """1 while the body turns under 1 rad/s, 0 once it turns over 2.5 rad/s: it is toppling and moving the
    legs fast to catch it is allowed. Not sinking speed: with that in, model_7998 (2026-10-10_02-38-58)
    sank faster on purpose to free its legs and lay down from standing in 0.5 s instead of 0.7 s."""
    turning = ((env.scene[asset_cfg.name].data.root_ang_vel_w.torch.norm(dim=1) - 1.0) / 1.5).clamp(0.0, 1.0)
    return 1.0 - turning


def joint_vel_over_calm(env: ManagerBasedRLEnv, limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint_vel_over, only while calm."""
    return joint_vel_over(env, limit, asset_cfg) * calm(env, asset_cfg)


def base_ang_vel_over(env: ManagerBasedRLEnv, limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    w = env.scene[asset_cfg.name].data.root_ang_vel_b.torch
    return (w.norm(dim=1) - limit).clamp(min=0.0).square()


def base_lin_vel_over(env: ManagerBasedRLEnv, limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    v = env.scene[asset_cfg.name].data.root_lin_vel_b.torch
    return (v.norm(dim=1) - limit).clamp(min=0.0).square()


def base_rise_over(env: ManagerBasedRLEnv, limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Upward base speed in the world above limit, squared."""
    vz = env.scene[asset_cfg.name].data.root_lin_vel_w.torch[:, 2]
    return (vz - limit).clamp(min=0.0).square()


def _lie_target(env: ManagerBasedRLEnv, pose: dict[str, float], asset_cfg: SceneEntityCfg) -> torch.Tensor:
    cache = env.__dict__.setdefault("_lie_targets", {})
    key = tuple(sorted(pose.items()))
    if key not in cache:
        asset: Articulation = env.scene[asset_cfg.name]
        target = asset.data.default_joint_pos.torch[0].clone()
        for pattern, value in pose.items():
            target[asset.find_joints(pattern)[0]] = value
        cache[key] = target
    return cache[key]


def _lie_error(env: ManagerBasedRLEnv, pose: dict[str, float], asset_cfg: SceneEntityCfg) -> torch.Tensor:
    q = env.scene[asset_cfg.name].data.joint_pos.torch
    return (q - _lie_target(env, pose, asset_cfg)).square().sum(dim=1)


def lying_gate(
    env: ManagerBasedRLEnv,
    lie_joint_pos: dict[str, float],
    max_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    belly_force: float = 30.0,
    contact_sensor: str = "contact_forces",
) -> torch.Tensor:
    """1 once belly down and level, base below max_height, joints near lie_joint_pos and at least belly_force
    on the belly (its pads); 0 when far off. Without the belly term the 30k run knelt on its chin and feet."""
    level = ((_up(env, asset_cfg) - 0.8) / 0.1).clamp(0.0, 1.0)
    low = ((max_height + 0.03 - _base_height(env, sensor_cfg, asset_cfg)) / 0.03).clamp(0.0, 1.0)
    pose = ((1.0 - _lie_error(env, lie_joint_pos, asset_cfg)) / 0.5).clamp(0.0, 1.0)
    sensor = env.scene.sensors[contact_sensor]
    if "_base_contact_id" not in env.__dict__:
        env._base_contact_id = sensor.find_bodies("base")[0][0]
    base_id = env._base_contact_id
    on_belly = (sensor.data.net_normal_forces_w_history.torch[:, 0, base_id].norm(dim=-1) / belly_force).clamp(0.0, 1.0)
    return level * low * pose * on_belly


def lie_pose(
    env: ManagerBasedRLEnv, lie_joint_pos: dict[str, float], std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Closeness to the lying joint pose, counted only while the belly faces the ground."""
    return _up(env, asset_cfg).clamp(min=0.0) * torch.exp(-_lie_error(env, lie_joint_pos, asset_cfg) / std**2)


def base_low(
    env: ManagerBasedRLEnv,
    lie_height: float,
    stand_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """0 at standing height, 1 at lying height, counted only while the belly faces the ground."""
    h = _base_height(env, sensor_cfg, asset_cfg)
    low = ((stand_height - h) / (stand_height - lie_height)).clamp(0.0, 1.0)
    return _up(env, asset_cfg).clamp(min=0.0) * low


def relaxed(
    env: ManagerBasedRLEnv,
    lie_joint_pos: dict[str, float],
    max_height: float,
    scale: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Once lying, low motor torque: the belly carries the weight, so going numb changes nothing."""
    tau = env.scene[asset_cfg.name].data.applied_torque.torch.abs().mean(dim=1)
    return lying_gate(env, lie_joint_pos, max_height, sensor_cfg, asset_cfg) * torch.exp(-tau / scale)


def joint_pos_limits_unless_lying(
    env: ManagerBasedRLEnv,
    lie_joint_pos: dict[str, float],
    max_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """joint_pos_limits, off once lying: folded flat on the pads the front calves rest at -2.69..-2.71,
    past the -2.63 soft limit, whatever the target, and that costs nothing while nothing moves."""
    from isaaclab.envs.mdp import joint_pos_limits

    return joint_pos_limits(env, asset_cfg) * (1.0 - lying_gate(env, lie_joint_pos, max_height, sensor_cfg, asset_cfg))


def lie_still(
    env: ManagerBasedRLEnv,
    lie_joint_pos: dict[str, float],
    max_height: float,
    std: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    data = env.scene[asset_cfg.name].data
    motion = data.root_lin_vel_b.torch.square().sum(dim=1) + data.root_ang_vel_b.torch.square().sum(dim=1)
    return lying_gate(env, lie_joint_pos, max_height, sensor_cfg, asset_cfg) * torch.exp(-motion / std**2)


def base_sink_over(env: ManagerBasedRLEnv, limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Downward base speed in the world above limit, squared."""
    vz = env.scene[asset_cfg.name].data.root_lin_vel_w.torch[:, 2]
    return (-vz - limit).clamp(min=0.0).square()


class body_hits(ManagerTermBase):
    """Sum over the given bodies of how much their peak contact force rose since the last step, past margin.

    Resting contact (the belly when lying, a knee put down gently) rises slowly and costs nothing; a body
    slamming into the ground jumps by hundreds of newtons in one step. Meant for everything but the feet.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._last = None

    def reset(self, env_ids=None) -> None:
        if self._last is not None:
            self._last[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(
        self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, margin: float = 30.0, square_scale: float | None = None
    ) -> torch.Tensor:
        """With square_scale, each body's excess is divided by it and squared, so one big hit costs far more
        than several small ones."""
        forces = env.scene.sensors[sensor_cfg.name].data.net_normal_forces_w_history.torch[:, :, sensor_cfg.body_ids]
        peak = forces.norm(dim=-1).amax(dim=1)
        if self._last is None:
            self._last = torch.zeros_like(peak)
        excess = (peak - self._last - margin).clamp(min=0.0)
        self._last = peak
        if square_scale:
            excess = (excess / square_scale).square()
        return excess.sum(dim=1)
