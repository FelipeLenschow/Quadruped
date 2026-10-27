from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

FOOT_RADIUS = 0.022


def _flipping(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """1 in the flip window, the first flip_s seconds after the switch goes on."""
    return env.command_manager.get_term(command_name).flipping


def _standing(env: ManagerBasedRLEnv, command_name: str | None) -> torch.Tensor | float:
    """1 outside the flip window; always 1 without command_name."""
    return 1.0 if command_name is None else 1.0 - _flipping(env, command_name)


def _ids(env: ManagerBasedRLEnv, kind: str, names: tuple[str, ...], asset: str = "robot") -> list[int]:
    """Body, joint or contact-sensor indices of names, in that order, looked up once."""
    cache = env.__dict__.setdefault("_trick_ids", {})
    key = (kind, asset, names)
    if key not in cache:
        entity = env.scene.sensors[asset] if kind == "sensor" else env.scene[asset]
        find = entity.find_joints if kind == "joint" else entity.find_bodies
        cache[key] = find(list(names), preserve_order=True)[0]
    return cache[key]


def _body_offset(env: ManagerBasedRLEnv, body: str, origin: str) -> torch.Tensor:
    """Position of body relative to origin, in the base frame (x forward, y left)."""
    data = env.scene["robot"].data
    b, o = _ids(env, "body", (body, origin))
    delta = data.body_pos_w.torch[:, b] - data.body_pos_w.torch[:, o]
    return quat_apply_inverse(data.root_quat_w.torch, delta)


def _ground(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Mean height of the rays that hit; the env origin if none did."""
    z = env.scene.sensors[sensor_cfg.name].data.ray_hits_w.torch[..., 2]
    hit = torch.isfinite(z)
    count = hit.sum(dim=1)
    ground = torch.where(hit, z, 0.0).sum(dim=1) / count.clamp(min=1)
    return torch.where(count > 0, ground, env.scene.env_origins[:, 2])


def _base_height(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_pos_w.torch[:, 2] - _ground(env, sensor_cfg)


def _foot_heights(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    z = env.scene[asset_cfg.name].data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
    return z - _ground(env, sensor_cfg).unsqueeze(1) - FOOT_RADIUS


def _in_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    forces = env.scene.sensors[sensor_cfg.name].data.net_normal_forces_w_history.torch[:, :, sensor_cfg.body_ids]
    return forces.norm(dim=-1).amax(dim=1) > 1.0


def _pitch_roll(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor]:
    """Pitch (+ nose down, as the safety check measures it) and roll from projected gravity."""
    g = env.scene[asset_cfg.name].data.projected_gravity_b.torch
    return torch.atan2(g[:, 0], -g[:, 2]), torch.atan2(-g[:, 1], -g[:, 2])


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    qvel = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque.torch[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


# ── Standing, outside the flip window ────────────────────────────────────────────────────────


def stand_pose_off(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    err = (asset.data.joint_pos.torch - asset.data.default_joint_pos.torch).square().sum(dim=1)
    return _standing(env, command_name) * torch.exp(-err / std**2)


def base_height_off(
    env: ManagerBasedRLEnv,
    target: float,
    std: float,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    err = _base_height(env, sensor_cfg, asset_cfg) - target
    return _standing(env, command_name) * torch.exp(-(err / std).square())


def level_off(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalty: base tilt (squared projected gravity xy) outside the flip window."""
    g = env.scene[asset_cfg.name].data.projected_gravity_b.torch
    return _standing(env, command_name) * g[:, :2].square().sum(dim=1)


def still(env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Base not moving, outside the flip window."""
    data = env.scene[asset_cfg.name].data
    motion = data.root_lin_vel_b.torch.square().sum(dim=1) + data.root_ang_vel_b.torch.square().sum(dim=1)
    return _standing(env, command_name) * torch.exp(-motion / std**2)


# ── Feet ─────────────────────────────────────────────────────────────────────────────────────


def support_feet_air(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str) -> torch.Tensor:
    """Penalty: feet off the ground, outside the flip window."""
    return _standing(env, command_name) * (~_in_contact(env, sensor_cfg)).float().sum(dim=1)


def feet_slip(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalty: squared horizontal speed of the feet that touch the ground."""
    v = env.scene[asset_cfg.name].data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    return (_in_contact(env, sensor_cfg).float() * v.square().sum(dim=-1)).sum(dim=1)


# ── Safety: what the robot's checks would trip on ────────────────────────────────────────────


def tilt_over(
    env: ManagerBasedRLEnv, limit: float, command_name: str | None = None, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalty: base tilt (any direction) past limit rad, squared; with command_name outside the flip window."""
    up = -env.scene[asset_cfg.name].data.projected_gravity_b.torch[:, 2]
    return _standing(env, command_name) * (torch.acos(up.clamp(-1.0, 1.0)) - limit).clamp(min=0.0).square()


class safe_rom(ManagerTermBase):
    """Penalty: how far (rad, summed) the joints are outside the deploy check's range, which leaves out the
    outer `margin` of each joint's hardware range (config.yaml joint_limits, joint_rom_safety_margin)."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("robot")).name]
        lo = torch.full((asset.num_joints,), -torch.inf, device=env.device)
        hi = torch.full((asset.num_joints,), torch.inf, device=env.device)
        margin = cfg.params["margin"]
        for pattern, (a, b) in cfg.params["hard_limits"].items():
            ids = asset.find_joints(pattern)[0]
            lo[ids] = a + margin * (b - a)
            hi[ids] = b - margin * (b - a)
        self._lo, self._hi = lo, hi

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        hard_limits: dict[str, tuple[float, float]],
        margin: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        q = env.scene[asset_cfg.name].data.joint_pos.torch
        return ((self._lo - q).clamp(min=0.0) + (q - self._hi).clamp(min=0.0)).sum(dim=1)


def joint_vel_over(
    env: ManagerBasedRLEnv, limit: float, command_name: str | None = None, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    qd = env.scene[asset_cfg.name].data.joint_vel.torch[:, asset_cfg.joint_ids]
    return _standing(env, command_name) * (qd.abs() - limit).clamp(min=0.0).square().sum(dim=1)


def base_ang_vel_over(
    env: ManagerBasedRLEnv, limit: float, command_name: str | None = None, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    w = env.scene[asset_cfg.name].data.root_ang_vel_b.torch
    return _standing(env, command_name) * (w.norm(dim=1) - limit).clamp(min=0.0).square()


def base_lin_vel_over(
    env: ManagerBasedRLEnv, limit: float, command_name: str | None = None, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    v = env.scene[asset_cfg.name].data.root_lin_vel_b.torch
    return _standing(env, command_name) * (v.norm(dim=1) - limit).clamp(min=0.0).square()


class body_hits(ManagerTermBase):
    """Sum over the given bodies of how much their peak contact force rose since the last step, past margin."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._last = None

    def reset(self, env_ids=None) -> None:
        if self._last is not None:
            self._last[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, margin: float = 30.0) -> torch.Tensor:
        forces = env.scene.sensors[sensor_cfg.name].data.net_normal_forces_w_history.torch[:, :, sensor_cfg.body_ids]
        peak = forces.norm(dim=-1).amax(dim=1)
        if self._last is None:
            self._last = torch.zeros_like(peak)
        excess = (peak - self._last - margin).clamp(min=0.0)
        self._last = peak
        return excess.sum(dim=1)


# ── Backflip: in the flip window, jump, turn 2*pi backward about the body y axis, land on the feet ──────


def _ref_pitch(env: ManagerBasedRLEnv, command_name: str, start: float, end: float) -> torch.Tensor:
    """Reference pitch (rad, + nose down): 0 before start, then a smooth turn to -2*pi (nose up first) at end."""
    t = env.command_manager.get_term(command_name).on_time
    x = ((t - start) / (end - start)).clamp(0.0, 1.0)
    return -2.0 * torch.pi * x * x * (3.0 - 2.0 * x)


def flip_attitude(
    env: ManagerBasedRLEnv, start: float, end: float, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Gravity in the base frame close to where the reference pitch puts it, in the flip window. Gravity has
    no wrap at pi, so it follows the whole turn."""
    pitch = _ref_pitch(env, command_name, start, end)
    ref = torch.stack([torch.sin(pitch), torch.zeros_like(pitch), -torch.cos(pitch)], dim=1)
    err = (env.scene[asset_cfg.name].data.projected_gravity_b.torch - ref).square().sum(dim=1)
    return _flipping(env, command_name) * torch.exp(-err / std**2)


def flip_rotation(
    env: ManagerBasedRLEnv, cap: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Backward pitch rate (rad/s, up to cap) in the flip window, until the turn adds up to 2*pi."""
    term = env.command_manager.get_term(command_name)
    rate = (-env.scene[asset_cfg.name].data.root_ang_vel_b.torch[:, 1]).clamp(0.0, cap)
    return term.flipping * (term.rotation < 2.0 * torch.pi).float() * rate


def flip_height(
    env: ManagerBasedRLEnv,
    start: float,
    end: float,
    stand: float,
    rise: float,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Base height above the standing height, as a share of rise, between start and end of the turn."""
    t = env.command_manager.get_term(command_name).on_time
    window = ((t > start) & (t < end)).float()
    up = ((_base_height(env, sensor_cfg, asset_cfg) - stand) / rise).clamp(0.0, 1.0)
    return _flipping(env, command_name) * window * up


def flip_twist(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalty: roll and yaw rate in the flip window, squared: turn about the y axis only."""
    w = env.scene[asset_cfg.name].data.root_ang_vel_b.torch
    return _flipping(env, command_name) * (w[:, 0].square() + w[:, 2].square())


def sideways_flip(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalty: the body's y axis off horizontal (sin squared of the lean), in the flip window. Unlike roll from
    atan2, it holds up with the body upside down."""
    g = env.scene[asset_cfg.name].data.projected_gravity_b.torch
    return _flipping(env, command_name) * g[:, 1].square()


def flip_missed(env: ManagerBasedRLEnv, share: float, command_name: str) -> torch.Tensor:
    """Penalty: on, past the flip window, with less than share of the 2*pi turn done."""
    term = env.command_manager.get_term(command_name)
    return (term.on - term.flipping) * (term.rotation < share * 2.0 * torch.pi).float()


def _yaw_from_anchor(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    anchor = env.command_manager.get_term(command_name).anchor_yaw
    return wrap_to_pi(env.scene[asset_cfg.name].data.heading_w.torch - anchor)


