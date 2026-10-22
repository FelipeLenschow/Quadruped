from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

FOOT_RADIUS = 0.022


def _on(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """1 while any of the trick's switches is on."""
    return env.command_manager.get_term(command_name).on


def _per_switch(env: ManagerBasedRLEnv, command_name: str, legs: list[str], fn) -> torch.Tensor:
    """fn(leg) for the leg whose switch is on (legs[i] goes with switch i), 0 while all are off."""
    command = env.command_manager.get_command(command_name)
    return sum(command[:, i] * fn(leg) for i, leg in enumerate(legs))


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


# ── Standing, while the trick is off ─────────────────────────────────────────────────────────


def stand_pose_off(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    err = (asset.data.joint_pos.torch - asset.data.default_joint_pos.torch).square().sum(dim=1)
    return (1.0 - _on(env, command_name)) * torch.exp(-err / std**2)


def base_height_off(
    env: ManagerBasedRLEnv,
    target: float,
    std: float,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    err = _base_height(env, sensor_cfg, asset_cfg) - target
    return (1.0 - _on(env, command_name)) * torch.exp(-(err / std).square())


def level_off(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalty: base tilt (squared projected gravity xy) while off."""
    g = env.scene[asset_cfg.name].data.projected_gravity_b.torch
    return (1.0 - _on(env, command_name)) * g[:, :2].square().sum(dim=1)


def still(
    env: ManagerBasedRLEnv, std: float, command_name: str, when_on: bool = True, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Base not moving; with when_on=False only while off (a trick that moves the base)."""
    data = env.scene[asset_cfg.name].data
    motion = data.root_lin_vel_b.torch.square().sum(dim=1) + data.root_ang_vel_b.torch.square().sum(dim=1)
    gate = 1.0 if when_on else 1.0 - _on(env, command_name)
    return gate * torch.exp(-motion / std**2)


# ── Feet ─────────────────────────────────────────────────────────────────────────────────────


def support_feet_air(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str, lifted: list[list[int]] | None = None
) -> torch.Tensor:
    """Penalty: feet off the ground. lifted[i] (indices in sensor_cfg) are the feet switch i lifts; they do
    not count while that switch is on."""
    air = (~_in_contact(env, sensor_cfg)).float()
    command = env.command_manager.get_command(command_name)
    for i, feet in enumerate(lifted or []):
        air[:, feet] *= 1.0 - command[:, i : i + 1]
    return air.sum(dim=1)


def feet_slip(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalty: squared horizontal speed of the feet that touch the ground."""
    v = env.scene[asset_cfg.name].data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    return (_in_contact(env, sensor_cfg).float() * v.square().sum(dim=-1)).sum(dim=1)


# ── Safety: what the robot's checks would trip on ────────────────────────────────────────────


def tilt_over(
    env: ManagerBasedRLEnv, limit: float, command_name: str | None = None, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalty: base tilt (any direction) past limit rad, squared; with command_name only while off."""
    up = -env.scene[asset_cfg.name].data.projected_gravity_b.torch[:, 2]
    gate = 1.0 if command_name is None else 1.0 - _on(env, command_name)
    return gate * (torch.acos(up.clamp(-1.0, 1.0)) - limit).clamp(min=0.0).square()


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


def joint_vel_over(env: ManagerBasedRLEnv, limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    qd = env.scene[asset_cfg.name].data.joint_vel.torch[:, asset_cfg.joint_ids]
    return (qd.abs() - limit).clamp(min=0.0).square().sum(dim=1)


def base_ang_vel_over(env: ManagerBasedRLEnv, limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    w = env.scene[asset_cfg.name].data.root_ang_vel_b.torch
    return (w.norm(dim=1) - limit).clamp(min=0.0).square()


def base_lin_vel_over(env: ManagerBasedRLEnv, limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    v = env.scene[asset_cfg.name].data.root_lin_vel_b.torch
    return (v.norm(dim=1) - limit).clamp(min=0.0).square()


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


# ── Handstand: switch 0 up on the front legs (nose down), switch 1 up on the rear legs (nose up) ─────
# Up on one pair of legs, the base pitched `target` rad toward the other pair, which is off the ground.
# Each term takes legs = [support pair of switch 0, of switch 1], e.g. ["F", "R"].


def _pair(leg: str) -> tuple[str, str]:
    return (f"{leg}L_foot", f"{leg}R_foot")


def _other(leg: str) -> str:
    return "R" if leg == "F" else "F"


def _signed_pitch(env: ManagerBasedRLEnv, leg: str) -> torch.Tensor:
    """Pitch toward the support pair: nose down for the front legs, nose up for the rear."""
    pitch, _ = _pitch_roll(env, SceneEntityCfg("robot"))
    return pitch if leg == "F" else -pitch


def on_legs_pitch(env: ManagerBasedRLEnv, legs: list[str], target: dict[str, float], command_name: str) -> torch.Tensor:
    """Pitch toward the support pair as a share of target[pair]: pays from the first degree."""
    return _per_switch(env, command_name, legs, lambda leg: (_signed_pitch(env, leg) / target[leg]).clamp(0.0, 1.0))


def on_legs_pitch_exact(
    env: ManagerBasedRLEnv, legs: list[str], target: dict[str, float], std: float, command_name: str
) -> torch.Tensor:
    return _per_switch(env, command_name, legs, lambda leg: torch.exp(-((_signed_pitch(env, leg) - target[leg]) / std).square()))


def on_legs_unload(env: ManagerBasedRLEnv, legs: list[str], load: float, command_name: str) -> torch.Tensor:
    """Weight off the pair that goes up: 0 at `load` N or more on them together, 1 at none. Pays for
    moving the weight over the support pair before the others can leave the ground."""
    forces = env.scene.sensors["contact_forces"].data.net_normal_forces_w_history.torch[:, 0]

    def unload(leg):
        ids = _ids(env, "sensor", _pair(_other(leg)), "contact_forces")
        return (1.0 - forces[:, ids].norm(dim=-1).sum(dim=1) / load).clamp(0.0, 1.0)

    return _per_switch(env, command_name, legs, unload)


def on_legs_balance(env: ManagerBasedRLEnv, legs: list[str], std: float, command_name: str) -> torch.Tensor:
    """Centre of mass over the support pair: horizontal distance from the robot's centre of mass to the
    middle of the two support feet."""
    data = env.scene["robot"].data
    masses = data.body_mass.torch
    com =(data.body_com_pos_w.torch[..., :2] * masses.unsqueeze(-1)).sum(dim=1) / masses.sum(dim=1, keepdim=True)

    def balance(leg):
        feet = data.body_pos_w.torch[:, _ids(env, "body", _pair(leg)), :2].mean(dim=1)
        return torch.exp(-(com - feet).square().sum(dim=1) / std**2)

    return _per_switch(env, command_name, legs, balance)


def on_legs_still(
    env: ManagerBasedRLEnv,
    legs: list[str],
    target: dict[str, float],
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Base not moving, once within 0.2 rad of the target pitch: hold the pose, do not rock in it."""
    data = env.scene[asset_cfg.name].data
    motion = data.root_lin_vel_b.torch.square().sum(dim=1) + data.root_ang_vel_b.torch.square().sum(dim=1)

    def hold(leg):
        near = ((_signed_pitch(env, leg) - target[leg]).abs() < 0.2).float()
        return near * torch.exp(-motion / std**2)

    return _per_switch(env, command_name, legs, hold)


def sideways_on(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalty: the body's y axis off horizontal (sin squared of the lean), while on. Unlike roll from
    atan2, it holds up with the body near vertical."""
    g = env.scene[asset_cfg.name].data.projected_gravity_b.torch
    return _on(env, command_name) * g[:, 1].square()


def _yaw_from_anchor(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    anchor = env.command_manager.get_term(command_name).anchor_yaw
    return wrap_to_pi(env.scene[asset_cfg.name].data.heading_w.torch - anchor)


