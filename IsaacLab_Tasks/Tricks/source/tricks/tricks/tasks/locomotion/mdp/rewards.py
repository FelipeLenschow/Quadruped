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
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str, lifted: list[int] | None = None
) -> torch.Tensor:
    """Penalty: feet off the ground. lifted[i] (an index in sensor_cfg) is the foot switch i lifts; it does
    not count while that switch is on."""
    air = (~_in_contact(env, sensor_cfg)).float()
    command = env.command_manager.get_command(command_name)
    for i, foot in enumerate(lifted or []):
        air[:, foot] *= 1.0 - command[:, i]
    return air.sum(dim=1)


def feet_slip(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalty: squared horizontal speed of the feet that touch the ground."""
    v = env.scene[asset_cfg.name].data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    return (_in_contact(env, sensor_cfg).float() * v.square().sum(dim=-1)).sum(dim=1)


# ── Safety: what the robot's checks would trip on ────────────────────────────────────────────


def tilt_over(env: ManagerBasedRLEnv, limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalty: base tilt (any direction) past limit rad, squared."""
    up = -env.scene[asset_cfg.name].data.projected_gravity_b.torch[:, 2]
    return (torch.acos(up.clamp(-1.0, 1.0)) - limit).clamp(min=0.0).square()


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


# ── Pee: one rear foot up, out to the side, calf folded; legs[i] is the rear leg switch i lifts ──────


def _foot_height(env: ManagerBasedRLEnv, leg: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    z = env.scene["robot"].data.body_pos_w.torch[:, _ids(env, "body", (f"{leg}_foot",))[0], 2]
    return z - _ground(env, sensor_cfg) - FOOT_RADIUS


def _joint_pos(env: ManagerBasedRLEnv, joint: str) -> torch.Tensor:
    return env.scene["robot"].data.joint_pos.torch[:, _ids(env, "joint", (joint,))[0]]


def pee_unload(env: ManagerBasedRLEnv, legs: list[str], load: float, command_name: str) -> torch.Tensor:
    """Weight off the foot: 0 at `load` N or more, 1 at none. Pays for shifting the weight onto the other
    three before the foot can leave the ground."""
    forces = env.scene.sensors["contact_forces"].data.net_normal_forces_w_history.torch[:, 0]

    def unload(leg):
        force = forces[:, _ids(env, "sensor", (f"{leg}_foot",), "contact_forces")[0]].norm(dim=-1)
        return (1.0 - force / load).clamp(0.0, 1.0)

    return _per_switch(env, command_name, legs, unload)


def pee_lift(env: ManagerBasedRLEnv, legs: list[str], target: float, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Foot height as a share of target: pays from the first centimetre."""
    return _per_switch(env, command_name, legs, lambda leg: (_foot_height(env, leg, sensor_cfg) / target).clamp(0.0, 1.0))


def pee_up(
    env: ManagerBasedRLEnv, legs: list[str], target: float, std: float, command_name: str, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Foot at least target m off the ground; full reward from there up."""
    def up(leg):
        h = _foot_height(env, leg, sensor_cfg)
        return torch.exp(-((h.clamp(max=target) - target) / std).square())

    return _per_switch(env, command_name, legs, up)


def pee_hip_out(env: ManagerBasedRLEnv, legs: list[str], target: float, command_name: str) -> torch.Tensor:
    """Hip turned outward (+ on left legs, - on right) up to target rad."""
    def out(leg):
        sign = 1.0 if leg[1] == "L" else -1.0
        return (sign * _joint_pos(env, f"{leg}_hip_joint") / target).clamp(0.0, 1.0)

    return _per_switch(env, command_name, legs, out)


def pee_calf(env: ManagerBasedRLEnv, legs: list[str], target: float, std: float, command_name: str) -> torch.Tensor:
    """Calf folded to target rad."""
    return _per_switch(
        env, command_name, legs, lambda leg: torch.exp(-((_joint_pos(env, f"{leg}_calf_joint") - target) / std).square())
    )


def pee_beside_hip(env: ManagerBasedRLEnv, legs: list[str], std: float, command_name: str) -> torch.Tensor:
    """Foot level with its hip front to back, so it goes out to the side, not back."""
    return _per_switch(
        env, command_name, legs, lambda leg: torch.exp(-(_body_offset(env, f"{leg}_foot", f"{leg}_hip")[:, 0] / std).square())
    )


def pee_front_clear(env: ManagerBasedRLEnv, legs: list[str], distance: float, command_name: str) -> torch.Tensor:
    """The front foot on the lifted side ahead of its hip, up to distance m: out of the stream's way."""
    def clear(leg):
        front = "F" + leg[1]
        return (_body_offset(env, f"{front}_foot", f"{front}_hip")[:, 0] / distance).clamp(0.0, 1.0)

    return _per_switch(env, command_name, legs, clear)


def pee_support_pose(env: ManagerBasedRLEnv, legs: list[str], std: float, command_name: str) -> torch.Tensor:
    """The two legs on the other side near the standing pose."""
    asset: Articulation = env.scene["robot"]

    def pose(leg):
        other = "R" if leg[1] == "L" else "L"
        ids = _ids(env, "joint", tuple(f"{f}{other}_{j}_joint" for f in "FR" for j in ("hip", "thigh", "calf")))
        err = (asset.data.joint_pos.torch[:, ids] - asset.data.default_joint_pos.torch[:, ids]).square().sum(dim=1)
        return torch.exp(-err / std**2)

    return _per_switch(env, command_name, legs, pose)


# ── Sniff: nose down, body low ───────────────────────────────────────────────────────────────


def pitch_on(
    env: ManagerBasedRLEnv, target: float, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    pitch, _ = _pitch_roll(env, asset_cfg)
    return _on(env, command_name) * torch.exp(-((pitch - target) / std).square())


def pitch_track(
    env: ManagerBasedRLEnv,
    center: float,
    amplitude: float,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Pitch following center + amplitude * sin(clock phase); the command carries the sin."""
    pitch, _ = _pitch_roll(env, asset_cfg)
    target = center + amplitude * env.command_manager.get_command(command_name)[:, 1]
    return _on(env, command_name) * torch.exp(-((pitch - target) / std).square())


def base_height_on(
    env: ManagerBasedRLEnv,
    target: float,
    std: float,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    err = _base_height(env, sensor_cfg, asset_cfg) - target
    return _on(env, command_name) * torch.exp(-(err / std).square())


def roll_on(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalty: squared roll while on."""
    _, roll = _pitch_roll(env, asset_cfg)
    return _on(env, command_name) * roll.square()


def joint_vel_on(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalty: squared velocity of asset_cfg's joints while on."""
    qd = env.scene[asset_cfg.name].data.joint_vel.torch[:, asset_cfg.joint_ids]
    return _on(env, command_name) * qd.square().sum(dim=1)


# ── Swinging: Wiggle's yaw, Sniff's pitch ────────────────────────────────────────────────────
# No clock: turning fast pays, reaching the ends of the swing pays, going past them costs, so the best
# is to swing between them at the rate. Period about 4 x amplitude / rate.


def _yaw_from_anchor(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    anchor = env.command_manager.get_term(command_name).anchor_yaw
    return wrap_to_pi(env.scene[asset_cfg.name].data.heading_w.torch - anchor)


def _swing(env: ManagerBasedRLEnv, axis: str, command_name: str, asset_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor]:
    """Angle and its rate: yaw from the anchor and world yaw rate, or pitch (+ nose down) and body pitch rate."""
    data = env.scene[asset_cfg.name].data
    if axis == "yaw":
        return _yaw_from_anchor(env, command_name, asset_cfg), data.root_ang_vel_w.torch[:, 2]
    pitch, _ = _pitch_roll(env, asset_cfg)
    return pitch, data.root_ang_vel_b.torch[:, 1]


def swing_rate(
    env: ManagerBasedRLEnv, axis: str, rate: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    _, w = _swing(env, axis, command_name, asset_cfg)
    return _on(env, command_name) * (w.abs() / rate).clamp(max=1.0)


def swing_reach(
    env: ManagerBasedRLEnv,
    axis: str,
    center: float,
    amplitude: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """0 within half the amplitude of center, 1 from 0.9 of it to the amplitude, back to 0 at 1.5 of it."""
    angle, _ = _swing(env, axis, command_name, asset_cfg)
    d = (angle - center).abs()
    up = ((d - 0.5 * amplitude) / (0.4 * amplitude)).clamp(0.0, 1.0)
    down = 1.0 - ((d - amplitude) / (0.5 * amplitude)).clamp(0.0, 1.0)
    return _on(env, command_name) * up * down


def swing_over(
    env: ManagerBasedRLEnv,
    axis: str,
    center: float,
    amplitude: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty: past the amplitude from center, squared, counted up to half the amplitude past it: far off
    center (Sniff starting level) it would swamp everything else."""
    angle, _ = _swing(env, axis, command_name, asset_cfg)
    return _on(env, command_name) * ((angle - center).abs() - amplitude).clamp(0.0, 0.5 * amplitude).square()


def pivot_still(
    env: ManagerBasedRLEnv, forward: float, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The point `forward` m ahead of the base centre stays where it was when the trick went on, so the
    body turns about it and only the back end swings."""
    term = env.command_manager.get_term(command_name)
    data = env.scene[asset_cfg.name].data
    yaw = data.heading_w.torch
    now = data.root_pos_w.torch[:, :2] + forward * torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=1)
    then = term.anchor_xy + forward * torch.stack((torch.cos(term.anchor_yaw), torch.sin(term.anchor_yaw)), dim=1)
    return _on(env, command_name) * torch.exp(-(now - then).square().sum(dim=1) / std**2)
