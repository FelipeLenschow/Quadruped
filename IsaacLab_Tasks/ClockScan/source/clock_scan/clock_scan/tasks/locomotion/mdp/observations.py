from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg


class biased_obs(ManagerTermBase):
    """Another observation term plus a per-episode constant offset, resampled at reset.

    Sensor error on hardware is not zero-mean within an episode: the velocity estimate drifts,
    the IMU sits at whatever mounting tilt it was bolted at, the joint encoders keep the zero
    they were last calibrated to. A Unoise term cannot stand in for any of that -- it draws
    again every step, so a policy can average it away. This holds one draw for the episode.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._bias = None

    def reset(self, env_ids: torch.Tensor | None = None) -> dict:
        if self._bias is None:
            return {}
        bias = float(self.cfg.params.get("bias", 0.0))
        ids = slice(None) if env_ids is None else env_ids
        self._bias[ids] = torch.empty_like(self._bias[ids]).uniform_(-bias, bias)
        return {}

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        func,
        bias: float = 0.0,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        obs = func(env, asset_cfg=asset_cfg)
        if self._bias is None:
            self._bias = torch.zeros_like(obs)
            self.reset()
        return obs + self._bias


def speed_clock(env: ManagerBasedRLEnv, command_name: str = "clock") -> torch.Tensor:
    return env.command_manager.get_term(command_name).clock


def speed_clock_contact(env: ManagerBasedRLEnv, command_name: str = "clock") -> torch.Tensor:
    return env.command_manager.get_term(command_name).desired_contact


def base_height_ground(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Base height above the ground under it (rewards.ground_height)."""
    from .rewards import ground_height

    ground = ground_height(env, sensor_cfg)
    return (env.scene[asset_cfg.name].data.root_pos_w.torch[:, 2] - ground).unsqueeze(1)


class biased_height_scan(ManagerTermBase):
    """height_scan plus one offset per episode for every ray, drawn from +-bias at reset: the height map's
    error against the base height, which a per-ray noise term averages away."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._bias = torch.zeros(env.num_envs, 1, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> dict:
        bias = float(self.cfg.params.get("bias", 0.0))
        ids = slice(None) if env_ids is None else env_ids
        self._bias[ids] = torch.empty_like(self._bias[ids]).uniform_(-bias, bias)
        return {}

    def __call__(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, bias: float = 0.0, offset: float = 0.5) -> torch.Tensor:
        from isaaclab.envs.mdp import height_scan

        return height_scan(env, sensor_cfg, offset) + self._bias


def leash_error(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """Leash reference minus base: position in the base's yaw frame (2, m) and heading (1, rad)."""
    from isaaclab.utils.math import euler_xyz_from_quat

    term = env.command_manager.get_term(command_name)
    yaw = euler_xyz_from_quat(env.scene["robot"].data.root_quat_w.torch)[2]
    c, s = torch.cos(yaw), torch.sin(yaw)
    e = term.leash_pos_error_w
    return torch.stack([c * e[:, 0] + s * e[:, 1], -s * e[:, 0] + c * e[:, 1], term.leash_yaw_error], dim=1)


def head_clearance(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, max_clearance: float = 1.0
) -> torch.Tensor:
    """Height of the body of asset_cfg (the head) above the ground straight under it, from one downward ray,
    clipped to [0, max_clearance]."""
    hit = env.scene.sensors[sensor_cfg.name].data.ray_hits_w.torch[:, 0, 2]
    body = env.scene[asset_cfg.name].data.body_pos_w.torch[:, asset_cfg.body_ids[0], 2]
    gap = torch.where(torch.isfinite(hit), body - hit, torch.full_like(body, max_clearance))
    return gap.clamp(0.0, max_clearance).unsqueeze(1)


class lidar_ranges(ManagerTermBase):
    """Distance along each ray of a ray caster from its origin (the lidar), clipped to max_range (no hit reads
    max_range), as the L1 measures it, for the frames lags steps back (0 is now), oldest last. Each ray of each
    frame is, independently: max_range with drop_prob (no return), an older reading (1 to stale_steps more steps
    back) with stale_prob (the sparse, non-repeating scan), plus uniform noise of +-noise m."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        p = cfg.params
        self._sensor = env.scene.sensors[p["sensor_cfg"].name]
        # data.pos_w is the frame the sensor rides on; the rays start at the offset from it.
        self._offset = torch.tensor(self._sensor.cfg.offset.pos, device=env.device).unsqueeze(0)
        self._lags = torch.tensor(p.get("lags", (0,)), device=env.device)
        self._depth = int(self._lags.max()) + int(p.get("stale_steps", 0)) + 1
        rays = self._sensor.num_rays if hasattr(self._sensor, "num_rays") else self._sensor.ray_directions.shape[1]
        self._buf = torch.zeros(env.num_envs, self._depth, rays, device=env.device)
        self._fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self._step = -1

    def reset(self, env_ids: torch.Tensor | None = None) -> dict:
        self._fresh[slice(None) if env_ids is None else env_ids] = True
        return {}

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        max_range: float = 3.0,
        lags: tuple[int, ...] = (0,),
        noise: float = 0.0,
        drop_prob: float = 0.0,
        stale_prob: float = 0.0,
        stale_steps: int = 0,
    ) -> torch.Tensor:
        from isaaclab.utils.math import quat_apply

        data = self._sensor.data
        hits = data.ray_hits_w.torch
        quat = data.quat_w.torch
        origin = data.pos_w.torch + quat_apply(quat, self._offset.expand(quat.shape[0], -1))
        dist = torch.linalg.norm(hits - origin.unsqueeze(1), dim=-1)
        dist = torch.nan_to_num(dist, nan=max_range, posinf=max_range).clamp(max=max_range)
        if env.common_step_counter != self._step:
            self._step = env.common_step_counter
            self._buf = torch.roll(self._buf, 1, dims=1)
        self._buf[:, 0] = dist
        fresh = self._fresh
        self._buf[fresh] = dist[fresh].unsqueeze(1).expand(-1, self._depth, -1)
        self._fresh[:] = False
        n, rays = dist.shape
        lag = self._lags.view(1, -1, 1).expand(n, -1, rays)
        if stale_prob > 0.0 and stale_steps > 0:
            extra = torch.randint(1, stale_steps + 1, lag.shape, device=env.device)
            lag = lag + extra * (torch.rand(lag.shape, device=env.device) < stale_prob)
        out = torch.gather(self._buf, 1, lag)
        if noise > 0.0:
            out = out + torch.empty_like(out).uniform_(-noise, noise)
        if drop_prob > 0.0:
            out = torch.where(torch.rand_like(out) < drop_prob, torch.full_like(out, max_range), out)
        return out.clamp(0.0, max_range).reshape(n, -1)
