from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


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


class base_lin_vel_estimate(biased_obs):
    """Kept for the Noises run, whose params/env.yaml names this term. Superseded by biased_obs."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        bias: float = 0.0,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        from isaaclab.envs.mdp import base_lin_vel

        return super().__call__(env, func=base_lin_vel, bias=bias, asset_cfg=asset_cfg)
