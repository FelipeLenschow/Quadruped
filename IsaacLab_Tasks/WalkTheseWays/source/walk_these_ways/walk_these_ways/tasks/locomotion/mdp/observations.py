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


def gait_command(env: ManagerBasedRLEnv, command_name: str = "gait") -> torch.Tensor:
    return env.command_manager.get_term(command_name).command


def gait_clock(env: ManagerBasedRLEnv, command_name: str = "gait") -> torch.Tensor:
    return env.command_manager.get_term(command_name).clock


def gait_desired_contact(env: ManagerBasedRLEnv, command_name: str = "gait") -> torch.Tensor:
    return env.command_manager.get_term(command_name).desired_contact
