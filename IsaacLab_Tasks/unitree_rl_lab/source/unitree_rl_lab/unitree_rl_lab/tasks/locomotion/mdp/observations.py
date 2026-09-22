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


class base_lin_vel_estimate(ManagerTermBase):
    """Root linear velocity with a per-episode constant offset.

    On hardware this channel is a Kalman estimate, not a measurement: its error is dominated by
    a slowly-varying offset, not by the zero-mean sample noise a Unoise term adds. The offset is
    resampled at every reset and held for the episode; white noise stays on the ObsTerm.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._bias = torch.zeros(env.num_envs, 3, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> dict:
        bias = float(self.cfg.params.get("bias", 0.0))
        ids = slice(None) if env_ids is None else env_ids
        self._bias[ids] = torch.empty_like(self._bias[ids]).uniform_(-bias, bias)
        return {}

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        bias: float = 0.0,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        vel = env.scene[asset_cfg.name].data.root_lin_vel_b
        return (vel.torch if hasattr(vel, "torch") else vel) + self._bias
