from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg


class biased_obs(ManagerTermBase):
    """Another observation term plus a per-episode constant offset, resampled at reset.

    Sensor error on hardware is not zero-mean within an episode: the IMU sits at whatever mounting
    tilt it was bolted at, the joint encoders keep the zero they were last calibrated to. A Unoise
    term draws again every step, so a policy can average it away. This holds one draw for the episode.
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


def base_height_ground(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    from .rewards import _base_height

    return _base_height(env, sensor_cfg, asset_cfg).unsqueeze(1)


def body_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    forces = env.scene.sensors[sensor_cfg.name].data.net_normal_forces_w_history.torch[:, :, sensor_cfg.body_ids]
    return (forces.norm(dim=-1).amax(dim=1) > threshold).float()


def feet_height(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    from .rewards import _foot_heights

    return _foot_heights(env, sensor_cfg, asset_cfg)


def trick_yaw(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    from .rewards import _yaw_from_anchor

    return _yaw_from_anchor(env, command_name, asset_cfg).unsqueeze(1)
