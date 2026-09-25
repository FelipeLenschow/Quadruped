from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardManager


class PositiveRewardManager(RewardManager):
    """Walk These Ways' reward: positive-weight terms times exp(negative-weight terms / sigma).

    The contact and height terms sit with the positive weights but are themselves negative, so the
    product is clipped at zero: a step never pays less than an ended episode does.
    """

    def __init__(self, cfg, env, sigma: float):
        super().__init__(cfg, env)
        self.sigma = sigma

    def compute(self, dt: float) -> torch.Tensor:
        positive = torch.zeros_like(self._reward_buf)
        negative = torch.zeros_like(self._reward_buf)
        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                continue
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            if term_cfg.weight > 0.0:
                positive += value
            else:
                negative += value
            self._episode_sums[name] += value
            self._step_reward[:, term_idx] = value / dt
        self._reward_buf[:] = torch.clamp(positive * torch.exp(negative / self.sigma), min=0.0)
        return self._reward_buf


class WTWEnv(ManagerBasedRLEnv):
    def load_managers(self):
        super().load_managers()
        self.reward_manager = PositiveRewardManager(self.cfg.rewards, self, self.cfg.positive_reward_sigma)
        print("[WTW] reward = positive terms * exp(negative terms /", self.cfg.positive_reward_sigma, ")")
