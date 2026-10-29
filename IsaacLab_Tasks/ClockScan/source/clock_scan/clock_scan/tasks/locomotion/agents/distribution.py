"""Gaussian action distribution with its mean kept inside +-bound by bound * tanh(x / bound)."""

import torch
from torch import nn

from rsl_rl.modules.distribution import GaussianDistribution


class _BoundedMean(nn.Module):
    def __init__(self, bound: float) -> None:
        super().__init__()
        self.bound = bound

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return self.bound * torch.tanh(mlp_output / self.bound)


class BoundedGaussianDistribution(GaussianDistribution):
    """With the env clipping actions, a mean past the clip changes nothing in the env, so PPO lets it drift
    (to 68 in getting-up states) and the log-probs of those samples blow up the KL and the learning rate."""

    def __init__(self, output_dim: int, bound: float = 10.0, **kwargs) -> None:
        super().__init__(output_dim, **kwargs)
        self.bounded_mean = _BoundedMean(bound)

    def update(self, mlp_output: torch.Tensor) -> None:
        super().update(self.bounded_mean(mlp_output))

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return self.bounded_mean(mlp_output)

    def as_deterministic_output_module(self) -> nn.Module:
        return _BoundedMean(self.bounded_mean.bound)
