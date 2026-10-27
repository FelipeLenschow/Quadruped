from __future__ import annotations

import torch
from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.sensors.ray_caster.patterns import PatternBaseCfg
from isaaclab.utils import configclass


def footprint_pattern(cfg: FootprintPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Parallel rays straight down on the grid x_points by y_points (sensor frame, m), x outer, y inner."""
    x = torch.tensor(cfg.x_points, device=device)
    y = torch.tensor(cfg.y_points, device=device)
    gx, gy = torch.meshgrid(x, y, indexing="ij")
    starts = torch.stack([gx, gy, torch.zeros_like(gx)], dim=-1).reshape(-1, 3)
    directions = torch.zeros_like(starts)
    directions[:, 2] = -1.0
    return starts, directions


@configclass
class FootprintPatternCfg(PatternBaseCfg):
    func: Callable = footprint_pattern
    x_points: tuple[float, ...] = MISSING
    y_points: tuple[float, ...] = MISSING
