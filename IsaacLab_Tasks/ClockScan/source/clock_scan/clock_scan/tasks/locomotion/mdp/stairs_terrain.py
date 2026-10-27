from __future__ import annotations

import numpy as np
import trimesh
from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass


def stair_count_and_height(difficulty: float, cfg: CountedStairsTerrainCfg) -> tuple[int, float]:
    """Difficulty split in len(step_counts) equal parts: the part sets the step count, the place inside the
    part sets the step height, from the bottom of step_height_range again at each new count."""
    parts = len(cfg.step_counts)
    i = min(int(difficulty * parts), parts - 1)
    sub = float(np.clip(difficulty * parts - i, 0.0, 1.0))
    lo, hi = cfg.step_height_range
    return int(cfg.step_counts[i]), lo + sub * (hi - lo)


def _ring(center: tuple[float, float], inner: float, outer: float, top: float, bottom: float) -> list[trimesh.Trimesh]:
    """Square ring between sides inner and outer (inner 0: a full square), from bottom to top."""
    cx, cy = center
    h = top - bottom
    z = 0.5 * (top + bottom)
    if inner <= 0.0:
        return [trimesh.creation.box((outer, outer, h), trimesh.transformations.translation_matrix((cx, cy, z)))]
    w = 0.5 * (outer - inner)
    d = 0.25 * (outer + inner)
    boxes = []
    for s in (-1.0, 1.0):
        boxes.append(trimesh.creation.box((outer, w, h), trimesh.transformations.translation_matrix((cx, cy + s * d, z))))
        boxes.append(trimesh.creation.box((w, inner, h), trimesh.transformations.translation_matrix((cx + s * d, cy, z))))
    return boxes


def counted_stairs_terrain(difficulty: float, cfg: CountedStairsTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Square stairs of n steps around a platform, flat ground beyond the last step. Not inverted: the
    platform is on top, n steps above the ground. Inverted: the platform is in a pit, n steps below it.
    The robot starts on the platform."""
    n, h = stair_count_and_height(difficulty, cfg)
    center = (0.5 * cfg.size[0], 0.5 * cfg.size[1])
    size = min(cfg.size)
    sign = -1.0 if cfg.inverted else 1.0
    bottom = min(0.0, sign * n * h) - 0.2
    meshes = []
    inner = 0.0
    for j in range(n + 1):
        outer = cfg.platform_width + 2.0 * j * cfg.step_width if j < n else size
        meshes += _ring(center, inner, outer, sign * (n - j) * h, bottom)
        inner = outer
    return meshes, np.array([center[0], center[1], sign * n * h])


@configclass
class CountedStairsTerrainCfg(SubTerrainBaseCfg):
    function: Callable = counted_stairs_terrain
    step_counts: tuple[int, ...] = MISSING
    step_height_range: tuple[float, float] = MISSING
    step_width: float = 0.3
    platform_width: float = 3.0
    inverted: bool = False
