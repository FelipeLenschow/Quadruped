from __future__ import annotations

import numpy as np
import trimesh
from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass


def _block_and_height(difficulty: float, blocks: tuple, height_range: tuple[float, float]) -> tuple:
    """Difficulty split in len(blocks) equal parts: the part picks the block value, the place inside the part
    sets the step height, from the bottom of height_range again at each new block."""
    parts = len(blocks)
    i = min(int(difficulty * parts), parts - 1)
    sub = float(np.clip(difficulty * parts - i, 0.0, 1.0))
    lo, hi = height_range
    return blocks[i], lo + sub * (hi - lo)


def stair_count_and_height(difficulty: float, cfg: CountedStairsTerrainCfg) -> tuple[int, float]:
    """The step count from the block, the step height from the place inside it."""
    n, h = _block_and_height(difficulty, cfg.step_counts, cfg.step_height_range)
    return int(n), h


def _box(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> trimesh.Trimesh:
    return trimesh.creation.box(
        (x1 - x0, y1 - y0, z1 - z0),
        trimesh.transformations.translation_matrix((0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1))),
    )


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


def narrow_stairs_terrain(difficulty: float, cfg: NarrowStairsTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Two floors joined by one straight flight of num_steps steps along +x, in the middle of the tile in y. The
    width comes from the block of widths, the step height from the place inside it. Each side of the flight, and
    the upper floor's edge on that side of the stairwell, is open, has balusters with a handrail, or a wall
    (side_probs). The upper floor has a wall on the tile edges. Inverted: the robot starts on the lower floor
    facing the flight; not inverted: on the upper floor facing down it."""
    w, h = _block_and_height(difficulty, cfg.widths, cfg.step_height_range)
    length, width = cfg.size
    c = 0.5 * width
    n, d = cfg.num_steps, cfg.step_width
    x0 = cfg.approach
    x1 = x0 + n * d
    top = n * h
    t = cfg.wall_thickness
    meshes = [_box(0.0, length, 0.0, width, -0.2, 0.0), _box(x1, length, 0.0, width, 0.0, top)]
    for k in range(n):
        meshes.append(_box(x0 + k * d, x0 + (k + 1) * d, c - 0.5 * w, c + 0.5 * w, 0.0, (k + 1) * h))
    meshes += [
        _box(length - t, length, 0.0, width, top, top + cfg.wall_height),
        _box(x1, length - t, 0.0, t, top, top + cfg.wall_height),
        _box(x1, length - t, width - t, width, top, top + cfg.wall_height),
    ]
    p = cfg.baluster_size
    for s in (-1.0, 1.0):
        kind = np.random.choice(3, p=np.asarray(cfg.side_probs) / np.sum(cfg.side_probs))
        inner = c + s * 0.5 * w
        if kind == 2:
            y0, y1 = sorted((inner, inner + s * t))
            meshes.append(_box(x0, x1, y0, y1, 0.0, top + cfg.wall_height))
            y0, y1 = sorted((inner, width - t if s > 0 else t))
            meshes.append(_box(x1, x1 + t, y0, y1, top, top + cfg.wall_height))
        elif kind == 1:
            yp = inner + s * 0.5 * p
            for x in np.arange(x0 + 0.5 * cfg.baluster_spacing, x1, cfg.baluster_spacing):
                z = (min(int((x - x0) / d), n - 1) + 1) * h
                meshes.append(_box(x - 0.5 * p, x + 0.5 * p, yp - 0.5 * p, yp + 0.5 * p, z, z + cfg.rail_height))
            run = np.hypot(n * d, top)
            rail = trimesh.creation.box((run, p, p))
            rail.apply_transform(trimesh.transformations.rotation_matrix(-np.arctan2(top, n * d), (0.0, 1.0, 0.0)))
            rail.apply_translation((0.5 * (x0 + x1), yp, 0.5 * top + cfg.rail_height))
            meshes.append(rail)
            end = width - t if s > 0 else t
            xp = x1 + 0.5 * p
            for y in np.arange(yp, end, s * cfg.baluster_spacing):
                meshes.append(_box(xp - 0.5 * p, xp + 0.5 * p, y - 0.5 * p, y + 0.5 * p, top, top + cfg.rail_height))
            y0, y1 = sorted((yp, end))
            meshes.append(_box(x1, x1 + p, y0, y1, top + cfg.rail_height - p, top + cfg.rail_height))
    if cfg.inverted:
        origin = (x0 - cfg.start_gap, c, 0.0)
    else:
        origin = (x1 + cfg.start_gap, c, top)
    return meshes, np.array(origin)


def start_heading(cfg: SubTerrainBaseCfg) -> float | None:
    """Yaw a robot starts at and is steered to on this sub-terrain, or None for any."""
    if isinstance(cfg, NarrowStairsTerrainCfg):
        return 0.0 if cfg.inverted else float(np.pi)
    return None


@configclass
class NarrowStairsTerrainCfg(SubTerrainBaseCfg):
    function: Callable = narrow_stairs_terrain
    widths: tuple[float, ...] = MISSING
    step_height_range: tuple[float, float] = MISSING
    num_steps: int = 15
    step_width: float = 0.3
    approach: float = 1.5
    start_gap: float = 0.6
    side_probs: tuple[float, float, float] = (0.4, 0.4, 0.2)
    """Open, balusters, wall."""
    baluster_spacing: float = 0.12
    baluster_size: float = 0.03
    rail_height: float = 0.9
    wall_height: float = 1.0
    wall_thickness: float = 0.1
    inverted: bool = False
