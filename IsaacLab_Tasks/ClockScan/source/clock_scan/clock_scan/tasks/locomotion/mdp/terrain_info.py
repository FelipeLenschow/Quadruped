"""Per-env facts about the sub-terrain each env stands on, read from the terrain generator's columns."""

from __future__ import annotations

import math
import numpy as np
import torch

from .stairs_terrain import start_heading


def _column_cfgs(env) -> list:
    """The sub-terrain cfg of each terrain column, in the generator's column order."""
    cached = getattr(env, "_column_cfgs", None)
    if cached is None:
        gen = env.scene.terrain.cfg.terrain_generator
        subs = list(gen.sub_terrains.values())
        props = np.cumsum([s.proportion for s in subs]) / sum(s.proportion for s in subs)
        cached = [subs[int(np.min(np.where(i / gen.num_cols + 0.001 < props)[0]))] for i in range(gen.num_cols)]
        env._column_cfgs = cached
    return cached


def tile_heading(env, ids: torch.Tensor) -> torch.Tensor:
    """Yaw the robot starts at and is steered to on its tile (start_heading), NaN where any yaw goes."""
    terrain = env.scene.terrain
    if terrain.cfg.terrain_generator is None or getattr(terrain, "terrain_types", None) is None:
        return torch.full((len(ids),), math.nan, device=env.device)
    table = getattr(env, "_column_heading", None)
    if table is None:
        values = [start_heading(c) for c in _column_cfgs(env)]
        table = torch.tensor([math.nan if v is None else v for v in values], device=env.device)
        env._column_heading = table
    return table[terrain.terrain_types[ids]]
