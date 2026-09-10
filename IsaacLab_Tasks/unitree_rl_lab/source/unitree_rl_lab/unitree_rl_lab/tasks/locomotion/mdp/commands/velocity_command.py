from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


class UniformLevelVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command with an explicit slow-command quota.

    Sampling velocity commands from a uniform BOX gives almost no low-speed coverage. The
    magnitude of a uniform draw over a cube concentrates near the corners (the r^2 volume
    effect), so with ranges of +-1.0 only ~1.4% of commands come out below ||c|| = 0.3 m/s,
    while ~79% land above 0.5. The band where the robot has to choose between holding a stance
    and taking one slow step is therefore almost never trained, and whatever it does there is
    nearly free in the aggregate return.

    That is where the low-speed dead zone lives, and the level curriculum makes it worse rather
    than better: it starts at +-0.1 (where every command is slow), then widens to the limits as
    soon as tracking is good enough -- so the slow band is covered only while the policy is too
    poor to walk, and abandoned as soon as it can.

    ``slow_command_fraction`` of each resample keeps the DIRECTION just drawn and rescales it to
    a magnitude drawn from ``slow_command_range``. Keeping the direction rather than resampling
    it means the mode covers slow walks, slow slides and slow turns in the same proportion as
    the main distribution. The magnitude is the 3-norm of (vx, vy, wz), the same quantity the
    standing/moving logic elsewhere compares against, so the range is stated in the units of
    that gate.

    Standing envs (``rel_standing_envs``) are left alone: "hold still" and "move slowly" are
    different instructions, and mixing them would make neither measurable.
    """

    cfg: UniformLevelVelocityCommandCfg

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)

        fraction = getattr(self.cfg, "slow_command_fraction", 0.0)
        if fraction <= 0.0:
            return

        ids = torch.as_tensor(env_ids, device=self.device).reshape(-1)
        if ids.numel() == 0:
            return

        draw = torch.rand(ids.numel(), device=self.device)
        # Exclude the standing envs, so the two modes stay disjoint and each stays interpretable.
        selected = (draw < fraction) & ~self.is_standing_env[ids]
        if not bool(selected.any()):
            return

        slow_ids = ids[selected]
        direction = self.vel_command_b[slow_ids, :3]
        # clamp guards the (vanishingly rare) near-zero draw from blowing up the rescale
        norm = torch.norm(direction, dim=1, keepdim=True).clamp(min=1e-6)
        low, high = self.cfg.slow_command_range
        magnitude = torch.empty(slow_ids.numel(), 1, device=self.device).uniform_(low, high)
        self.vel_command_b[slow_ids, :3] = direction / norm * magnitude


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = UniformLevelVelocityCommand

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING

    # Fraction of each resample rescaled to a low ||(vx, vy, wz)|| drawn from
    # slow_command_range. 0.0 reproduces upstream sampling exactly.
    slow_command_fraction: float = 0.0
    slow_command_range: tuple[float, float] = (0.05, 0.3)
