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

    def __init__(self, cfg: UniformLevelVelocityCommandCfg, env):
        super().__init__(cfg, env)
        self._standby = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """The sampled command, held at zero until this episode's standby has elapsed.

        The sampled command itself is left alone in vel_command_b, so resampling, the slow and
        axis-only shares and the standing envs all behave exactly as before; only what the
        observation, the rewards and the metrics read is masked.
        """
        if self.cfg.standby_duration_range[1] <= 0.0:
            return self.vel_command_b
        waiting = self._env.episode_length_buf * self._env.step_dt < self._standby
        return self.vel_command_b * (~waiting).unsqueeze(1).float()

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        low, high = self.cfg.standby_duration_range
        if high > 0.0:
            ids = slice(None) if env_ids is None else env_ids
            self._standby[ids] = torch.empty_like(self._standby[ids]).uniform_(low, high)
        return extras

    def _update_metrics(self):
        command = self.command
        self._error_xy_sum += torch.linalg.norm(
            command[:, :2] - self.robot.data.root_lin_vel_b.torch[:, :2], dim=-1
        )
        self._error_yaw_sum += torch.abs(command[:, 2] - self.robot.data.root_ang_vel_b.torch[:, 2])
        self._step_count += 1.0

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)

        ids = torch.as_tensor(env_ids, device=self.device).reshape(-1)
        if ids.numel() == 0:
            return
        self._apply_axis_only(ids)
        self._apply_slow(ids)

    def _apply_axis_only(self, ids: torch.Tensor):
        """Zero the other two components for a share of the draws.

        A uniform box almost never produces a command that is ONE axis: asking for yaw with no
        forward motion has probability ~0 under independent draws, and the whole sweep the
        MuJoCo evaluation runs is made of exactly those commands. Measured on the arm that
        trained with foot torsion: at a 0.3 rad/s yaw command it turns at 0.24 rad/s while also
        walking at 0.5 m/s, and at 0.18 rad/s in place. Turning on the spot is a corner of the
        command space, and it is the corner an operator with a joystick spends time in.

        Applied before the slow rescale, which preserves direction, so the two compose: a draw
        can be both axis-only and slow.
        """
        fractions = (
            float(getattr(self.cfg, "x_only_command_fraction", 0.0)),
            float(getattr(self.cfg, "y_only_command_fraction", 0.0)),
            float(getattr(self.cfg, "yaw_only_command_fraction", 0.0)),
        )
        if sum(fractions) <= 0.0:
            return
        draw = torch.rand(ids.numel(), device=self.device)
        lower = 0.0
        for axis, fraction in enumerate(fractions):
            if fraction <= 0.0:
                continue
            upper = lower + fraction
            selected = (draw >= lower) & (draw < upper) & ~self.is_standing_env[ids]
            lower = upper
            if not bool(selected.any()):
                continue
            axis_ids = ids[selected]
            others = [i for i in range(3) if i != axis]
            self.vel_command_b[axis_ids[:, None], torch.tensor(others, device=self.device)] = 0.0

    def _apply_slow(self, ids: torch.Tensor):
        fraction = getattr(self.cfg, "slow_command_fraction", 0.0)
        if fraction <= 0.0:
            return
        if self.cfg.slow_after_full_range:
            if self.cfg.ranges.lin_vel_x[1] < self.cfg.limit_ranges.lin_vel_x[1] - 1e-6:
                return
            if not getattr(self, "_slow_announced", False):
                self._slow_announced = True
                print(f"[Command] level curriculum at full range; slow-command share {fraction} now active.")

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

    # Share of each resample reduced to a single axis, the other two zeroed. Disjoint shares,
    # taken in the order x, y, yaw. 0.0 reproduces upstream sampling exactly.
    x_only_command_fraction: float = 0.0
    y_only_command_fraction: float = 0.0
    yaw_only_command_fraction: float = 0.0

    # Seconds at zero command at the start of every episode, drawn per episode. The sampled
    # command is revealed when it runs out. (0, 0) disables.
    standby_duration_range: tuple[float, float] = (0.0, 0.0)
    # Hold the slow share off until the level curriculum has widened lin_vel_x to its limit.
    slow_after_full_range: bool = False
