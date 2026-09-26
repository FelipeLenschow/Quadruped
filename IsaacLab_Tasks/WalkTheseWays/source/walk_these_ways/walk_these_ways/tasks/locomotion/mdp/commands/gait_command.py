from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from dataclasses import MISSING

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

# Walk These Ways phase / offset / bound per gait, feet in FL, FR, RL, RR order.
GAITS = {
    "trot": (0.5, 0.0, 0.0),
    "pace": (0.0, 0.0, 0.5),
    "bound": (0.0, 0.5, 0.0),
    "pronk": (0.0, 0.0, 0.0),
    # Lateral-sequence walk, one foot at a time: RL, FL, RR, FR a quarter cycle apart.
    "walk": (0.0, 0.75, 0.5),
}

# Columns of GaitCommand.command.
HEIGHT, FREQ, PHASE, OFFSET, BOUND, SWING, PITCH, WIDTH, DUTY = range(9)


class GaitCommand(CommandTerm):
    """Walk These Ways behaviour command and gait clock.

    The command is [body height offset, step frequency, phase, offset, bound, swing height,
    body pitch, stance width, duty]. The clock advances at the commanded frequency and gives each
    foot a phase in [0, 1): stance in [0, 0.5), swing in [0.5, 1), with stance taking the duty
    share of the step's time.

    A stance foot can sweep at most ``max_stride``, so the velocity command is held to
    max_stride * frequency / duty: slow steps only ever come with slow commands. With the
    curriculum the frequency is drawn above what the sampled command needs; the per-step clamp
    covers everything else (play, no curriculum). When the velocity command is zero
    every foot is a stance foot, so the robot stands instead of marching in place. During the
    velocity command's standby the posture commands read nominal.

    With ``curriculum`` on, this term also samples the velocity command, from a per-gait grid over
    (forward speed, yaw rate) as in Walk These Ways: a cell is sampled by its weight, and when a
    command drawn from it is tracked well (all four task terms over their thresholds) the cell and
    its neighbours gain weight, so each gait widens its own command range.
    """

    cfg: GaitCommandCfg

    def __init__(self, cfg: GaitCommandCfg, env):
        super().__init__(cfg, env)
        n = self.num_envs
        self._command = torch.zeros(n, 9, device=self.device)
        self.gait_index = torch.zeros(n, device=self.device)
        self.foot_phase = torch.zeros(n, 4, device=self.device)
        self.desired_contact = torch.ones(n, 4, device=self.device)
        self._gait_table = torch.tensor([GAITS[g] for g in cfg.gait_names], device=self.device)
        self._gait_probs = torch.tensor(cfg.gait_probs, device=self.device)
        self._nominal = torch.tensor(
            [
                0.0,
                cfg.nominal_frequency,
                *GAITS["trot"],
                cfg.nominal_swing_height,
                0.0,
                cfg.nominal_stance_width,
                cfg.nominal_duty,
            ],
            device=self.device,
        )
        self.metrics["contact_match"] = torch.zeros(n, device=self.device)
        self._match_steps = torch.zeros(n, device=self.device)
        self._grid = None
        if cfg.curriculum:
            for g in cfg.gait_names:
                self.metrics[f"cells_{g}"] = torch.zeros(n, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        velocity = self._env.command_manager.get_term(self.cfg.velocity_command_name)
        waiting = self._env.episode_length_buf * self._env.step_dt < velocity._standby
        return torch.where(waiting.unsqueeze(1), self._nominal, self._command)

    @property
    def standing(self) -> torch.Tensor:
        velocity = self._env.command_manager.get_command(self.cfg.velocity_command_name)
        return torch.norm(velocity, dim=1) < self.cfg.stand_threshold

    @property
    def clock(self) -> torch.Tensor:
        return torch.sin(2.0 * math.pi * self.foot_phase)

    def _sweep_speed(self, velocity: torch.Tensor) -> torch.Tensor:
        """How fast a stance foot moves relative to the body for a (vx, vy, yaw) command."""
        return torch.norm(velocity[:, :2], dim=1) + velocity[:, 2].abs() * self.cfg.yaw_radius

    def _init_curriculum(self):
        velocity = self._env.command_manager.get_term(self.cfg.velocity_command_name)
        limits = velocity.cfg.limit_ranges
        nb = self.cfg.curriculum_bins
        self._x_edges = torch.linspace(*limits.lin_vel_x, nb + 1, device=self.device)
        self._z_edges = torch.linspace(*limits.ang_vel_z, nb + 1, device=self.device)
        x_mid = (self._x_edges[:-1] + self._x_edges[1:]) / 2
        z_mid = (self._z_edges[:-1] + self._z_edges[1:]) / 2
        start = (x_mid.abs() <= self.cfg.curriculum_init[0]).unsqueeze(1) & (
            z_mid.abs() <= self.cfg.curriculum_init[1]
        ).unsqueeze(0)
        self._grid = start.float().unsqueeze(0).repeat(len(self.cfg.gait_names), 1, 1)
        rm = self._env.reward_manager
        self._task_terms = [
            (rm.active_terms.index(name), rm.get_term_cfg(name).weight) for name in self.cfg.curriculum_terms
        ]
        n = self.num_envs
        self._task_sum = torch.zeros(n, len(self._task_terms), device=self.device)
        self._task_steps = torch.zeros(n, device=self.device)
        self._env_gait = torch.zeros(n, dtype=torch.long, device=self.device)
        self._env_cell = torch.zeros(n, 2, dtype=torch.long, device=self.device)
        self._env_graded = torch.zeros(n, dtype=torch.bool, device=self.device)

    def _grade(self, ids: torch.Tensor):
        steps = self._task_steps[ids]
        graded = self._env_graded[ids] & (steps > 0)
        mean = self._task_sum[ids] / steps.clamp(min=1).unsqueeze(1)
        thresholds = torch.tensor(self.cfg.curriculum_thresholds, device=self.device)
        offsets = torch.tensor(self.cfg.curriculum_offsets, device=self.device)
        ok = graded & ((mean + offsets) > thresholds).all(dim=1)
        self._task_sum[ids] = 0.0
        self._task_steps[ids] = 0.0
        if not bool(ok.any()):
            return
        g = self._env_gait[ids][ok]
        cell = self._env_cell[ids][ok]
        r = self.cfg.curriculum_local
        span = torch.arange(-r, r + 1, device=self.device)
        dx, dz = torch.meshgrid(span, span, indexing="ij")
        nb = self.cfg.curriculum_bins
        xs = (cell[:, 0:1] + dx.reshape(1, -1)).clamp(0, nb - 1)
        zs = (cell[:, 1:2] + dz.reshape(1, -1)).clamp(0, nb - 1)
        gs = g.unsqueeze(1).expand_as(xs)
        bump = torch.full(xs.shape, 0.2, device=self.device)
        self._grid.index_put_((gs.reshape(-1), xs.reshape(-1), zs.reshape(-1)), bump.reshape(-1), accumulate=True)
        self._grid.clamp_(0.0, 1.0)

    def _sample_velocity(self, ids: torch.Tensor, gait: torch.Tensor):
        velocity = self._env.command_manager.get_term(self.cfg.velocity_command_name)
        k = ids.numel()
        nb = self.cfg.curriculum_bins
        cell = torch.multinomial(self._grid[gait].reshape(k, -1), 1).squeeze(1)
        xi, zi = cell // nb, cell % nb
        vx = self._x_edges[xi] + torch.rand(k, device=self.device) * (self._x_edges[1] - self._x_edges[0])
        wz = self._z_edges[zi] + torch.rand(k, device=self.device) * (self._z_edges[1] - self._z_edges[0])
        vy = torch.empty(k, device=self.device).uniform_(*velocity.cfg.ranges.lin_vel_y)
        velocity.vel_command_b[ids] = torch.stack([vx, vy, wz], dim=1)
        velocity.is_standing_env[ids] = torch.rand(k, device=self.device) <= velocity.cfg.rel_standing_envs
        velocity._apply_axis_only(ids)
        velocity._apply_slow(ids)
        final = velocity.vel_command_b[ids]
        self._env_gait[ids] = gait
        self._env_cell[ids, 0] = (torch.bucketize(final[:, 0].contiguous(), self._x_edges) - 1).clamp(0, nb - 1)
        self._env_cell[ids, 1] = (torch.bucketize(final[:, 2].contiguous(), self._z_edges) - 1).clamp(0, nb - 1)
        self._env_graded[ids] = ~velocity.is_standing_env[ids]
        for i, name in enumerate(self.cfg.gait_names):
            self.metrics[f"cells_{name}"][:] = (self._grid[i] > 0).float().mean()

    def _update_metrics(self):
        if self._grid is not None:
            values = self._env.reward_manager._step_reward
            for i, (column, weight) in enumerate(self._task_terms):
                self._task_sum[:, i] += values[:, column] / weight
            self._task_steps += 1.0
        sensor = self._env.scene.sensors[self.cfg.sensor_name]
        if not hasattr(self, "_foot_ids"):
            self._foot_ids = [sensor.find_bodies(name)[0][0] for name in self.cfg.foot_names]
        in_contact = sensor.data.current_contact_time.torch[:, self._foot_ids] > 0
        match = (in_contact == (self.desired_contact > 0.5)).float().mean(dim=1)
        self._match_steps += 1.0
        self.metrics["contact_match"] += (match - self.metrics["contact_match"]) / self._match_steps

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        self._match_steps[slice(None) if env_ids is None else env_ids] = 0.0
        return extras

    def _resample_command(self, env_ids: Sequence[int]):
        ids = torch.as_tensor(env_ids, device=self.device).reshape(-1)
        k = ids.numel()
        if k == 0:
            return

        def uniform(bounds):
            return torch.empty(k, device=self.device).uniform_(*bounds)

        cmd = torch.empty(k, 9, device=self.device)
        cmd[:, HEIGHT] = uniform(self.cfg.ranges.body_height)
        cmd[:, FREQ] = uniform(self.cfg.ranges.frequency)
        gait = torch.multinomial(self._gait_probs, k, replacement=True)
        cmd[:, PHASE:BOUND + 1] = self._gait_table[gait]
        cmd[:, SWING] = uniform(self.cfg.ranges.swing_height)
        cmd[:, PITCH] = uniform(self.cfg.ranges.body_pitch)
        cmd[:, WIDTH] = uniform(self.cfg.ranges.stance_width)
        cmd[:, DUTY] = uniform(self.cfg.ranges.duty)
        nominal = torch.rand(k, device=self.device) < self.cfg.nominal_fraction
        cmd[nominal] = self._nominal
        self._command[ids] = cmd
        if self.cfg.curriculum:
            if self._grid is None:
                self._init_curriculum()
            self._grade(ids)
            gait = torch.where(nominal, self.cfg.gait_names.index("trot"), gait)
            self._sample_velocity(ids, gait)
            velocity = self._env.command_manager.get_term(self.cfg.velocity_command_name)
            low, high = self.cfg.ranges.frequency
            needed = self._sweep_speed(velocity.vel_command_b[ids]) * cmd[:, DUTY] / self.cfg.max_stride
            floor = needed.clamp(low, high)
            freq = floor + torch.rand(k, device=self.device) * (high - floor)
            self._command[ids, FREQ] = torch.where(nominal, cmd[:, FREQ], freq)

    def _update_command(self):
        velocity = self._env.command_manager.get_term(self.cfg.velocity_command_name)
        limit = self.cfg.max_stride * self._command[:, FREQ] / self._command[:, DUTY]
        scale = (limit / self._sweep_speed(velocity.vel_command_b).clamp(min=1e-6)).clamp(max=1.0)
        velocity.vel_command_b[:] *= scale.unsqueeze(1)
        cmd = self.command
        self.gait_index = torch.remainder(self.gait_index + self._env.step_dt * cmd[:, FREQ], 1.0)
        g = self.gait_index
        phase, offset, bound = cmd[:, PHASE], cmd[:, OFFSET], cmd[:, BOUND]
        raw = torch.stack([g + phase + offset + bound, g + offset, g + bound, g + phase], dim=1)
        raw = torch.remainder(raw, 1.0)
        duration = cmd[:, DUTY:DUTY + 1]
        stance = raw < duration
        self.foot_phase = torch.where(
            stance, raw * (0.5 / duration), 0.5 + (raw - duration) * (0.5 / (1.0 - duration))
        )
        normal = torch.distributions.Normal(0.0, self.cfg.contact_smoothing)
        p = self.foot_phase
        contact = normal.cdf(p) * (1 - normal.cdf(p - 0.5)) + normal.cdf(p - 1) * (1 - normal.cdf(p - 1.5))
        self.desired_contact = torch.where(self.standing.unsqueeze(1), torch.ones_like(contact), contact)


@configclass
class GaitCommandCfg(CommandTermCfg):
    class_type: type = GaitCommand

    velocity_command_name: str = "base_velocity"
    sensor_name: str = "contact_forces"
    foot_names: tuple[str, ...] = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")

    @configclass
    class Ranges:
        body_height: tuple[float, float] = MISSING
        frequency: tuple[float, float] = MISSING
        swing_height: tuple[float, float] = MISSING
        body_pitch: tuple[float, float] = MISSING
        stance_width: tuple[float, float] = MISSING
        duty: tuple[float, float] = MISSING

    ranges: Ranges = MISSING
    gait_names: tuple[str, ...] = ("trot", "pace", "bound", "pronk", "walk")
    gait_probs: tuple[float, ...] = (0.3, 0.15, 0.15, 0.15, 0.25)
    # Share of resamples held at the nominal trot and posture, so the default gait stays good.
    nominal_fraction: float = 0.2
    nominal_frequency: float = 3.0
    nominal_swing_height: float = 0.08
    nominal_stance_width: float = 0.3
    nominal_duty: float = 0.5
    # Longest stance sweep a foot can make, m, and the lever arm that turns yaw rate into it.
    max_stride: float = 0.3
    yaw_radius: float = 0.3
    contact_smoothing: float = 0.07
    stand_threshold: float = 0.01

    # Walk These Ways' per-gait command curriculum over (forward speed, yaw rate). WTW starts at
    # +-1 and grows to +-5; here the limit is the velocity command's limit_ranges, so it starts
    # at curriculum_init and grows from there. Thresholds are WTW's, on the per-step mean of each
    # term's unweighted value plus offset (the contact terms are in [-1, 0], so offset 1).
    curriculum: bool = False
    curriculum_bins: int = 11
    curriculum_init: tuple[float, float] = (0.2, 0.2)
    curriculum_local: int = 1
    curriculum_terms: tuple[str, ...] = ("track_lin_vel_xy", "track_ang_vel_z", "gait_contact_force", "gait_contact_vel")
    curriculum_thresholds: tuple[float, ...] = (0.8, 0.5, 0.8, 0.8)
    curriculum_offsets: tuple[float, ...] = (0.0, 0.0, 1.0, 1.0)
