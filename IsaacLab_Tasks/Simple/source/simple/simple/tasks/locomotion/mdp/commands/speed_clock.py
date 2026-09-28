from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from .velocity_command import foot_sweep_speed


GAIT_OFFSETS = {
    "trot": (0.5, 0.0, 0.0, 0.5),
    # Lateral-sequence walk, one foot at a time: RL, FL, RR, FR a quarter cycle apart.
    "walk": (0.25, 0.75, 0.5, 0.0),
    # Cheetah rotary gallop, touchdowns RH, LH, LF, RF at 0, 0.12, 0.48, 0.6 of the stride, shifted
    # so no foot is more than 0.31 of a cycle from its trot phase.
    "gallop": (0.81, 0.69, 0.17, 0.29),
}


def _heading_forward_speed(env) -> torch.Tensor:
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat

    robot = env.scene["robot"].data
    return quat_apply_inverse(yaw_quat(robot.root_quat_w.torch), robot.root_lin_vel_w.torch)[:, 0]


def _update_speed_metrics(term: CommandTerm, steps: torch.Tensor):
    """forward_speed: episode mean of the heading-frame forward speed (0.5 s low-pass); top_speed: its
    episode maximum; best_speed: the maximum over the whole run."""
    term._forward += min(term._env.step_dt / 0.5, 1.0) * (_heading_forward_speed(term._env) - term._forward)
    term.metrics["forward_speed"] += (term._forward - term.metrics["forward_speed"]) / steps
    term.metrics["top_speed"] = torch.maximum(term.metrics["top_speed"], term._forward)
    term._best = max(term._best, float(term.metrics["top_speed"].max()))
    term.metrics["best_speed"][:] = term._best


class SpeedLogCommand(CommandTerm):
    """No command, only the speed metrics of _update_speed_metrics, for tasks without a clock."""

    def __init__(self, cfg: CommandTermCfg, env):
        super().__init__(cfg, env)
        n = self.num_envs
        for name in ("forward_speed", "top_speed", "best_speed"):
            self.metrics[name] = torch.zeros(n, device=self.device)
        self._forward = torch.zeros(n, device=self.device)
        self._best = 0.0
        self._steps = torch.zeros(n, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._forward.unsqueeze(1)

    def _update_metrics(self):
        self._steps += 1.0
        _update_speed_metrics(self, self._steps)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._steps[ids] = 0.0
        self._forward[ids] = 0.0
        return extras

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        pass


@configclass
class SpeedLogCommandCfg(CommandTermCfg):
    class_type: type = SpeedLogCommand
    resampling_time_range: tuple[float, float] = (1e9, 1e9)


class SpeedClockCommand(CommandTerm):
    """Gait clock whose rate follows the velocity command.

    Stride grows with speed, L = stride_min + stride_gain * v, so the frequency is v / L. Swing
    time is fixed, so duty = 1 - swing_time * f. v is the sweep speed of the fastest stance foot
    under a low-passed velocity command, turning included. Each foot gets a phase in [0, 1):
    stance in [0, 0.5), swing in [0.5, 1).
    Below stand_threshold every foot is a stance foot and the clock waits with the first foot to
    swing at lift-off, so the first step comes as soon as a command does.

    With speed_source "measured", v is the robot's own speed instead of the command's.

    Metrics: forward_speed is the episode mean of the heading-frame forward speed (0.5 s low-pass),
    top_speed its episode maximum, best_speed the maximum over the whole run.

    With fixed_duty set, each foot's duty is that share of the cycle at every speed instead.

    With fast_offsets set, the offsets and per-foot swing times blend from foot_offsets and
    swing_time to fast_offsets and fast_swing_times as v goes from blend_speed[0] to [1].

    With integrate_phase set, each foot's phase is integrated at its stance or swing rate instead of
    mapped from the global index through the current duty. A changing duty then changes how fast a
    foot moves through its phase, not where it is, so a swing always takes its full swing time
    (the mapping cut the first swing after a standing start to 0.12 s). Stance rates are corrected
    by sync_gain times each foot's offset error against the others, so the gait keeps its offsets.
    On a stop, feet in swing finish it before the clock settles into the ready pose.
    """

    cfg: SpeedClockCommandCfg

    def __init__(self, cfg: SpeedClockCommandCfg, env):
        super().__init__(cfg, env)
        n = self.num_envs
        self.speed = torch.zeros(n, device=self.device)
        self.frequency = torch.full((n,), cfg.min_frequency, device=self.device)
        self._base_offsets = torch.tensor(cfg.foot_offsets, device=self.device)
        fast = cfg.fast_offsets if cfg.fast_offsets is not None else cfg.foot_offsets
        self._offset_delta = torch.remainder(torch.tensor(fast, device=self.device) - self._base_offsets + 0.5, 1.0) - 0.5
        self._base_swing = torch.full((4,), cfg.swing_time, device=self.device)
        fast_swing = cfg.fast_swing_times if cfg.fast_swing_times is not None else (cfg.swing_time,) * 4
        self._swing_delta = torch.tensor(fast_swing, device=self.device) - self._base_swing
        self._blend(self.speed)
        self.duty = torch.zeros(n, 4, device=self.device)
        self.gait_index = torch.zeros(n, device=self.device)
        self._set_timing()
        self.gait_index = self._ready_index()
        self.foot_phase = self._to_phase(torch.remainder(self.gait_index.unsqueeze(1) + self._offsets, 1.0))
        self.desired_contact = torch.ones(n, 4, device=self.device)
        self.metrics["contact_match"] = torch.zeros(n, device=self.device)
        self.metrics["frequency"] = torch.zeros(n, device=self.device)
        self.metrics["forward_speed"] = torch.zeros(n, device=self.device)
        self.metrics["top_speed"] = torch.zeros(n, device=self.device)
        self.metrics["best_speed"] = torch.zeros(n, device=self.device)
        self._forward = torch.zeros(n, device=self.device)
        self._best = 0.0
        self._match_steps = torch.zeros(n, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.frequency.unsqueeze(1), self.duty], dim=1)

    @property
    def clock(self) -> torch.Tensor:
        angle = 2.0 * math.pi * self.foot_phase
        return torch.cat([torch.sin(angle), torch.cos(angle)], dim=1)

    def _update_metrics(self):
        sensor = self._env.scene.sensors[self.cfg.sensor_name]
        if not hasattr(self, "_foot_ids"):
            self._foot_ids = [sensor.find_bodies(name)[0][0] for name in self.cfg.foot_names]
        in_contact = sensor.data.current_contact_time.torch[:, self._foot_ids] > 0
        match = (in_contact == (self.desired_contact > 0.5)).float().mean(dim=1)
        self._match_steps += 1.0
        self.metrics["contact_match"] += (match - self.metrics["contact_match"]) / self._match_steps
        self.metrics["frequency"] += (self.frequency - self.metrics["frequency"]) / self._match_steps
        _update_speed_metrics(self, self._match_steps)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._match_steps[ids] = 0.0
        self.speed[ids] = 0.0
        self._forward[ids] = 0.0
        if self.cfg.reset_speed is not None:
            self.speed[ids] = self.cfg.reset_speed
        elif self.cfg.reset_to_command:
            vel = self._env.command_manager.get_term(self.cfg.velocity_command_name).command[ids]
            self.speed[ids] = foot_sweep_speed(vel, self.cfg.foot_x, self.cfg.foot_y)
        self._blend(self.speed)
        self._set_timing()
        self.gait_index[ids] = self._ready_index()[ids]
        self.foot_phase[ids] = self._ready_phase()[ids]
        return extras

    def _blend(self, speed: torch.Tensor):
        lo, hi = self.cfg.blend_speed
        s = ((speed - lo) / max(hi - lo, 1e-6)).clamp(0.0, 1.0).unsqueeze(1)
        self._offsets = self._base_offsets + s * self._offset_delta
        self._swing = self._base_swing + s * self._swing_delta

    def _set_timing(self):
        cfg = self.cfg
        stride = cfg.stride_min + cfg.stride_gain * self.speed
        self.frequency = (self.speed / stride).clamp(cfg.min_frequency, cfg.max_frequency)
        if cfg.fixed_duty is not None:
            self.duty = torch.tensor(cfg.fixed_duty, device=self.device).expand(self.num_envs, 4).clone()
        else:
            self.duty = (1.0 - self._swing * self.frequency.unsqueeze(1)).clamp(cfg.min_duty, cfg.max_duty)

    def _ready_index(self) -> torch.Tensor:
        """Clock index with the foot that swings first exactly at lift-off."""
        lead = torch.argmax(self._offsets, dim=1, keepdim=True)
        return torch.remainder(
            self.duty.gather(1, lead).squeeze(1) - self._offsets.gather(1, lead).squeeze(1), 1.0
        )

    def _to_phase(self, raw: torch.Tensor) -> torch.Tensor:
        d = self.duty
        return torch.where(raw < d, raw * (0.5 / d), 0.5 + (raw - d) * (0.5 / (1.0 - d)))

    def _to_raw(self, phase: torch.Tensor) -> torch.Tensor:
        d = self.duty
        return torch.where(phase < 0.5, phase * (2.0 * d), d + (phase - 0.5) * (2.0 * (1.0 - d)))

    def _ready_phase(self) -> torch.Tensor:
        return self._to_phase(torch.remainder(self._ready_index().unsqueeze(1) + self._offsets, 1.0))

    def _integrate(self, standing: torch.Tensor, dt: float):
        phase = self.foot_phase
        d = self.duty
        f = self.frequency.unsqueeze(1)
        raw = self._to_raw(phase)
        angle = 2.0 * math.pi * (raw - self._offsets)
        index = torch.atan2(torch.sin(angle).mean(dim=1), torch.cos(angle).mean(dim=1)) / (2.0 * math.pi)
        error = torch.remainder(index.unsqueeze(1) + self._offsets - raw + 0.5, 1.0) - 0.5
        sync = (1.0 + self.cfg.sync_gain * error).clamp(0.5, 1.5)
        swinging = phase > 0.5 + 1e-4
        rate = torch.where(swinging, 0.5 * f / (1.0 - d), 0.5 * f / d * sync)
        rate = torch.where(standing.unsqueeze(1) & ~swinging, torch.zeros_like(rate), rate)
        phase = torch.remainder(phase + rate * dt, 1.0)
        settled = standing & ~(phase > 0.5 + 1e-4).any(dim=1)
        self.foot_phase = torch.where(settled.unsqueeze(1), self._ready_phase(), phase)
        self.gait_index = torch.remainder(index, 1.0)

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        cfg = self.cfg
        dt = self._env.step_dt
        vel = self._env.command_manager.get_command(cfg.velocity_command_name)
        if cfg.speed_source == "measured":
            robot = self._env.scene["robot"].data
            twist = torch.cat([robot.root_lin_vel_b.torch[:, :2], robot.root_ang_vel_b.torch[:, 2:3]], dim=1)
            sweep = foot_sweep_speed(twist, cfg.foot_x, cfg.foot_y)
        else:
            sweep = foot_sweep_speed(vel, cfg.foot_x, cfg.foot_y)
        self.speed += min(dt / cfg.filter_time, 1.0) * (sweep - self.speed)
        self._blend(self.speed)
        self._set_timing()
        standing = torch.norm(vel, dim=1) < cfg.stand_threshold
        if cfg.integrate_phase:
            self._integrate(standing, dt)
            planted = standing.unsqueeze(1) & (self.foot_phase <= 0.5 + 1e-4)
        else:
            running = torch.remainder(self.gait_index + dt * self.frequency, 1.0)
            self.gait_index = torch.where(standing, self._ready_index(), running)
            self.foot_phase = self._to_phase(torch.remainder(self.gait_index.unsqueeze(1) + self._offsets, 1.0))
            planted = standing.unsqueeze(1).expand(-1, 4)
        normal = torch.distributions.Normal(0.0, cfg.contact_smoothing)
        p = self.foot_phase
        contact = normal.cdf(p) * (1 - normal.cdf(p - 0.5)) + normal.cdf(p - 1) * (1 - normal.cdf(p - 1.5))
        self.desired_contact = torch.where(planted, torch.ones_like(contact), contact)


@configclass
class SpeedClockCommandCfg(CommandTermCfg):
    class_type: type = SpeedClockCommand

    resampling_time_range: tuple[float, float] = (1e9, 1e9)
    velocity_command_name: str = "base_velocity"
    sensor_name: str = "contact_forces"
    foot_names: tuple[str, ...] = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
    foot_offsets: tuple[float, float, float, float] = GAIT_OFFSETS["trot"]
    fast_offsets: tuple[float, float, float, float] | None = None
    fast_swing_times: tuple[float, float, float, float] | None = None
    blend_speed: tuple[float, float] = (0.0, 0.0)
    fixed_duty: tuple[float, float, float, float] | None = None
    # Start each episode's filtered speed at the command's instead of at zero.
    reset_to_command: bool = False
    # Or at this speed, m/s.
    reset_speed: float | None = None
    # "command": v is the velocity command's. "measured": the robot's own base twist.
    speed_source: str = "command"

    stride_min: float = 0.10
    stride_gain: float = 0.3
    swing_time: float = 0.2
    min_frequency: float = 0.4
    max_frequency: float = 3.0
    min_duty: float = 0.35
    max_duty: float = 0.95
    filter_time: float = 0.25
    foot_x: float = 0.19
    foot_y: float = 0.14
    stand_threshold: float = 0.02
    contact_smoothing: float = 0.07
    integrate_phase: bool = False
    sync_gain: float = 4.0
