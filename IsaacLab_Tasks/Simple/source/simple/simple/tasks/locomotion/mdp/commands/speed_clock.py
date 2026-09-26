from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass


class SpeedClockCommand(CommandTerm):
    """Trot clock whose rate follows the velocity command.

    Stride grows with speed, L = stride_min + stride_gain * v, so the frequency is v / L. Swing
    time is fixed, so duty = 1 - swing_time * f. v is the stance-foot sweep speed of a low-passed
    velocity command. Each foot gets a phase in [0, 1): stance in [0, 0.5), swing in [0.5, 1).
    Below stand_threshold every foot is a stance foot and the clock waits with the FL/RR pair at
    lift-off, so the first step comes as soon as a command does.
    """

    cfg: SpeedClockCommandCfg

    def __init__(self, cfg: SpeedClockCommandCfg, env):
        super().__init__(cfg, env)
        n = self.num_envs
        self.speed = torch.zeros(n, device=self.device)
        self.frequency = torch.full((n,), cfg.min_frequency, device=self.device)
        self.duty = 1.0 - cfg.swing_time * self.frequency
        self.gait_index = torch.remainder(self.duty - 0.5, 1.0)
        self.foot_phase = torch.zeros(n, 4, device=self.device)
        self.desired_contact = torch.ones(n, 4, device=self.device)
        self._offsets = torch.tensor([0.5, 0.0, 0.0, 0.5], device=self.device)
        self.metrics["contact_match"] = torch.zeros(n, device=self.device)
        self.metrics["frequency"] = torch.zeros(n, device=self.device)
        self._match_steps = torch.zeros(n, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return torch.stack([self.frequency, self.duty], dim=1)

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

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._match_steps[ids] = 0.0
        self.speed[ids] = 0.0
        self.frequency[ids] = self.cfg.min_frequency
        self.duty[ids] = 1.0 - self.cfg.swing_time * self.cfg.min_frequency
        self.gait_index[ids] = torch.remainder(self.duty[ids] - 0.5, 1.0)
        return extras

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        cfg = self.cfg
        dt = self._env.step_dt
        vel = self._env.command_manager.get_command(cfg.velocity_command_name)
        sweep = torch.norm(vel[:, :2], dim=1) + vel[:, 2].abs() * cfg.yaw_radius
        self.speed += min(dt / cfg.filter_time, 1.0) * (sweep - self.speed)
        stride = cfg.stride_min + cfg.stride_gain * self.speed
        self.frequency = (self.speed / stride).clamp(cfg.min_frequency, cfg.max_frequency)
        self.duty = (1.0 - cfg.swing_time * self.frequency).clamp(cfg.min_duty, cfg.max_duty)
        standing = torch.norm(vel, dim=1) < cfg.stand_threshold
        running = torch.remainder(self.gait_index + dt * self.frequency, 1.0)
        self.gait_index = torch.where(standing, torch.remainder(self.duty - 0.5, 1.0), running)

        raw = torch.remainder(self.gait_index.unsqueeze(1) + self._offsets, 1.0)
        duty = self.duty.unsqueeze(1)
        self.foot_phase = torch.where(raw < duty, raw * (0.5 / duty), 0.5 + (raw - duty) * (0.5 / (1.0 - duty)))
        normal = torch.distributions.Normal(0.0, cfg.contact_smoothing)
        p = self.foot_phase
        contact = normal.cdf(p) * (1 - normal.cdf(p - 0.5)) + normal.cdf(p - 1) * (1 - normal.cdf(p - 1.5))
        self.desired_contact = torch.where(standing.unsqueeze(1), torch.ones_like(contact), contact)


@configclass
class SpeedClockCommandCfg(CommandTermCfg):
    class_type: type = SpeedClockCommand

    resampling_time_range: tuple[float, float] = (1e9, 1e9)
    velocity_command_name: str = "base_velocity"
    sensor_name: str = "contact_forces"
    foot_names: tuple[str, ...] = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")

    stride_min: float = 0.10
    stride_gain: float = 0.3
    swing_time: float = 0.2
    min_frequency: float = 0.4
    max_frequency: float = 3.0
    min_duty: float = 0.35
    max_duty: float = 0.95
    filter_time: float = 0.25
    yaw_radius: float = 0.3
    stand_threshold: float = 0.02
    contact_smoothing: float = 0.07
