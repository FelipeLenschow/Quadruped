from __future__ import annotations

import math
import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class TrickCommand(CommandTerm):
    """One on/off switch per variant of the trick, in the 3-wide velocity slot of unitree's layout, zero-padded.
    At most one is on: do that variant; all off: stand.

    With clock_hz (one switch only), the other two are sin and cos of a phase that starts at 0 when the trick
    goes on and advances clock_hz turns a second while it stays on; zero while off. The pipeline makes the
    same clock on the robot.

    Each episode starts off, then alternates on and off segments; each on segment picks a switch at random.
    A reset event can put env.trick_start_switch[i] = k (-1 for none): that episode then starts with switch k
    on, with probability start_on_prob.
    The base heading and position when a switch goes on are kept in anchor_yaw and anchor_xy.

    With flip_s, the first flip_s seconds of each on segment are the flip (flipping); `rotation` adds up the
    backward pitch (rad, about the body y axis) in them. After them, and while off, the robot stands.
    """

    cfg: TrickCommandCfg

    def __init__(self, cfg: TrickCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.trick_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.anchor_yaw = torch.zeros(self.num_envs, device=self.device)
        self.anchor_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.on_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.on_time = torch.zeros(self.num_envs, device=self.device)
        self.rotation = torch.zeros(self.num_envs, device=self.device)
        if cfg.clock_hz and cfg.switches != 1:
            raise ValueError("TrickCommand: clock_hz needs the two spare slots, so a single switch")

    def __str__(self) -> str:
        return (f"TrickCommand: {self.cfg.switches} switch(es), on {self.cfg.on_time_range} s,"
                f" off {self.cfg.off_time_range} s, first off {self.cfg.first_off_range} s")

    @property
    def command(self) -> torch.Tensor:
        return self.trick_command

    @property
    def on(self) -> torch.Tensor:
        return self.trick_command[:, : self.cfg.switches].sum(dim=1).clamp(max=1.0)

    @property
    def flipping(self) -> torch.Tensor:
        return self.on * (self.on_time < self.cfg.flip_s).float()

    def _resample(self, env_ids):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        env_ids = torch.as_tensor(env_ids, device=self.device)
        if len(env_ids) == 0:
            return
        first = self.command_counter[env_ids] == 0
        turn_on = ~first & (self.on[env_ids] < 0.5)
        start = getattr(self._env, "trick_start_switch", None)
        start = start[env_ids] if start is not None else torch.full_like(env_ids, -1)
        start_on = first & (start >= 0) & (torch.rand(len(env_ids), device=self.device) < self.cfg.start_on_prob)
        turn_on = turn_on | start_on
        first = first & ~start_on
        self.trick_command[env_ids] = 0.0
        on_ids = env_ids[turn_on]
        if self.cfg.cycle_switches:
            switch = self.on_count[on_ids] % self.cfg.switches
        else:
            switch = torch.randint(self.cfg.switches, (len(on_ids),), device=self.device)
        switch = torch.where(start_on[turn_on], start[turn_on], switch)
        self.on_count[on_ids] += 1
        self.trick_command[on_ids, switch] = 1.0
        lo = torch.where(first, self.cfg.first_off_range[0],
                         torch.where(turn_on, self.cfg.on_time_range[0], self.cfg.off_time_range[0]))
        hi = torch.where(first, self.cfg.first_off_range[1],
                         torch.where(turn_on, self.cfg.on_time_range[1], self.cfg.off_time_range[1]))
        self.time_left[env_ids] = lo + (hi - lo) * torch.rand(len(env_ids), device=self.device)
        self.phase[env_ids] = 0.0
        self.on_time[env_ids] = 0.0
        self.rotation[env_ids] = 0.0
        self.anchor_yaw[on_ids] = self.robot.data.heading_w.torch[on_ids]
        self.anchor_xy[on_ids] = self.robot.data.root_pos_w.torch[on_ids, :2]
        self.command_counter[env_ids] += 1

    def _resample_command(self, env_ids):
        pass

    def _update_command(self):
        dt = self._env.step_dt
        self.rotation -= self.flipping * self.robot.data.root_ang_vel_b.torch[:, 1] * dt
        self.on_time += self.on * dt
        if not self.cfg.clock_hz:
            return
        on = self.trick_command[:, 0]
        self.phase += 2.0 * math.pi * self.cfg.clock_hz * self._env.step_dt * on
        self.trick_command[:, 1] = on * torch.sin(self.phase)
        self.trick_command[:, 2] = on * torch.cos(self.phase)

    def _update_metrics(self):
        pass


@configclass
class TrickCommandCfg(CommandTermCfg):
    class_type: type = TrickCommand
    asset_name: str = MISSING
    switches: int = 1
    # Each on segment takes the next switch in turn instead of a random one (play, for videos).
    cycle_switches: bool = False
    clock_hz: float = 0.0
    flip_s: float = 0.0
    start_on_prob: float = 0.0
    resampling_time_range: tuple[float, float] = (math.inf, math.inf)
    first_off_range: tuple[float, float] = (0.5, 2.0)
    on_time_range: tuple[float, float] = (4.0, 8.0)
    off_time_range: tuple[float, float] = (1.5, 3.0)
