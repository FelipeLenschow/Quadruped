from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import wrap_to_pi

from ..terrain_info import tile_heading


def foot_sweep_speed(velocity: torch.Tensor, foot_x: float, foot_y: float) -> torch.Tensor:
    """Speed of the fastest stance foot under a body twist (vx, vy, wz): max over feet of |v + w x r|,
    feet at (+-foot_x, +-foot_y) from the base."""
    vx, vy, wz = velocity[:, 0:1], velocity[:, 1:2], velocity[:, 2:3]
    rx = torch.tensor([foot_x, foot_x, -foot_x, -foot_x], device=velocity.device)
    ry = torch.tensor([foot_y, -foot_y, foot_y, -foot_y], device=velocity.device)
    return torch.hypot(vx - wz * ry, vy + wz * rx).amax(dim=1)


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
        self._stop = torch.full((self.num_envs,), float("inf"), device=self.device)
        self._moving = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._switch_time = torch.full((self.num_envs,), -1e9, device=self.device)
        self.commanded_path = torch.zeros(self.num_envs, device=self.device)
        self.tracked_path = torch.zeros(self.num_envs, device=self.device)
        self.metrics["max_distance"] = torch.zeros(self.num_envs, device=self.device)
        # Set per env by mdp.terrain_stages at reset.
        self.forward_only = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Heading an operator holds on tiles that have one (mdp.tile_heading), NaN elsewhere.
        self.steer_heading = torch.full((self.num_envs,), float("nan"), device=self.device)
        # Leash: a reference pose moved by the command, from the robot's pose one step after reset.
        self.ref_pos_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.ref_yaw = torch.zeros(self.num_envs, device=self.device)
        self.leash_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.leash_pos_error_w = torch.zeros(self.num_envs, 2, device=self.device)
        self.leash_yaw_error = torch.zeros(self.num_envs, device=self.device)
        self.metrics["leash_distance"] = torch.zeros(self.num_envs, device=self.device)
        self._leash_steps = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """The sampled command, held at zero until this episode's standby has elapsed.

        The sampled command itself is left alone in vel_command_b, so resampling, the slow and
        axis-only shares and the standing envs all behave exactly as before; only what the
        observation, the rewards and the metrics read is masked.
        """
        if self.cfg.standby_duration_range[1] <= 0.0 and self.cfg.stop_time_range[1] <= 0.0:
            return self.vel_command_b
        t = self._env.episode_length_buf * self._env.step_dt
        active = (t >= self._standby) & (t < self._stop)
        return self.vel_command_b * active.unsqueeze(1).float()

    @property
    def max_distance(self) -> torch.Tensor:
        """Furthest the base has been from the env origin this episode, m."""
        return self.metrics["max_distance"]

    @property
    def transition_age(self) -> torch.Tensor:
        """Seconds since the command last switched between standing and moving (stand_threshold)."""
        return self._env.episode_length_buf * self._env.step_dt - self._switch_time

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        ids = slice(None) if env_ids is None else env_ids
        self._moving[ids] = False
        self._switch_time[ids] = -1e9
        self.commanded_path[ids] = 0.0
        self.tracked_path[ids] = 0.0
        self.leash_valid[ids] = False
        self.leash_pos_error_w[ids] = 0.0
        self.leash_yaw_error[ids] = 0.0
        self._leash_steps[ids] = 0.0
        extras = super().reset(env_ids)
        low, high = self.cfg.standby_duration_range
        if high > 0.0:
            ids = slice(None) if env_ids is None else env_ids
            self._standby[ids] = torch.empty_like(self._standby[ids]).uniform_(low, high)
        low, high = self.cfg.stop_time_range
        if high > 0.0:
            ids = slice(None) if env_ids is None else env_ids
            self._stop[ids] = torch.empty_like(self._stop[ids]).uniform_(low, high)
        return extras

    def _update_metrics(self):
        command = self.command
        self._error_xy_sum += torch.linalg.norm(
            command[:, :2] - self.robot.data.root_lin_vel_b.torch[:, :2], dim=-1
        )
        self._error_yaw_sum += torch.abs(command[:, 2] - self.robot.data.root_ang_vel_b.torch[:, 2])
        self._step_count += 1.0
        moving = torch.norm(command, dim=1) >= self.cfg.stand_threshold
        switched = moving != self._moving
        self._switch_time = torch.where(switched, self._env.episode_length_buf * self._env.step_dt, self._switch_time)
        self._moving = moving
        speed = torch.linalg.norm(command[:, :2], dim=-1)
        along = torch.sum(self.robot.data.root_lin_vel_b.torch[:, :2] * command[:, :2], dim=-1) / speed.clamp(min=1e-6)
        self.commanded_path += speed * self._env.step_dt
        self.tracked_path += torch.minimum(along.clamp(min=0.0), speed) * self._env.step_dt
        offset = self.robot.data.root_pos_w.torch[:, :2] - self._env.scene.env_origins[:, :2]
        self.metrics["max_distance"] = torch.maximum(self.metrics["max_distance"], torch.linalg.norm(offset, dim=1))
        self._update_leash(command)

    def _update_leash(self, command: torch.Tensor):
        """Walk's leash without its cap: the reference turns at the commanded yaw rate and moves at the commanded
        velocity in its own heading, and its error to the robot is never clipped."""
        from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

        dt = self._env.step_dt
        pos = self.robot.data.root_pos_w.torch[:, :2]
        yaw = euler_xyz_from_quat(self.robot.data.root_quat_w.torch)[2]
        start = ~self.leash_valid & (self._env.episode_length_buf >= 1)
        self.ref_pos_xy[start] = pos[start]
        self.ref_yaw[start] = yaw[start]
        self.leash_valid |= start
        live = self.leash_valid.float()
        self.ref_yaw += command[:, 2] * dt * live
        c, s = torch.cos(self.ref_yaw), torch.sin(self.ref_yaw)
        self.ref_pos_xy[:, 0] += (command[:, 0] * c - command[:, 1] * s) * dt * live
        self.ref_pos_xy[:, 1] += (command[:, 0] * s + command[:, 1] * c) * dt * live
        self.leash_pos_error_w = (self.ref_pos_xy - pos) * live.unsqueeze(1)
        self.leash_yaw_error = wrap_to_pi(self.ref_yaw - yaw) * live
        self._leash_steps += live
        distance = torch.linalg.norm(self.leash_pos_error_w, dim=1)
        self.metrics["leash_distance"] += (distance - self.metrics["leash_distance"]) * live / self._leash_steps.clamp(min=1.0)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)

        ids = torch.as_tensor(env_ids, device=self.device).reshape(-1)
        if ids.numel() == 0:
            return
        self._apply_forward_only(ids)
        self._apply_steer(ids)
        self._apply_axis_only(ids)
        self._apply_slow(ids)
        self._apply_foot_speed_limit(ids)
        self._apply_pause(ids)

    def _apply_forward_only(self, ids: torch.Tensor):
        """For the envs flagged forward_only: forward speed only, up to forward_only_x_max, sideways and
        turning scaled by forward_only_y_scale and forward_only_yaw_scale."""
        ids = ids[self.forward_only[ids]]
        if ids.numel() == 0:
            return
        self.vel_command_b[ids, 0] = self.vel_command_b[ids, 0].abs().clamp(max=self.cfg.forward_only_x_max)
        self.vel_command_b[ids, 1] *= self.cfg.forward_only_y_scale
        self.vel_command_b[ids, 2] *= self.cfg.forward_only_yaw_scale

    def _apply_steer(self, ids: torch.Tensor):
        """On tiles with a heading, sideways speed x steer_y_scale; the turning is set every step by _update_command."""
        self.steer_heading[ids] = tile_heading(self._env, ids) if self.cfg.steer_gain > 0.0 else float("nan")
        steered = ids[~torch.isnan(self.steer_heading[ids])]
        self.vel_command_b[steered, 1] *= self.cfg.steer_y_scale

    def _update_command(self):
        """Upstream's standing envs at zero, then on tiles with a heading, while the command moves the robot, the
        turning an operator would give to hold that heading: steer_gain x the heading error, within +-steer_max."""
        super()._update_command()
        if self.cfg.steer_gain <= 0.0:
            return
        moving = torch.linalg.norm(self.vel_command_b[:, :2], dim=1) > 0.0
        steered = (~torch.isnan(self.steer_heading) & moving).nonzero(as_tuple=False).flatten()
        if steered.numel() == 0:
            return
        error = wrap_to_pi(self.steer_heading[steered] - self.robot.data.heading_w.torch[steered])
        self.vel_command_b[steered, 2] = (self.cfg.steer_gain * error).clamp(-self.cfg.steer_max, self.cfg.steer_max)

    def _apply_pause(self, ids: torch.Tensor):
        """Turn a share of resamples into a short zero-command pause, so the next command is a start
        from standstill. Otherwise a policy starts about once per episode, after the standby."""
        if self.cfg.pause_fraction <= 0.0:
            return
        pause = ids[torch.rand(ids.numel(), device=self.device) < self.cfg.pause_fraction]
        if pause.numel() == 0:
            return
        self.is_standing_env[pause] = True
        self.time_left[pause] = torch.empty(pause.numel(), device=self.device).uniform_(*self.cfg.pause_duration_range)

    def _apply_foot_speed_limit(self, ids: torch.Tensor):
        """Scale a command down until its fastest stance foot moves at max_foot_speed."""
        limit = self.cfg.max_foot_speed
        if limit <= 0.0:
            return
        sweep = foot_sweep_speed(self.vel_command_b[ids], self.cfg.foot_x, self.cfg.foot_y)
        scale = (limit / sweep.clamp(min=1e-6)).clamp(max=1.0)
        self.vel_command_b[ids] *= scale.unsqueeze(1)

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
            selected = (draw >= lower) & (draw < upper) & ~self.is_standing_env[ids] & ~self.forward_only[ids]
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
    # Episode time, s, drawn per episode, after which the command is zero again. (0, 0) disables.
    stop_time_range: tuple[float, float] = (0.0, 0.0)
    # Hold the slow share off until the level curriculum has widened lin_vel_x to its limit.
    slow_after_full_range: bool = False

    # Command norm below which the robot counts as standing, for transition_age.
    stand_threshold: float = 0.02

    # Share of resamples that become a zero-command pause lasting pause_duration_range seconds. 0 disables.
    pause_fraction: float = 0.0
    pause_duration_range: tuple[float, float] = (0.5, 2.0)

    # Cap on the fastest stance foot's speed, turning included (see foot_sweep_speed). 0 disables.
    max_foot_speed: float = 0.0
    # Forward speed cap, and sideways speed and turning kept as a share, on forward_only commands.
    forward_only_x_max: float = float("inf")
    forward_only_y_scale: float = 0.0
    forward_only_yaw_scale: float = 0.3
    # On tiles with a heading (mdp.tile_heading): turning steer_gain x heading error within +-steer_max, as an
    # operator holding the line, and sideways speed x steer_y_scale. steer_gain 0 disables.
    steer_gain: float = 0.0
    steer_max: float = 0.5
    steer_y_scale: float = 1.0
    foot_x: float = 0.19
    foot_y: float = 0.14
