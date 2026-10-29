from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import CurriculumTermCfg


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)

    # Average every reset since the last check, not only envs resetting on an exact multiple of max_episode_length.
    state = getattr(env, "_lin_vel_level_state", None)
    if state is None:
        state = {"step": env.common_step_counter, "sum": 0.0, "n": 0}
        env._lin_vel_level_state = state
    state["sum"] += float(torch.sum(env.reward_manager._episode_sums[reward_term_name][env_ids]))
    state["n"] += len(env_ids)

    if env.common_step_counter - state["step"] >= env.max_episode_length and state["n"] > 0:
        reward = (state["sum"] / state["n"]) / env.max_episode_length_s
        state["step"], state["sum"], state["n"] = env.common_step_counter, 0.0, 0
        if reward > reward_term.weight * 0.7:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def _level_moves(env, env_ids, command_name, up, down, min_path, after_full_speed, leave_distance):
    term = env.command_manager.get_term(command_name)
    commanded = term.commanded_path[env_ids]
    ratio = term.tracked_path[env_ids] / commanded.clamp(min=1e-6)
    fell = env.termination_manager.terminated[env_ids]
    enough = commanded > min_path
    move_up = enough & (ratio > up) & ~fell
    if after_full_speed and term.cfg.ranges.lin_vel_x[1] < term.cfg.limit_ranges.lin_vel_x[1] - 1e-6:
        move_up[:] = False
    move_down = fell | (enough & (ratio < down))
    if leave_distance > 0.0:
        left = term.max_distance[env_ids] > leave_distance
        move_up &= left
        move_down |= (commanded > 2.0 * leave_distance) & ~left
    return move_up, move_down


def terrain_levels_tracking(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    up: float = 0.6,
    down: float = 0.35,
    min_path: float = 0.5,
    after_full_speed: bool = True,
    leave_distance: float = 0.0,
) -> torch.Tensor:
    """Terrain levels from the share of the commanded path the robot covered, for commands that change
    direction and pause mid-episode, where net distance from the origin says little. Up above `up`, down
    below `down` or on a fall; episodes commanded less than min_path stay put unless they fell. With
    after_full_speed nothing moves up until the speed curriculum has opened the full range, so harder
    terrain does not hold tracking under its bar.

    With leave_distance, going up also needs the base to have been that far from the tile centre, and an
    episode commanded twice that far that never got there goes down: otherwise a robot can track its
    commands on the flat middle of a stair tile and never take a step."""
    move_up, move_down = _level_moves(env, env_ids, command_name, up, down, min_path, after_full_speed, leave_distance)
    terrain = env.scene.terrain
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


class terrain_stages(ManagerTermBase):
    """Terrain types opened in stages, with terrain_levels_tracking's levels inside each type.

    The terrain grid holds every type from the start; `stages` gives, per stage, the share of envs placed on
    each type (sub-terrain name -> share). An env moved to another type starts it at level 0, at its next
    reset. Going from stage i to i + 1 needs, for `hold` checks in a row (one per episode length of steps),
    the full speed range open, mean tracking over the resets above track_ratio x the weight of track_term,
    and the mean level of envs on `level_types` of stage i at least level_gates[i]. Moving between stages
    keeps as many envs as it can on their type.

    Once a stage has stairs (stair_types), envs on stairs get forward-only commands: all of them for
    forward_hold_steps, then a share falling linearly to forward_end_share over forward_ramp_steps.

    With stair_rows_per_count, stair levels come in blocks of that many rows, one block per entry of
    stair_counts (the step count): a fall or poor tracking at the first row of a block does not move the env
    down, so it never loses a step count. stair_leave_distance is leave_distance for stair tiles only; an env
    that fails it goes down also from the first row of a block. With stair_top_stay, a stair env going up from
    the top row goes to a random row of the last block, not of all rows, so the hardest stairs keep their robots.

    Logs the stage, the mean level, the forward-only share, the share of stair episodes that got the
    base report_distance from the tile centre (onto the steps), and the mean step count on stairs. Per stair
    direction (down: platform on top, up: the inverted type), as running means over about 200 episodes:
    <dir>_left (got report_distance out), <dir>_falls (ended by a termination other than the time limit),
    <dir>_stuck (commanded more than twice report_distance and never got out), and over all envs on that
    type: <dir>_steps (mean step count), <dir>_height_cm (mean step height, with stair_height_range) and
    <dir>_top (share in the last block of step counts)."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        p = cfg.params
        terrain = env.scene.terrain
        gen = terrain.cfg.terrain_generator
        self.names = list(gen.sub_terrains.keys())
        props = np.array([sub.proportion for sub in gen.sub_terrains.values()], dtype=float)
        props /= props.sum()
        col_kind = [int(np.min(np.where(i / gen.num_cols + 0.001 < np.cumsum(props))[0])) for i in range(gen.num_cols)]
        self.columns = [torch.tensor([c for c, k in enumerate(col_kind) if k == t], device=env.device) for t in range(len(self.names))]
        self.mixes = []
        for mix in p["stages"]:
            unknown = set(mix) - set(self.names)
            if unknown:
                raise ValueError(f"terrain_stages: no sub-terrain named {sorted(unknown)}; have {self.names}")
            share = torch.tensor([float(mix.get(n, 0.0)) for n in self.names], device=env.device)
            for t, n in enumerate(self.names):
                if share[t] > 0 and len(self.columns[t]) == 0:
                    raise ValueError(f"terrain_stages: '{n}' has no column; raise its proportion")
            self.mixes.append(share / share.sum())
        self.level_types = [self.names.index(n) for n in p.get("level_types", ()) if n in self.names]
        self.stair_types = torch.tensor([self.names.index(n) for n in p.get("stair_types", ()) if n in self.names], device=env.device, dtype=torch.long)
        self.stage = 0
        self.stairs_since = None
        self.kind = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.target = self._retarget(self.mixes[0], fresh=True)
        self._move(torch.arange(env.num_envs, device=env.device), keep_level=True)
        env._terrain_kind = self.kind
        env._terrain_kind_names = self.names
        self._check = {"step": env.common_step_counter, "sum": 0.0, "n": 0, "passed": 0}
        self._stair_means = {}

    def _running(self, key: str, values: torch.Tensor, window: float = 200.0):
        n = values.numel()
        if n == 0:
            return
        mean = float(values.float().mean())
        old = self._stair_means.get(key)
        self._stair_means[key] = mean if old is None else old + (mean - old) * (1.0 - (1.0 - 1.0 / window) ** n)

    def _stair_stats(self, env, ids, term, report_distance, rows_per_count, counts, height_range):
        terrain = env.scene.terrain
        kind = self.kind[ids]
        died = env.termination_manager.terminated[ids]
        for t in self.stair_types.tolist():
            tag = "up" if self.names[t].endswith("_inv") else "down"
            sel = ids[kind == t]
            if sel.numel():
                left = term.max_distance[sel] > report_distance
                self._running(f"{tag}_left", left)
                self._running(f"{tag}_falls", died[kind == t])
                self._running(f"{tag}_stuck", (term.commanded_path[sel] > 2.0 * report_distance) & ~left)
            on = self.kind == t
            if rows_per_count > 0 and len(counts) and bool(on.any()):
                levels = terrain.terrain_levels[on]
                block = (levels // rows_per_count).clamp(max=len(counts) - 1)
                self._stair_means[f"{tag}_steps"] = float(torch.tensor(counts, device=env.device, dtype=torch.float)[block].mean())
                self._stair_means[f"{tag}_top"] = float((block == len(counts) - 1).float().mean())
                if height_range is not None:
                    row = (levels % rows_per_count).float()
                    lo, hi = height_range
                    self._stair_means[f"{tag}_height_cm"] = 100.0 * float((lo + (hi - lo) * (row + 0.5) / rows_per_count).mean())

    def _retarget(self, share: torch.Tensor, fresh: bool = False) -> torch.Tensor:
        n = self._env.num_envs
        if fresh:
            return torch.multinomial(share, n, replacement=True)
        old = torch.bincount(self.target, minlength=len(self.names)).float() / n
        keep_p = torch.where(old > 0, (share / old.clamp(min=1e-9)).clamp(max=1.0), torch.zeros_like(old))
        keep = torch.rand(n, device=share.device) < keep_p[self.target]
        target = self.target.clone()
        deficit = (share - old * keep_p).clamp(min=0.0)
        moved = ~keep
        if bool(moved.any()) and float(deficit.sum()) > 0:
            target[moved] = torch.multinomial(deficit / deficit.sum(), int(moved.sum()), replacement=True)
        return target

    def _move(self, ids: torch.Tensor, keep_level: bool = False):
        """Put envs ids on their target type, on a random column of it, at level 0 unless keep_level."""
        terrain = self._env.scene.terrain
        for t in range(len(self.names)):
            sel = ids[self.target[ids] == t]
            if sel.numel() == 0:
                continue
            cols = self.columns[t]
            terrain.terrain_types[sel] = cols[torch.randint(len(cols), (sel.numel(),), device=sel.device)]
        if not keep_level:
            terrain.terrain_levels[ids] = 0
        self.kind[ids] = self.target[ids]
        terrain.env_origins[ids] = terrain.terrain_origins[terrain.terrain_levels[ids], terrain.terrain_types[ids]]

    def _gate(self, env, env_ids, track_term, track_ratio, hold, level_gates, command_name):
        state = self._check
        state["sum"] += float(torch.sum(env.reward_manager._episode_sums[track_term][env_ids]))
        state["n"] += len(env_ids)
        if env.common_step_counter - state["step"] < env.max_episode_length or state["n"] == 0:
            return False
        tracking = (state["sum"] / state["n"]) / env.max_episode_length_s
        state["step"], state["sum"], state["n"] = env.common_step_counter, 0.0, 0
        weight = env.reward_manager.get_term_cfg(track_term).weight
        cmd = env.command_manager.get_term(command_name).cfg
        full = cmd.ranges.lin_vel_x[1] >= cmd.limit_ranges.lin_vel_x[1] - 1e-6
        level_ok = True
        on_levels = torch.isin(self.kind, torch.tensor(self.level_types, device=env.device, dtype=torch.long))
        if self.stage < len(level_gates) and bool(on_levels.any()):
            level_ok = float(env.scene.terrain.terrain_levels[on_levels].float().mean()) >= level_gates[self.stage]
        state["passed"] = state["passed"] + 1 if (full and tracking > track_ratio * weight and level_ok) else 0
        return state["passed"] >= hold

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        stages: list,
        level_types: tuple = (),
        level_gates: tuple = (),
        stair_types: tuple = (),
        stair_rows_per_count: int = 0,
        stair_counts: tuple = (),
        stair_height_range: tuple | None = None,
        stair_top_stay: bool = False,
        track_term: str = "track_lin_vel_xy",
        track_ratio: float = 0.75,
        hold: int = 3,
        forward_hold_steps: int = 0,
        forward_ramp_steps: int = 0,
        forward_end_share: float = 0.0,
        report_distance: float = 2.0,
        command_name: str = "base_velocity",
        up: float = 0.6,
        down: float = 0.35,
        min_path: float = 0.5,
        after_full_speed: bool = True,
        leave_distance: float = 0.0,
        stair_leave_distance: float = 0.0,
    ) -> dict:
        ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).reshape(-1)
        term = env.command_manager.get_term(command_name)
        terrain = env.scene.terrain

        on_stairs = torch.isin(self.kind[ids], self.stair_types)
        self._stair_stats(env, ids, term, report_distance, stair_rows_per_count, stair_counts, stair_height_range)
        stairs_left = float((term.max_distance[ids][on_stairs] > report_distance).float().mean()) if bool(on_stairs.any()) else 0.0

        move_up, move_down = _level_moves(env, ids, command_name, up, down, min_path, after_full_speed, leave_distance)
        if stair_rows_per_count > 0:
            move_down &= ~(on_stairs & (terrain.terrain_levels[ids] % stair_rows_per_count == 0))
        if stair_leave_distance > 0.0:
            left = term.max_distance[ids] > stair_leave_distance
            stayed = on_stairs & (term.commanded_path[ids] > 2.0 * stair_leave_distance) & ~left
            move_up &= ~(on_stairs & ~left)
            move_down |= stayed
            move_up &= ~move_down
        moving = self.target[ids] != self.kind[ids]
        top = terrain.terrain_origins.shape[0] - 1
        past_top = on_stairs & move_up & ~moving & (terrain.terrain_levels[ids] == top)
        terrain.update_env_origins(ids, move_up & ~moving, move_down & ~moving)
        if stair_top_stay and stair_rows_per_count > 0 and bool(past_top.any()):
            sel = ids[past_top]
            terrain.terrain_levels[sel] = torch.randint(top + 1 - stair_rows_per_count, top + 1, (sel.numel(),), device=env.device)
            terrain.env_origins[sel] = terrain.terrain_origins[terrain.terrain_levels[sel], terrain.terrain_types[sel]]
        if bool(moving.any()):
            self._move(ids[moving])

        if self.stage < len(self.mixes) - 1 and self._gate(env, ids, track_term, track_ratio, hold, level_gates, command_name):
            self.stage += 1
            self._check["passed"] = 0
            self.target = self._retarget(self.mixes[self.stage])
            print(f"[terrain_stages] stage {self.stage}: " + ", ".join(
                f"{n} {float(s):.2f}" for n, s in zip(self.names, self.mixes[self.stage]) if s > 0))
        if self.stairs_since is None and len(self.stair_types) and bool(self.mixes[self.stage][self.stair_types].sum() > 0):
            self.stairs_since = env.common_step_counter

        share = 0.0
        if self.stairs_since is not None:
            t = env.common_step_counter - self.stairs_since - forward_hold_steps
            ramp = 1.0 if t <= 0 else max(0.0, 1.0 - t / max(forward_ramp_steps, 1))
            share = forward_end_share + (1.0 - forward_end_share) * ramp
        on_stairs = torch.isin(self.kind[ids], self.stair_types)
        term.forward_only[ids] = on_stairs & (torch.rand(len(ids), device=env.device) < share)

        stats = {"stair_steps": 0.0}
        all_stairs = torch.isin(self.kind, self.stair_types)
        if stair_rows_per_count > 0 and len(stair_counts) and bool(all_stairs.any()):
            counts = torch.tensor(stair_counts, device=env.device, dtype=torch.float)
            block = (terrain.terrain_levels[all_stairs] // stair_rows_per_count).clamp(max=len(stair_counts) - 1)
            stats["stair_steps"] = float(counts[block].mean())
        return {
            **stats,
            **self._stair_means,
            "stage": float(self.stage),
            "level": float(terrain.terrain_levels.float().mean()),
            "forward_only_share": share,
            "stairs_left": stairs_left,
        }
