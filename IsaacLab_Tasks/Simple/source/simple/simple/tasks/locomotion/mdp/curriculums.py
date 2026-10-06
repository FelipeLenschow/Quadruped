from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)

    state = getattr(env, "_ang_vel_level_state", None)
    if state is None:
        state = {"step": env.common_step_counter, "sum": 0.0, "n": 0}
        env._ang_vel_level_state = state
    state["sum"] += float(torch.sum(env.reward_manager._episode_sums[reward_term_name][env_ids]))
    state["n"] += len(env_ids)

    if env.common_step_counter - state["step"] >= env.max_episode_length and state["n"] > 0:
        reward = (state["sum"] / state["n"]) / env.max_episode_length_s
        state["step"], state["sum"], state["n"] = env.common_step_counter, 0.0, 0
        if reward > reward_term.weight * 0.7:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def lin_vel_x_max_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    step: float = 0.5,
    threshold: float = 0.7,
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    """Raise only the top of the forward command range by `step`, up to its limit, each time the mean
    tracking reward over an episode's worth of resets clears threshold x weight."""
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit = command_term.cfg.limit_ranges.lin_vel_x[1]
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)

    state = getattr(env, "_lin_vel_x_max_state", None)
    if state is None:
        state = {"step": env.common_step_counter, "sum": 0.0, "n": 0}
        env._lin_vel_x_max_state = state
    state["sum"] += float(torch.sum(env.reward_manager._episode_sums[reward_term_name][env_ids]))
    state["n"] += len(env_ids)

    if env.common_step_counter - state["step"] >= env.max_episode_length and state["n"] > 0:
        reward = (state["sum"] / state["n"]) / env.max_episode_length_s
        state["step"], state["sum"], state["n"] = env.common_step_counter, 0.0, 0
        if reward > reward_term.weight * threshold:
            ranges.lin_vel_x = (ranges.lin_vel_x[0], min(ranges.lin_vel_x[1] + step, limit))

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def terrain_levels_tracking(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    up: float = 0.6,
    down: float = 0.35,
    min_path: float = 0.5,
    after_full_speed: bool = True,
) -> torch.Tensor:
    """Terrain levels from the share of the commanded path the robot covered, for commands that change
    direction and pause mid-episode, where net distance from the origin says little. Up above `up`, down
    below `down` or on a fall; episodes commanded less than min_path stay put unless they fell. With
    after_full_speed nothing moves up until the speed curriculum has opened the full range, so harder
    terrain does not hold tracking under its bar."""
    term = env.command_manager.get_term(command_name)
    commanded = term.commanded_path[env_ids]
    ratio = term.tracked_path[env_ids] / commanded.clamp(min=1e-6)
    fell = env.termination_manager.terminated[env_ids]
    enough = commanded > min_path
    move_up = enough & (ratio > up) & ~fell
    if after_full_speed and term.cfg.ranges.lin_vel_x[1] < term.cfg.limit_ranges.lin_vel_x[1] - 1e-6:
        move_up[:] = False
    move_down = fell | (enough & (ratio < down))
    terrain = env.scene.terrain
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
