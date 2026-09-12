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
        if reward > reward_term.weight * 0.8:
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
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)
