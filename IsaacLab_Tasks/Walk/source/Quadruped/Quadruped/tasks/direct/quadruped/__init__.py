# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import gymnasium as gym

from . import agents


def _agent_cfg_for_phase() -> str:
    """Pick the skrl config file for the phase this process was launched with.

    A phase whose optimiser settings differ from my own -- `unitree`, which reproduces rsl_rl's
    PPO so that arm U1 is comparable to U0 -- ships its own `skrl_ppo_<phase>_cfg.yaml` next to
    the default one. Selecting it here rather than through a separate gym id keeps launcher.py
    unchanged: it already passes QUADRUPED_TRAINING_PHASE through to the training subprocess.

    Falls back to skrl_ppo_cfg.yaml whenever no phase-specific file exists, which is every phase
    but that one.
    """
    raw = os.environ.get("QUADRUPED_TRAINING_PHASE", "phase1")
    # "phase1_to_phase2" / "phase1_onward" name a chain; the segment actually being run is the
    # first one, and launcher.py overrides this variable per segment anyway.
    phase = raw.split("_to_")[0].replace("_onward", "")
    candidate = f"skrl_ppo_{phase}_cfg.yaml"
    if os.path.isfile(os.path.join(os.path.dirname(agents.__file__), candidate)):
        print(f"[Quadruped] phase '{phase}' -> agent config {candidate}")
        return candidate
    return "skrl_ppo_cfg.yaml"


_SKRL_CFG = _agent_cfg_for_phase()

##
# Register Gym environments.
##


gym.register(
    id="Template-Quadruped-Direct-v0",
    entry_point=f"{__name__}.quadruped_env:QuadrupedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadruped_env_cfg:QuadrupedEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:{_SKRL_CFG}",
    },
)

gym.register(
    id="Template-Quadruped-Sim2Sim-v0",
    entry_point=f"{__name__}.quadruped_sim2sim_env:QuadrupedSim2SimEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadruped_sim2sim_env_cfg:QuadrupedSim2SimEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:{_SKRL_CFG}",
    },
)