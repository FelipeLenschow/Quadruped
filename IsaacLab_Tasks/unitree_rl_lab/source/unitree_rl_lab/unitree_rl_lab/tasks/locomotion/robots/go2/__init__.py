import gymnasium as gym

# Baseline, plus the three paper arms. Each arm differs from the baseline by exactly one
# thing -- see the "Paper arms" block in velocity_env_cfg.py. Separate task ids rather than
# environment-variable switches, so a run's arm is recorded in its own command line and in
# the log directory name.
_ARMS = {
    "Unitree-Go2-Velocity": ("RobotEnvCfg", "RobotPlayEnvCfg"),
    # reward SHAPE: tolerance scales with the commanded speed
    "Unitree-Go2-Velocity-Sigma": ("RobotSigmaEnvCfg", "RobotSigmaPlayEnvCfg"),
    # command COVERAGE: an explicit quota of slow commands
    "Unitree-Go2-Velocity-Coverage": ("RobotCoverageEnvCfg", "RobotCoveragePlayEnvCfg"),
    # both, to check neither is redundant given the other
    "Unitree-Go2-Velocity-Both": ("RobotBothEnvCfg", "RobotBothPlayEnvCfg"),
    # base_lin_vel in the ACTOR (48-dim observation instead of 45) -- unitree's actor is
    # velocity-blind. Alone, to attribute the sim2sim gap:
    "Unitree-Go2-Velocity-Vel": ("RobotVelEnvCfg", "RobotVelPlayEnvCfg"),
    # and on top of both fixes -- the version intended for hardware:
    "Unitree-Go2-Velocity-Both-Vel": ("RobotBothVelEnvCfg", "RobotBothVelPlayEnvCfg"),
}

for _task_id, (_env_cfg, _play_cfg) in _ARMS.items():
    gym.register(
        id=_task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:{_env_cfg}",
            "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:{_play_cfg}",
            "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
        },
    )
