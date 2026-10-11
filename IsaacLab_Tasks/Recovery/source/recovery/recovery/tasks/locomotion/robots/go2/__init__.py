import gymnasium as gym

gym.register(
    id="Go2-Recovery",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.recovery_env_cfg:RobotRecoveryEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.recovery_env_cfg:RobotRecoveryPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "recovery.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Go2-LieDown",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.recovery_env_cfg:RobotLieDownEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.recovery_env_cfg:RobotLieDownPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "recovery.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
