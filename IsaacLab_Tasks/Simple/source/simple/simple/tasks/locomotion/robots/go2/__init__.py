import gymnasium as gym

gym.register(
    id="Unitree-Go2-Velocity-Sigma-Vel-Foot-Rough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSigmaVelFootRoughEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSigmaVelFootRoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "simple.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Velocity-Sigma-Vel-Foot-Rough-Deploy",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSigmaVelFootRoughDeployEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotSigmaVelFootRoughDeployPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "simple.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Velocity-Clock",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotClockEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotClockPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "simple.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
