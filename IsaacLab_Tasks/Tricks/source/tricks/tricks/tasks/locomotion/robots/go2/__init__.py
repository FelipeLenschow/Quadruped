import gymnasium as gym

gym.register(
    id="Go2-Pee",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tricks_env_cfg:PeeEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tricks_env_cfg:PeePlayEnvCfg",
        "rsl_rl_cfg_entry_point": "tricks.tasks.locomotion.agents.rsl_rl_ppo_cfg:PeePPORunnerCfg",
    },
)

gym.register(
    id="Go2-Sniff",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tricks_env_cfg:SniffEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tricks_env_cfg:SniffPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "tricks.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Go2-Wiggle",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tricks_env_cfg:WiggleEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tricks_env_cfg:WigglePlayEnvCfg",
        "rsl_rl_cfg_entry_point": "tricks.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
