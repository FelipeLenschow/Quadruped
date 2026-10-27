import gymnasium as gym

gym.register(
    id="Go2-Backflip",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.backflip_env_cfg:BackflipEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.backflip_env_cfg:BackflipPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "backflip.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
