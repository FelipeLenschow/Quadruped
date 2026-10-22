import gymnasium as gym

gym.register(
    id="Go2-Handstand",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.handstand_env_cfg:HandstandEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.handstand_env_cfg:HandstandPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "handstand.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
