import gymnasium as gym

gym.register(
    id="Go2-WTW",
    entry_point="walk_these_ways.tasks.locomotion.wtw_env:WTWEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotWTWEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotWTWPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "walk_these_ways.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
