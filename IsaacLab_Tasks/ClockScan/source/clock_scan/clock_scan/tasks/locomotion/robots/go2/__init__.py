import gymnasium as gym

gym.register(
    id="Go2-Clock-Scan",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.clock_scan_env_cfg:ClockScanEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.clock_scan_env_cfg:ClockScanPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "clock_scan.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
