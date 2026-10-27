# Backflip

Go2 backflip for `skill_server`, one policy, on Isaac Lab with rsl_rl. Cut from `IsaacLab_Tasks/Handstand`.

Task: `Go2-Backflip`

- **Switch and clock.** The actor input is unitree's 45-wide layout with `[on, sin, cos]` in the velocity
  command's place: a 0.25 Hz clock (`BACKFLIP_CLOCK_HZ`) that starts at 0 when the switch goes on, as the
  deploy pipeline makes it (`clock_hz` in the run's `env.yaml`). Each episode (20 s) starts off for 0.5-2 s,
  then on for 2.5-3.5 s and off for 2-4 s, in turn. The on time must stay below one clock turn (4 s), or the
  clock comes round and the robot flips again.
- **Flip window.** The first 1.5 s after the switch goes on (`BACKFLIP_FLIP_S`). In it: gravity in the base
  frame close to a reference turn (level until 0.3 s, then a smooth 2π backward until 1.0 s,
  `BACKFLIP_TURN_START`, `_TURN_END`); backward pitch rate up to 15 rad/s until the turn adds up to 2π; base
  height up to 0.35 m above standing in the reference turn (`BACKFLIP_RISE`); no roll or yaw rate, no lean to
  the side. No tilt limit, no speed caps, no feet-off-the-ground penalty.
- **After the window, and while off:** stand, as in Handstand: standing pose, base height 0.32 m, level, still,
  feet down. On, past the window, with less than 90% of the turn done (`BACKFLIP_DONE_SHARE`): a penalty.
- **Always:** the joint range penalty (outer 17%), joint limits, impacts above 300 N, hits, torque, action rate.
- **Ends:** body, head, a thigh or a calf on the ground, or tilted past 1.0 rad outside the flip window
  (`BACKFLIP_TIPPED`). The fall costs 20 (`BACKFLIP_TERMINATED`), not 200: a failed flip ends on the head or
  the back, and a high cost stops the policy from trying.
- **Critic only:** the flip window, its time and the turn done.
- **Iterations:** 3000.

## Deploy

Do not use it on the real robot before it is good in Isaac Lab and in MuJoCo (`Tools/mujoco_trick_video.py`).
The flip is far past the tilt and pitch checks, and the landing loads the calves to their torque limit. In
`src/quadruped_bringup/config/policies.json`:

    "Backflip": {"path": "IsaacLab_Tasks/Backflip/logs/rsl_rl/go2_backflip/<run>/model_<n>.pt",
                 "commands": "trick", "protections": ["heartbeat"]}

Turn the switch on for 2-3 s, one flip each time. Hold it for less than 4 s.

## Training

Settings are environment variables prefixed `BACKFLIP_` (see `robots/go2/backflip_env_cfg.py`).
Train from the repo root with `python launcher.py` -> train -> Backflip, or directly:

    PYTHONPATH=source/backflip python scripts/rsl_rl/train.py --task=Go2-Backflip --headless

Play with `BACKFLIP_PLAY_OFF` / `BACKFLIP_PLAY_ON` seconds in turn (default 2 and 3):

    PYTHONPATH=source/backflip python scripts/rsl_rl/play.py --task=Go2-Backflip

Video: `Record Video? y` in the launcher starts `scripts/rsl_rl/video_watcher.py`, which plays every 500th
checkpoint and saves `<run>/videos/progress/model_<iter>.mp4`. By hand:

    python scripts/rsl_rl/video_watcher.py --run-dir logs/rsl_rl/go2_backflip/<run> --once --every 1000
