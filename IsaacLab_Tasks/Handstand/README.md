# Handstand

Go2 handstand and hindstand for `skill_server`, one policy, on Isaac Lab with rsl_rl. Cut from
`IsaacLab_Tasks/Tricks`, with the handstand and hindstand starts of Recovery's `Go2-LieDown` (Lie2).

Task: `Go2-Handstand`

- **Switches.** The actor input is unitree's 45-wide layout with two switches in the velocity command's place:
  `[1, 0, 0]` handstand (up on the front legs, nose down), `[0, 1, 0]` hindstand (up on the rear legs, nose up).
  Both off: get back down to the standing pose on four feet and stand still. Each episode (20 s) starts off for
  0.5-2 s, then a random switch goes on for 5-8 s and everything is off for 3-5 s, in turn.
- **Starts.** 70% standing as in Tricks. 30% (`HANDSTAND_ON_LEGS_FRACTION`) already up as in Lie2, at rest:
  nose down on straight front legs or tail down on straight rear legs, 1.0-1.5 rad from level. Half of those
  start with that switch on (hold it), half off (get down from it).
- **Up.** Pitch toward the support pair 1.35 rad (`HANDSTAND_HAND_PITCH`, `HANDSTAND_HIND_PITCH`), as a share
  of it (pays from the first degree) and close to it. Weight off the other pair (pays before those feet can
  leave the ground), centre of mass over the middle of the support feet, still once within 0.2 rad of the
  target, no lean to the side. The lifted feet do not count as feet off the ground.
- **Shared with Tricks:** while off, the standing pose, base height 0.32 m, level base; feet sliding, impacts,
  hits, torque, action rate. Speed caps are wider: joints 6 rad/s, base 2 rad/s and 0.5 m/s
  (`HANDSTAND_JOINT_VEL_CAP`, `_BASE_ANG_VEL_CAP`, `_BASE_LIN_VEL_CAP`). Pushes ±0.2 m/s every 4-8 s.
- **Ends:** body, head, a thigh or a calf on the ground (the first run knelt on the rear knees for the
  hindstand), leaning sideways past 1.0 rad (`HANDSTAND_SIDE_LIMIT`), or tilted past
  1.9 rad from level (`HANDSTAND_FALL_TILT`).
- **Iterations:** 3000.

## Deploy

Up on two legs is far past the robot's tilt and pitch checks (30 deg). The tilt penalty only counts while
both switches are off; the joint range penalty (outer 17%) still counts, to keep it inside the ROM check. In `src/quadruped_bringup/config/policies.json`:

    "Handstand": {"path": "IsaacLab_Tasks/Handstand/logs/rsl_rl/go2_handstand/<run>/model_<n>.pt",
                  "commands": "trick", "protections": ["heartbeat", "rom"]}

Then the app shows two buttons, Handstand and Hindstand (hold to do it, let go to come down), and
`skill_server` has the skills `handstand` and `hindstand` (`--handstand_time`, `--hindstand_time`, 5 s):

    ros2 service call /skill quadruped_interfaces/srv/Skill "{name: handstand, duration: 5.0}"

## Training

Settings are environment variables prefixed `HANDSTAND_` (see `robots/go2/handstand_env_cfg.py`).
Train from the repo root with `python launcher.py` -> train -> Handstand, or directly:

    PYTHONPATH=source/handstand python scripts/rsl_rl/train.py --task=Go2-Handstand --headless

Play with `HANDSTAND_PLAY_OFF` / `HANDSTAND_PLAY_ON` seconds in turn (default 2 and 5); each on goes to the
next switch (handstand, then hindstand):

    PYTHONPATH=source/handstand python scripts/rsl_rl/play.py --task=Go2-Handstand

Video: `Record Video? y` in the launcher starts `scripts/rsl_rl/video_watcher.py`, which plays every 500th
checkpoint and saves `<run>/videos/progress/model_<iter>.mp4`. By hand:

    python scripts/rsl_rl/video_watcher.py --run-dir logs/rsl_rl/go2_handstand/<run> --once --every 1000
