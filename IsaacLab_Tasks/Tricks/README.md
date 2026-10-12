# Tricks

Go2 tricks for `skill_server`, one policy each, on Isaac Lab with rsl_rl. Cut from `IsaacLab_Tasks/Recovery`.

Tasks: `Go2-Pee`, `Go2-Sniff`, `Go2-Wiggle`

- **Switches.** The actor input is unitree's 45-wide layout with the trick's switches in the velocity command's
  place: `[on, 0, 0]`, or for Pee `[left, right, 0]`. One on: do that trick. All off: get back to the standing
  pose and stand still. Each episode (20 s) starts off for 0.5-2 s, then a random switch goes on for 4-8 s and
  everything is off for 1.5-3 s, in turn, so the policy learns to start the trick from standing and get back.
- **Deploy.** In `policies.json`, `"commands": "trick"`. The pipeline feeds what keeps arriving on
  `/pipeline/trick` (Float32MultiArray), which `skill_server` sends for the skill's duration. If it stops
  (skill_server gone), the switches go off within `cmd_vel.timeout` and the robot stands.
- **Iterations:** 600 (Pee 1000). The first runs went to 3000 and had nothing left to learn after ~600.
- **Starts:** standing in the nominal pose, joints off it by up to 0.15 rad, drifting at up to 0.3 m/s and
  0.3 rad/s: how the walker hands over when it stops.
- **Shared rewards:** while off, the standing pose (two widths: with only the narrow one, Pee and Wiggle never
  came back to it), base height 0.32 m, level base. Base still (Pee also while on). Feet planted: feet off the ground and feet sliding cost. Slow and gentle: Recovery's speed
  caps (joints 4 rad/s, base 1 rad/s and 0.3 m/s), impacts, hits, torque, action rate.
- **Inside the robot's checks**, so a trick keeps all protections: tilt past 22 deg (the check trips at 30)
  and joints in the outer 17% of their hardware range (config.yaml `joint_limits`; the check uses 15%) cost.
  Body or head on the ground, or tilt past 57 deg, ends the episode.
- **Sim-to-real:** Recovery's (sensor bias, actuator delay, PD gains, CoM, joint plant, calf knee
  reduction, foot torsion), pushes of ±0.3 m/s every 4-8 s. No belly pads.

## Go2-Pee

Two switches: left lifts the rear-left foot, right the rear-right. That foot at least 12 cm off the ground
(`TRICKS_PEE_FOOT_HEIGHT`), the hip turned out 0.55 rad (`TRICKS_PEE_HIP`; the check allows 0.73), the calf
folded to -2.3 rad (`TRICKS_PEE_CALF`: lie_flat's -2.6 is past the check's -2.39), the foot level with its hip
front to back so it goes out to the side, not back (the first runs kicked it back). The front foot on that side
steps 8 cm ahead of its hip (`TRICKS_PEE_FRONT_CLEAR`), out of the stream's way; the two legs on the other side
stay near the standing pose. A reward for taking weight off the foot pays before it can lift: the Go2's
centre of mass sits on the line between the other two diagonal feet, and without it the first runs never
shifted their weight and never lifted the foot.

## Go2-Sniff

Nose down 0.3 rad (`TRICKS_SNIFF_PITCH`), base at 0.26 m (`TRICKS_SNIFF_HEIGHT`), no roll, feet planted, and
the nose bobbing: the pitch follows 0.3 + 0.06 sin φ (`TRICKS_SNIFF_AMPLITUDE`) with φ a 1 Hz clock
(`TRICKS_SNIFF_HZ`). The command is `[on, sin φ, cos φ]`; φ starts at 0 when the trick goes on. On the robot the
pipeline makes the same clock, reading `clock_hz` from the run's `params/env.yaml`, so `skill_server` still
sends `[1, 0, 0]`. That folds the front calves to about -2.15 rad, inside the check's -2.39.

The first two runs had no clock (the swing reward below, on pitch) and gamed it: one vibrated at ~8 Hz, ±0.01
rad; the other held the nose still at one end of the band and shook.

## Go2-Wiggle

The back end swings side to side: base yaw ±0.08 rad (`TRICKS_WIGGLE_AMPLITUDE`, ~3 cm at the rear hips) at up to
0.55 rad/s (`TRICKS_WIGGLE_RATE`), ~0.6 s a swing, from where it pointed when the
trick went on, about a point 0.2 m ahead of the base centre (`TRICKS_WIGGLE_PIVOT`, between the front hips)
that stays put, feet planted.

Wiggle swings without a clock: turning at up to the rate pays, reaching 0.9 of the amplitude pays,
going past it costs, so the best is to swing between the ends at the rate (period ~4 × amplitude / rate).

## Training

Settings are environment variables prefixed `TRICKS_` (see `robots/go2/tricks_env_cfg.py`).
Train from the repo root with `python launcher.py` -> train -> Tricks, or directly:

    PYTHONPATH=source/tricks python scripts/rsl_rl/train.py --task=Go2-Pee --headless

Video: `Record Video? y` in the launcher starts `scripts/rsl_rl/video_watcher.py`, which plays every 500th
checkpoint with 16 robots, all 2 s off and 5 s on in turn, and saves `<run>/videos/progress/model_<iter>.mp4`.
By hand:

    python scripts/rsl_rl/video_watcher.py --run-dir logs/rsl_rl/go2_pee/<run> --once --every 1000

Play with `TRICKS_PLAY_OFF` / `TRICKS_PLAY_ON` seconds in turn (default 2 and 5); each on goes to the next
switch (Pee: left, then right). `--video --video_eye x,y,z` records one robot from that offset:

    PYTHONPATH=source/tricks python scripts/rsl_rl/play.py --task=Go2-Pee

Then add it to `src/quadruped_bringup/config/policies.json`:

    "Pee": {"path": "IsaacLab_Tasks/Tricks/logs/rsl_rl/go2_pee/<run>/model_<n>.pt", "commands": "trick"}

and call it with `ros2 service call /skill quadruped_interfaces/srv/Skill "{name: pee_right, duration: 4.0}"`
after `walk`.
