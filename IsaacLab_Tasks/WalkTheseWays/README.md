# WalkTheseWays

Walk These Ways (Margolis & Agrawal, 2022) for the Go2 on Isaac Lab, trained with rsl_rl.
Cut down from `IsaacLab_Tasks/Simple`, keeping only what the one task needs.

Task: `Go2-WTW`

- Commands: velocity (x, y, yaw) plus body height, step frequency, gait (trot / pace / bound /
  pronk), swing height, body pitch and stance width, with a gait clock. A zero velocity command
  means stand.
- Rewards: WTW's set and weights, combined as positive terms x exp(penalties / 0.02), clipped at 0.
- Per-gait (vx, yaw) command curriculum, and a 30-step observation history on the actor.
- The Deploy arm's robustness: sensor noise and bias, actuator delay, PD gain, CoM and joint
  randomization, foot torsion, lying and dropped starts, standby at zero command.

Settings are environment variables prefixed `WTW_` and `PAPER_DEPLOY_` (see
`robots/go2/velocity_env_cfg.py`).

Train from the repo root with `python launcher.py` -> train -> WalkTheseWays, or directly:

    PYTHONPATH=source/walk_these_ways python scripts/rsl_rl/train.py --task=Go2-WTW --headless

`Controller/policy_runner.py` recognises these runs from their `params/env.yaml` (layout `wtw`)
and builds the stacked observation for MuJoCo and the robot. Pick the gait with
`QUADRUPED_GAIT=trot|pace|bound|pronk` or `QUADRUPED_GAIT_CMD="height,freq,phase,offset,bound,swing,pitch,width"`.
