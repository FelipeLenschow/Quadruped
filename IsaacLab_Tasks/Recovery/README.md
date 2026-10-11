# Recovery

Go2 fall recovery on Isaac Lab, trained with rsl_rl. Cut down from `IsaacLab_Tasks/Simple`.
The policy only has to get back to its standing pose and stay there, slowly and gently;
how fast it gets up does not matter.

Tasks: `Go2-Recovery`, and `Go2-LieDown` (below)

- Starts (weights `RECOVERY_STAND/DROP/LYING/FLIPPED/TUMBLE`, 0.05/0.05/0.3/0.3/0.3 at full
  difficulty): standing; dropped from up to
  0.5 m with tilt; lying on its belly in four leg poses; on its side or back with random
  joints; thrown from 0.6-1 m in any orientation, random joints, spinning.
- Terrain: flat. (Reco1 and Reco2 trained on 0-4 cm random rough ground, where some feet got caught.)
- Disturbances: a shove every 3-6 s.
  Nothing ends an episode early, so every fall has to be got out of.
- Curriculum: each env has a level from 0 to 1. It goes up 0.05 when the robot is standing
  as the episode ends and down 0.1 when it is not, so it settles where about 2/3 of
  episodes end standing. Level 0 only stands, drops low and lies on its belly; going up
  adds side/back and thrown starts and makes drops and throws stronger. `Curriculum/level` in TensorBoard is the mean. `RECOVERY_CURRICULUM=0` trains
  at level 1 from the start; play holds `RECOVERY_PLAY_LEVEL` (default 1).
- Rewards: belly down, base up, standing pose, all feet down, standing still. Joint speed
  over 3 rad/s, base turning over 1.5 rad/s, base speed over 0.4 m/s, rising faster than
  0.25 m/s and contact forces over 200 N are penalized, plus the usual torque, power,
  action rate and joint limit terms. Hits on anything but the feet (body, head, hips, thighs,
  calves) are penalized by how much their contact force jumps in one step past 30 N
  (`RECOVERY_HIT_MARGIN`, `RECOVERY_W_HITS`), so resting contact is free and slamming is not;
  this is in both tasks. (The first run, Reco1, had 4 rad/s, 2 rad/s, 0.6 m/s
  and no rise cap, and got up from its back in about 1 s. Reco2, at 2 rad/s and 1 rad/s,
  gave up from a third of belly starts; gamma went to 0.995 and base height to weight 2.)
- Resuming (`--resume`) starts the curriculum at level 1 unless `RECOVERY_LEVEL_INIT` is set.
- Simple's Deploy robustness: sensor bias, actuator delay, PD gains, CoM, joint plant,
  calf knee reduction, foot torsion.
- The Go2's four rubber belly pads, which the stock collision model leaves out, are added as
  convex hulls cut from the body's visual mesh (48 vertices each, `assets/robots/go2_belly_pads.json`,
  rebuilt by `python assets/robots/go2_belly_pads.py <menagerie unitree_go2/assets>`), and the chin
  collider is moved up 1 cm to where the chin is (`RECOVERY_BELLY_PADS=0` turns both off; ~3% slower
  per iteration). Lying on them tips the body ~5 deg nose up with the chin ~1.6 cm off the ground;
  without them the chin is the lowest point and the robot lies on its face.
- Actor input is unitree's 45-wide layout with the command always zero, so
  `policy_runner.py` runs it in MuJoCo and on the robot as it does Simple's policies.

Settings are environment variables prefixed `RECOVERY_` (see
`robots/go2/recovery_env_cfg.py`).

Train from the repo root with `python launcher.py` -> train -> Recovery, or directly:

    PYTHONPATH=source/recovery python scripts/rsl_rl/train.py --task=Go2-Recovery --headless

Video: recording inside training makes it ~20x slower here, so `Record Video? y` in the
launcher starts `scripts/rsl_rl/video_watcher.py` instead. It plays every 500th checkpoint
(and the last one) with 16 robots in a grid, seen from a fixed camera, all going through the
five starts together, 4 s each, on dark grey ground, no pushes, and saves `<run>/videos/progress/model_<iter>.mp4`
(a few minutes of work per clip; `--robots` changes the count). By hand:

    python scripts/rsl_rl/video_watcher.py --log-root logs/rsl_rl/go2_recovery
    python scripts/rsl_rl/video_watcher.py --run-dir logs/rsl_rl/go2_recovery/<run> --once --every 1000

Watch one kind of start with `RECOVERY_PLAY_START=tumble` (or stand, drop, lying, flipped):

    RECOVERY_PLAY_START=flipped PYTHONPATH=source/recovery python scripts/rsl_rl/play.py --task=Go2-Recovery

## Go2-LieDown

A safety controller to switch to from whatever the robot is doing: get belly down into config.yaml's
`lie_flat` pose (hips 0.45 front, 0.65 rear, thighs 1.25, calves −2.6), slowly, and rest the weight on the
belly pads so the motors can then be switched off. Measured in sim: ~145 N of ~150 N on the pads, chin
clear, and with all gains zeroed it does not move. "Lying" requires at least 30 N on the belly; the
joint limit penalty is off once lying, since folded flat the front calves rest at −2.7. The switch to
zero gains after a few seconds is not part of the policy; it belongs to whatever takes control.

- Starts (`RECOVERY_LIE_STAND/MOVING/DROP/LYING/FLIPPED/TUMBLE/HANDSTAND/HINDSTAND/TIPPING`): standing,
  walking or turning at up to 1 m/s and 1 rad/s, dropped, lying in another pose, on its side or back,
  thrown, nose down on straight front legs or tail down on straight rear legs (60-100 deg), and
  standing but already tipping over at 1.5-4 rad/s. Episodes are 10 s.
- Pushes: ±0.6 m/s on the first frame, on top of the start (`RECOVERY_LIE_PUSH_START=0` turns it off),
  then every 3-6 s.
- Rewards: belly down, base low, close to the pose, then low motor torque and stillness once there
  (so the weight is on the belly when the motors are switched off). Same speed caps as Recovery,
  plus a cap on sinking (0.3 m/s), and belly and head contact force over 100 N is penalized
  (`RECOVERY_LIE_BELLY_FORCE`, `RECOVERY_LIE_W_BELLY`). From standing, reaching the pose instantly
  peaks at ~6100 N on the belly, easing in over 1 s ~500 N, over 3 s ~160 N. On top of the linear
  non-foot hit term, jumps over 300 N are charged squared (per 100 N, `RECOVERY_LIE_W_HARD_HIT`), so
  one slam costs far more than several touches; foot jumps over 800 N are penalized too. The joint
  speed cap is off while the body is toppling (turning faster than 1-2.5 rad/s), so it can swing its
  legs under itself.
- Any head (chin, lidar cage) contact costs 0.1 per N (`RECOVERY_LIE_W_HEAD`), and a jump in it past
  50 N costs ((jump − 50) / 50 N)² × 0.25 on top (`RECOVERY_LIE_W_FACE_HIT`): falling out of a
  handstand onto the face. Belly contact over 200 N counts as a belly hit.
- Curriculum, sim-to-real, observation layout and the video watcher are Recovery's; the curriculum's
  goal is lying in the pose. Logs go to `logs/rsl_rl/go2_liedown`.

    PYTHONPATH=source/recovery python scripts/rsl_rl/train.py --task=Go2-LieDown --headless

Per-start test (lying %, time, hardest non-foot and foot hits):

    PYTHONPATH=source/recovery python scripts/rsl_rl/eval_liedown.py --checkpoint logs/rsl_rl/go2_liedown/<run>/model_<n>.pt
