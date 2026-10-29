# ClockScan

Go2 velocity tracking with a trot clock on rough ground, on Isaac Lab with rsl_rl. Cut from
`IsaacLab_Tasks/Simple`: only its `Clock-Rough` task, in one flat config, with a height scan in the actor.

Task: `Go2-Clock-Scan`

- **Same as Simple's Clock-Rough:** the trot clock whose rate follows the command, Walk These Ways' contact
  and swing-height terms, the Deploy sensor model, plant randomization and lying or dropped starts, and the
  tracking curriculum.
- **Terrain stages** (`mdp.terrain_stages`): the grid holds every type from the start; the stage sets the share of
  robots on each.
  1. Simple's ground: 30% flat, 70% smooth (noise to 2 cm).
  2. 15% flat, 15% smooth, 35% rough (to 4 cm), 35% boxes (2-10 cm).
  3. 50% stairs (`CLOCK_SCAN_STAIRS_SHARE`; half down from a platform on top, half up from a pit), the rest in
     stage 2's mix.

  A stage opens after 3 checks in a row (`CLOCK_SCAN_STAGE_HOLD`, one per episode length of steps) with the full
  speed range, mean tracking above 0.65 x its weight (`CLOCK_SCAN_STAGE_TRACK`), and for stage 3 a mean boxes level
  of at least 0.4 of the rows (`CLOCK_SCAN_STAGE_BOXES_DIFFICULTY`). A robot moved to another type starts it at level 0; as many robots
  as possible keep their type. Inside a type, levels go up after tracking more than 60% of the commanded path, down
  after a fall or under 35%.
- **Stair levels** (`mdp.CountedStairsTerrainCfg`): a 3 m platform, steps 0.3 m deep, flat ground beyond the last
  step. The 16 rows are 4 blocks of 4, one block per step count (1, 2, 3, 6; `CLOCK_SCAN_STEP_COUNTS`,
  `_STEP_HEIGHTS_PER_COUNT`); inside a block the step height rises from 5 to 22 cm (`CLOCK_SCAN_STEP_LO`, `_HI`),
  and starts at 5 cm again in the next block; the target stairs are 18 x 30 cm, in the last row (17.75-22 cm). A
  stair robot going up from the last row goes to a random row of the 6-step block, not of all rows. A fall lowers the height, never the step count. The first edge is
  1.5 m from the centre for every count. On stair tiles a level up also needs the base 2 m from the centre (past
  the first edge; `CLOCK_SCAN_STAIRS_LEAVE_DISTANCE`), and an episode commanded more than 4 m that never got there
  goes down a level, also from the first row of a step count. Without this, a robot that never climbs can track its
  commands on the 3 m platform and go up the levels.
- **Stairs:** robots on stairs get forward-only commands: 0 <= x <= 0.6 m/s, sideways kept, turning x0.3
  (`CLOCK_SCAN_FORWARD_X_MAX`, `_Y_SCALE`, `_YAW_SCALE`), always (`CLOCK_SCAN_FORWARD_END` 1.0; with less, the share falls
  to it over `_RAMP_ITERS` after `_HOLD_ITERS`). On stair tiles a fall costs 0.25 of a fall elsewhere (`CLOCK_SCAN_STAIRS_FALL_SCALE`); a foot hitting
  a vertical face and body contacts cost the same as elsewhere (`CLOCK_SCAN_STAIRS_STUMBLE_SCALE`,
  `_CONTACT_SCALE`, 1.0).
- **Stumble:** a foot hits a vertical face when its sideways contact force is more than 2x its vertical force
  (`CLOCK_SCAN_STUMBLE_RATIO`; the friction is at most 1.2, so a slipping foot stays under it), 5 per second
  (`CLOCK_SCAN_W_STUMBLE`). At 0.25x on stairs, 2 per second and a 4x ratio, about one foot swing in five hit a riser
  going up, the rear legs most.
- **Body tilt** (`mdp.ground_orientation_l2`, 2.5 per unit, `CLOCK_SCAN_W_ORIENT`): the tilt penalty is measured
  against a plane fitted to the critic's height grid (slope clipped at 35 degrees), not against gravity, so on stairs
  the body may pitch with the steps; on flat ground it is the same as `flat_orientation_l2`
  (`CLOCK_SCAN_GROUND_ORIENTATION=0` goes back to that). Against gravity, robots climbed with the body 7 degrees
  flatter than 15 cm stairs and the lower head 10-11 cm above the step under it.
- **Head clearance:** one ray down from the lower head (`head_ground`, where the lidar sits) gives the critic the
  head's height above the ground under it (critic only), and costs 20 per m below 0.15 m while moving
  (`CLOCK_SCAN_HEAD_CLEARANCE`, `_W_HEAD`).
- **Swing height:** the clock's swing profile (8 cm, `CLOCK_SCAN_SWING_HEIGHT`) is measured above the ground right
  under each foot, from one downward ray per foot (`foot_ground_FL` ... `_RR`), so it follows steps up and down.
  Measured from where the foot last stood, as Simple does, a step down cost as much as a missed lift.
  `CLOCK_SCAN_LEAVE_DISTANCE` (default 0, off) is the 2 m rule above on every tile type.
- **Leash** (Walk's, with no cap): a reference pose moved each step by the command (turning at the commanded yaw
  rate, moving at the commanded velocity in its own heading), set to the robot's pose one step after reset. The
  penalty is 2.0 per m from it and 0.5 per rad of heading to it (`CLOCK_SCAN_W_LEASH_POS`, `_W_LEASH_YAW`). On stair tiles
  the distance penalty can be scaled (`CLOCK_SCAN_STAIRS_LEASH_SCALE`, default 1.0): at 0.25 robots learned to
  stand at the first riser going up instead of climbing, as standing then cost less than climbing. The error
  is never clipped, so stopping where the command says go costs more every second. The critic gets the error (x, y in
  the base's yaw frame, heading); the actor does not. Logged as `Metrics/base_velocity/leash_distance`.
- **Logged:** `Curriculum/terrain_levels/stage`, `/level`, `/forward_only_share`, `/stair_steps` (mean step count of
  robots on stairs), and `/stairs_left`, the share of stair episodes that got the base 2 m from the tile centre
  (past the first edge), and `Metrics/base_velocity/max_distance`. Per stair direction (`down_`, `up_`), as running
  means over about 200 episodes: `_left` (got 2 m out), `_falls` (ended by a fall), `_stuck` (commanded more than 4 m
  and never got 2 m out); and over all robots on that type: `_steps` (mean step count), `_height_cm` (mean step
  height) and `_top` (share on 6 steps). The eval viewer shows them under Curriculum.
- **Symmetry:** every PPO mini-batch is also trained on its left-right mirror (rsl_rl's symmetry augmentation,
  `mdp.mirror_left_right`): left and right legs swapped with the hip signs flipped, y and turning negated, the clock
  phases of the two leg pairs swapped, the height map and height grid flipped in y. Without it the policy fell to
  its right on stairs. A few rare states (mostly the first steps after a dropped start) give very large actor
  outputs; both this and the mirror loss (`CLOCK_SCAN_SYMMETRY=loss`, `_MIRROR_LOSS_COEF`) push hardest there, and
  the mirror loss broke a run in one update. `CLOCK_SCAN_SYMMETRY=0` turns it off.
- **Reset on tilt:** `bad_orientation` ends an episode past 0.8 rad of tilt (`CLOCK_SCAN_TILT_LIMIT`) against the
  fitted ground plane. Against gravity (`CLOCK_SCAN_GROUND_TERMINATION=0`), an 18 x 30 cm stair (0.54 rad) left only
  0.26 rad: on model_46000, 321 of 323 episode ends on 18 cm steps were this reset, nose up 0.7-0.8 rad to keep
  the head off the next step, with the robot still walking.
- **Actions** are clipped to +-10 (`CLOCK_SCAN_CLIP_ACTIONS`; +-2.5 rad from the default pose, the whole joint range),
  so the last_action input is too. At +-100, after a resume that restarted the optimizer, a policy fed its own large
  last action grew it about 10x per step to ~40000 in a few states, and the adaptive learning rate fell to its 1e-5
  floor. A resume that widens the critic now keeps the optimizer state (`cli_args.load_checkpoint`).
  The clip alone did not stop the drift: when the robot gets up from a lying start, any mean past the clip gives the
  same result, so the mean grew to 68 in 3000 iterations, and the log-probs of those samples (and of their mirrors)
  pushed the KL up and the learning rate down. The actor's mean is now 10 * tanh(x / 10)
  (`agents/distribution.py`, `BoundedGaussianDistribution`; bound = the clip), with no new weights, so old
  checkpoints load. The C++ export reads the bare MLP and does not apply the tanh.
- **Height map (actor and critic):** vertical rays turned with the base yaw, as a height map built from the lidars
  and sampled about the base would be: from x -0.2 m (just behind the rear feet) to 1.0 m, y +-0.5 m; 5 cm apart up
  to x 0.4 m and |y| 0.3 m, 10 cm beyond (19 x 17 = 323; `CLOCK_SCAN_MAP_*`). Each value is base height minus ground
  height minus 0.5 m, clipped to +-1 m. The actor's copy has uniform noise of +-0.05 m per ray
  (`CLOCK_SCAN_SCAN_NOISE`) and one offset of up to +-0.03 m per episode for the whole map (`CLOCK_SCAN_MAP_BIAS`).
- **Height grid (critic only):** a 1.6 x 1.0 m grid at 0.1 m (187 rays) under the base, turned with its yaw
  (`CLOCK_SCAN_SCAN_LENGTH`, `_SCAN_WIDTH`, `_SCAN_RESOLUTION`), same values. The base-height reward and the critic's
  base height come from it.
- **Actor input:** 379 = Simple's 48-wide layout, the clock (8), the height map (323), in this order.
  The critic gets 592 (the head clearance last).

## Training

Settings are environment variables prefixed `CLOCK_SCAN_` (see `robots/go2/clock_scan_env_cfg.py`).
Train from the repo root with `python launcher.py` -> train -> ClockScan, or directly:

    PYTHONPATH=source/clock_scan python scripts/rsl_rl/train.py --task=Go2-Clock-Scan --headless

Play on one terrain type, `CLOCK_SCAN_PLAY_TERRAIN` (default `boxes`; the stages are off in play; also `hf_pyramid_slope` and
`hf_pyramid_slope_inv`, which are not in training) at difficulty 0.6-1.0. On stairs the difficulty sets the step count
and the height as in training; `CLOCK_SCAN_PLAY_STEP_HEIGHT` (m) gives 6 steps at exactly that height instead.

    PYTHONPATH=source/clock_scan python scripts/rsl_rl/play.py --task=Go2-Clock-Scan

## Runs

- **`Stairs1/model_60999.pt`**: the last of a chain of resumes from scratch (iterations 0-46000 in runs not kept),
  then 15000 iterations with the tilt reset against the ground plane. On 6 steps at 0.5 m/s straight: up 18 cm
  100% across, 3% fell; 20 cm 95% / 20%; 22 cm 81% / 53%; down 100% across to 22 cm. Best at 0.4 m/s: up 18 cm
  98-100% across, at most 6% fell, with -0.2 to +0.2 m/s sideways; at 0.6 m/s 14-30% fell. Charts and videos in
  the run folder.

## Evaluation and deploy

The MuJoCo eval and the C++ policy runner do not give a height scan, so they cannot run this policy.
Use the Isaac sweep:

    PYTHONPATH=source/clock_scan python scripts/rsl_rl/sweep.py --task=Go2-Clock-Scan --headless \
        --checkpoint logs/rsl_rl/go2_clock_scan/<run>/model_<n>.pt

## Stair chart

`stair_eval.py` puts 8 robots on each of 16 step heights (6 steps over the training range), down and up, walking +x
at one pinned speed from the platform (or, with `--heights 15,18,20`, `--rows` x `--per_tile` robots on 6 steps at
exactly each height in cm), and draws the share that got 2 m out and the share that fell against the step
height, one line per checkpoint (`--compare`) and sideways speed (`--vy`). It writes `stair_eval_<checkpoint>.png`
and `.json` next to the first checkpoint:

    PYTHONPATH=source/clock_scan python scripts/rsl_rl/stair_eval.py --headless \
        --checkpoint logs/rsl_rl/go2_clock_scan/<run>/model_<n>.pt --compare <older .pt> --vy 0,-0.2,0.2

## Video

`CLOCK_SCAN_PLAY_CMD_X` pins every robot to one forward speed from a standing start at heading `CLOCK_SCAN_PLAY_YAW`
(default 0, +x), with no pushes. Rows run along x, so with `CLOCK_SCAN_PLAY_ROWS=3` a robot that starts on
one stair pyramid walks down it and up the next:

    CLOCK_SCAN_PLAY_TERRAIN=pyramid_stairs CLOCK_SCAN_PLAY_CMD_X=0.6 CLOCK_SCAN_PLAY_ROWS=3 \
    CLOCK_SCAN_PLAY_STEP_HEIGHT=0.18 \
    PYTHONPATH=source/clock_scan python scripts/rsl_rl/play.py --task=Go2-Clock-Scan --headless --num_envs=1 \
        --video --video_length=950 --video_eye=0,3.5,1.2 --checkpoint=logs/rsl_rl/go2_clock_scan/<run>/model_<n>.pt
