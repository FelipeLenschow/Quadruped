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
  2. 15% flat, 15% smooth, 25% rough (to 4 cm), 25% boxes (2-10 cm), 10% each pyramid slope up and down (to 0.4
     rise over run, 22 deg; `CLOCK_SCAN_SLOPE`; none up to Lidar2).
  3. 50% square stairs (`CLOCK_SCAN_STAIRS_SHARE`; half down from a platform on top, half up from a pit), the rest
     in stage 2's mix.
  4. Half of the stair robots (`CLOCK_SCAN_NARROW_SHARE`) moved to narrow stairs, half down and half up.

  A stage opens after 3 checks in a row (`CLOCK_SCAN_STAGE_HOLD`, one per episode length of steps) with the full
  speed range, mean tracking above 0.65 x its weight (`CLOCK_SCAN_STAGE_TRACK`), for stage 3 a mean boxes level
  of at least 0.4 of the rows (`CLOCK_SCAN_STAGE_BOXES_DIFFICULTY`), and for stage 4 a mean square-stairs level of at
  least 0.75 of the rows, the 6-step block (`CLOCK_SCAN_STAGE_STAIRS_DIFFICULTY`). A robot moved to another type starts it at level 0; as many robots
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
- **Narrow stairs** (`mdp.NarrowStairsTerrainCfg`, `narrow_stairs` down, `narrow_stairs_inv` up): two floors joined
  by one straight flight of 15 steps, 0.3 m deep (`CLOCK_SCAN_NARROW_STEPS`), in the middle of the tile. Its levels
  use the same 4 blocks: the flight is 1.6, 1.2, 1.0 and 0.8 m wide (`CLOCK_SCAN_NARROW_WIDTHS`), the height rising
  from 5 to 22 cm inside each, so the top row is a 2.7-3.3 m storey on an 0.8 m flight. Each side of the flight, and
  the upper floor's edge on that side of the stairwell, is open (40%), has balusters 12 cm apart with a handrail at
  0.9 m (40%), or a wall (20%). In the 10 m tiles (`CLOCK_SCAN_TILE_SIZE`, all types) the lower floor runs 3 m
  before the flight (`CLOCK_SCAN_NARROW_APPROACH`) and the upper floor 2.5 m after it, with a 1 m wall on the tile edges. The robot starts before the flight
  (up) or on the upper floor (down), facing it within 0.3 rad and 0.3 x 0.15 m of the start
  (`CLOCK_SCAN_ALIGNED_X`, `_Y`, `_YAW`; `mdp.reset_root_state_aligned`, `reset_start_pose`). It never turns by
  itself: an operator in the command holds the flight's heading (turning 1.0 x the heading error, at most 0.5 rad/s;
  `CLOCK_SCAN_STEER_GAIN`, `_MAX`), and the sideways command is halved (`CLOCK_SCAN_STEER_Y_SCALE`), so it still
  pushes robots toward the edges. Going over an edge ends the episode (`falling`: the base down faster than
  3 m/s, `CLOCK_SCAN_FALL_SPEED`) at 750 on top of the fall penalty (`CLOCK_SCAN_W_FALL_OFF`). Logged as
  `narrow_down_` and `narrow_up_`, with `_width_cm` in place of `_steps`. Stairs1 never saw this: on 0.8 m at 10 cm
  it got 0% up and 34% down, crouching onto the steps (see Ground below).
- **Ground under the robot** (`mdp.ground_plane`): a plane fitted to the critic's height grid from the rays within
  0.25 m of the base's line and within 0.7 m of their median height. The base height (reward and critic) is that
  plane at the base, and the tilt penalty and reset use its slope. With the whole grid, the floor beside an 0.8 m
  flight put a robot standing 0.28 m above the steps at 0.79 m, and the tilt was off by up to the 35 degree clip.
- **Stairs:** 70% of the robots on stairs get forward-only commands: 0 <= x <= 0.6 m/s, sideways kept, turning x0.3
  (`CLOCK_SCAN_FORWARD_X_MAX`, `_Y_SCALE`, `_YAW_SCALE`), and on narrow stairs the heading operator; only these episodes
  move the stair levels. The others get any command, also over an open edge, so the policy learns not to follow one
  there (`CLOCK_SCAN_FORWARD_END` 0.7, 1.0 up to Lidar2; at 0.3, the policy stopped before steps it climbed before, after
  2800 iterations: narrow up 18 cm 98% to 16% across, few falls; the share falls to it over `_RAMP_ITERS` after `_HOLD_ITERS`). On stair tiles a fall costs the same as elsewhere (`CLOCK_SCAN_STAIRS_FALL_SCALE`, 0.25 up to Lidar1), and every step alive gives 1 per second (`CLOCK_SCAN_W_ALIVE`): with the exploration noise, a step on stairs was worth about 0, so a fall cost only 5; a foot hitting
  a vertical face and body contacts cost the same as elsewhere (`CLOCK_SCAN_STAIRS_STUMBLE_SCALE`,
  `_CONTACT_SCALE`, 1.0).
- **Stumble:** a foot hits a vertical face when its sideways contact force is more than 2x its vertical force
  (`CLOCK_SCAN_STUMBLE_RATIO`; the friction is at most 1.2, so a slipping foot stays under it), 5 per second
  (`CLOCK_SCAN_W_STUMBLE`). At 0.25x on stairs, 2 per second and a 4x ratio, about one foot swing in five hit a riser
  going up, the rear legs most.
- **Body tilt** (`mdp.ground_orientation_l2`, 2.5 per unit, `CLOCK_SCAN_W_ORIENT`): the tilt penalty is measured
  against the ground plane (Ground under the robot; slope clipped at 35 degrees), not against gravity, so on stairs
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
- **Trunk contact:** 20 per second while the base touches anything (`CLOCK_SCAN_W_BASE_CONTACT`; 5 in the first run with it, after which the falls near the top of the stairs grew), and the episode ends
  only when the touch lasts over 1 s (`mdp.contact_longer_than`, `CLOCK_SCAN_BASE_CONTACT_TIME`; 0: any touch ends
  it, as before). On model_30000, all 64 episode ends going up 18 cm steps were a touch: nose up on the first
  steps, rear legs low, belly on the first step's edge.
- **Actions** are clipped to +-10 (`CLOCK_SCAN_CLIP_ACTIONS`; +-2.5 rad from the default pose, the whole joint range),
  so the last_action input is too. At +-100, after a resume that restarted the optimizer, a policy fed its own large
  last action grew it about 10x per step to ~40000 in a few states, and the adaptive learning rate fell to its 1e-5
  floor. A resume that widens the critic now keeps the optimizer state (`cli_args.load_checkpoint`).
  The clip alone did not stop the drift: when the robot gets up from a lying start, any mean past the clip gives the
  same result, so the mean grew to 68 in 3000 iterations, and the log-probs of those samples (and of their mirrors)
  pushed the KL up and the learning rate down. The actor's mean is now 10 * tanh(x / 10)
  (`agents/distribution.py`, `BoundedGaussianDistribution`; bound = the clip), with no new weights, so old
  checkpoints load. The C++ export reads the bare MLP and does not apply the tanh.
- **Height map (critic; actor with `CLOCK_SCAN_ACTOR_SCAN=map`):** vertical rays turned with the base yaw, as a height map built from the lidars
  and sampled about the base would be: from x -0.2 m (just behind the rear feet) to 1.0 m, y +-0.5 m; 5 cm apart up
  to x 0.4 m and |y| 0.3 m, 10 cm beyond (19 x 17 = 323; `CLOCK_SCAN_MAP_*`). Each value is base height minus ground
  height minus 0.5 m, clipped to +-1 m. The actor's copy has uniform noise of +-0.05 m per ray
  (`CLOCK_SCAN_SCAN_NOISE`) and one offset of up to +-0.03 m per episode for the whole map (`CLOCK_SCAN_MAP_BIAS`).
- **Height grid (critic only):** a 1.6 x 1.0 m grid at 0.1 m (187 rays) under the base, turned with its yaw
  (`CLOCK_SCAN_SCAN_LENGTH`, `_SCAN_WIDTH`, `_SCAN_RESOLUTION`), same values. The base-height reward and the critic's
  base height come from it.
- **Lidar ranges (actor, `CLOCK_SCAN_ACTOR_SCAN=lidar`, the default):** what the L1 measures, so the top of a tall
  step that its riser hides stays hidden. A fan of 16 x 21 = 336 rays from the L1's place on the base (x 0.289 m,
  z -0.047 m; config.yaml's `real_lidar`), turning with the whole base: 16 rows aimed at flat ground 0.07-1.3 m ahead
  of the lidar (x 0.36-1.53 m from the base standing; `CLOCK_SCAN_LIDAR_ROWS`, `_NEAR`, `_FAR`, `_HEIGHT`), 21 columns
  over +-60 deg (`_COLS`, `_AZIMUTH`). Each value is the range from the lidar, clipped to 3 m (no hit reads 3 m,
  `_RANGE`), the current scan only: 336 inputs (`CLOCK_SCAN_LIDAR_LAG=5` adds the scan 0.1 s back, as the 2026-10-30/31
  runs had). Per ray and step: +-0.02 m noise (`_NOISE`), and a
  reading 1-10 steps older (`_STALE_STEPS`) with a probability drawn per episode from 0 to 30% (`_STALE`): an L1 bin
  keeps its last return. Empty rays first read 3 m at 15%; that policy (2026-10-30 run, 21300) climbed 12 cm 94% of
  the time with that noise and 3% without it.
  Per episode, the mount is moved +-1 cm and turned +-1 deg on each axis (`_MOUNT_POS`, `_MOUNT_ROT`;
  `mdp.randomize_ray_mount`), as the error of the measured mount; the ranges are from the moved mount.
  `mdp.lidar_ranges`, pattern `mdp.FanPatternCfg`; the mirror swaps the columns. Plan and deploy notes:
  `LIDAR_RANGES_PLAN.md`.
- **Actor input:** 392 = Simple's 48-wide layout, the clock (8), the lidar ranges (336), in this order (379 with the
  height map, 728 with `CLOCK_SCAN_LIDAR_LAG=5`). The critic gets 592 (the head clearance last). To run or evaluate a height-map checkpoint (Stairs1),
  set `CLOCK_SCAN_ACTOR_SCAN=map`. To resume a height-map checkpoint into the lidar actor, set
  `CLOCK_SCAN_NEW_INPUTS_FROM=56`: the first 56 inputs keep their weights, the lidar ones start at zero.

## Training

Settings are environment variables prefixed `CLOCK_SCAN_` (see `robots/go2/clock_scan_env_cfg.py`).
Train from the repo root with `python launcher.py` -> train -> ClockScan, or directly:

    PYTHONPATH=source/clock_scan python scripts/rsl_rl/train.py --task=Go2-Clock-Scan --headless

Play on one terrain type, `CLOCK_SCAN_PLAY_TERRAIN` (default `boxes`; the stages are off in play) at difficulty 0.6-1.0. On stairs the difficulty sets the step count
and the height as in training; `CLOCK_SCAN_PLAY_STEP_HEIGHT` (m) gives 6 steps at exactly that height instead.

    PYTHONPATH=source/clock_scan python scripts/rsl_rl/play.py --task=Go2-Clock-Scan

## Runs

A checkpoint does not keep the curriculum. To resume on stairs: `CLOCK_SCAN_START_STAGE=3` (the stage; with a stage
above 0 the speed range also starts full, as from +-0.1 m/s on hard levels the tracking stayed under the gate and
the levels only went down), `CLOCK_SCAN_START_LEVEL=15` and `CLOCK_SCAN_START_LEVEL_MIN` (robots at random levels
between the two; 12 for a run that ended in the top block).

- **`Stairs1/model_60999.pt`**: the last of a chain of resumes from scratch (iterations 0-46000 in runs not kept),
  then 15000 iterations with the tilt reset against the ground plane. On 6 steps at 0.5 m/s straight: up 18 cm
  100% across, 3% fell; 20 cm 95% / 20%; 22 cm 81% / 53%; down 100% across to 22 cm. Best at 0.4 m/s: up 18 cm
  98-100% across, at most 6% fell, with -0.2 to +0.2 m/s sideways; at 0.6 m/s 14-30% fell. Charts and videos in
  the run folder.
- **`Lidar1`**: the lidar actor (current scan only) from scratch, 0-30700, any trunk touch ending the episode.
  model_24500 on 6 steps at 0.5 m/s: up 18 cm 92% across, 86% fell; down 100% / 0%. On the 0.8 m flight at
  0.4 m/s: up 18 cm 19% / 77%, down 98% / 6%. Up at 18 cm stopped there: the falls were the belly on the first
  step's edge (see Trunk contact).
- **`Lidar2/model_37000.pt`**: Lidar1's model_30700, then 300 iterations with the trunk touch a penalty (5 per
  second, ended past 1 s), then 6000 with the alive bonus, the full fall penalty on stairs and 20 per second for
  the trunk (`CLOCK_SCAN_START_STAGE=3 CLOCK_SCAN_START_LEVEL=15` on each resume). With 5 per second alone, the falls
  near the top of the stairs grew after 1000 iterations; here every checkpoint from 32500 to 40000 scored 0.95-0.98
  (`compare_40000/`). Pyramid at 0.5 m/s: up and down 15-22 cm 100% across, at most 2% fell. 0.8 m flight at
  0.4 m/s: up 18 cm 95% / 2%, down 100% / 0%. At 0.3-0.6 m/s with -0.2 to +0.2 m/s sideways (`wide_34000/`, on
  model_34000): at most 4% fell on the pyramid and 9% on the flight. Turning is slow below 0.5 rad/s (50-65% of the
  command).
- **`Lidar3/model_46500.pt`**: Lidar2's model_37000, then 7500 iterations with the lidar mount error, pyramid slopes
  and any command on 30% of the stair robots (no operator for them), then 2000 more from levels 12-15
  (`START_LEVEL_MIN=12`, full speed range). It dipped for about 3000 iterations, then came back. Stair test: score
  0.97 (Lidar2 0.98): pyramid 15-22 cm up and down 100% across, at most 1% fell; 0.8 m flight up 18 cm 90% / 4%,
  down 98% / 0%. Command test (`command_eval.py`, no operator): backward, sideways and turning on stairs at most 3%
  fell, against up to 25% for Lidar2 at the open edge; it goes down backward and sideways, but stops on the first
  steps going up backward or sideways (`videos/commands_46500/`). Straight with no operator
  (`command_eval_straight_*`): the open flight up at 18 cm fell 6% at 0.4 m/s and 15% at 0.6 m/s (Lidar2 4% and 6%).

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

With `--narrow --heights 10,15,18` it uses narrow stairs of `--flight_width` m (0.8) instead, with the operator
holding the heading, and across means onto the far floor (5.4 m). Give `--seconds 20` so slow robots can get there.

## Video

`CLOCK_SCAN_PLAY_CMD_X` (and `_CMD_Y`, sideways) pins every robot to one command from a standing start at heading `CLOCK_SCAN_PLAY_YAW`
(default 0, +x), with no pushes. Rows run along x, so with `CLOCK_SCAN_PLAY_ROWS=3` a robot that starts on
one stair pyramid walks down it and up the next:

    CLOCK_SCAN_PLAY_TERRAIN=pyramid_stairs CLOCK_SCAN_PLAY_CMD_X=0.6 CLOCK_SCAN_PLAY_ROWS=3 \
    CLOCK_SCAN_PLAY_STEP_HEIGHT=0.18 \
    PYTHONPATH=source/clock_scan python scripts/rsl_rl/play.py --task=Go2-Clock-Scan --headless --num_envs=1 \
        --video --video_length=950 --video_eye=0,3.5,1.2 --checkpoint=logs/rsl_rl/go2_clock_scan/<run>/model_<n>.pt
