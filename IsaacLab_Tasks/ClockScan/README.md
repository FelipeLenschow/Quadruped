# ClockScan

Go2 velocity tracking with a trot clock on rough ground, on Isaac Lab with rsl_rl. Cut from
`IsaacLab_Tasks/Simple`: only its `Clock-Rough` task, in one flat config, with a height scan in the actor.

Task: `Go2-Clock-Scan`

- **Same as Simple's Clock-Rough:** the trot clock whose rate follows the command, Walk These Ways' contact
  and swing-height terms, the Deploy sensor model, plant randomization and lying or dropped starts, and the
  tracking curriculum.
- **Terrain:** 60% stairs (30% down, 30% up), step height 5 cm at the easiest row to 20 cm at the hardest;
  15% random rough to 4 cm, 15% boxes 2-10 cm, 10% flat (`CLOCK_SCAN_SHARE_*`, `CLOCK_SCAN_STEP_LO`, `_STEP_HI`).
- **Height scan:** a 1.6 x 1.0 m grid at 0.1 m (187 rays) under the base, turned with its yaw
  (`CLOCK_SCAN_SCAN_LENGTH`, `_SCAN_WIDTH`, `_SCAN_RESOLUTION`). Each value is base height minus ground
  height minus 0.5 m, clipped to +-1 m. The actor gets it with uniform noise of +-0.05 m per ray
  (`CLOCK_SCAN_SCAN_NOISE`), the critic without noise.
- **Actor input:** 243 = Simple's 48-wide layout, the clock (8), the scan (187), in this order.
  The critic gets 265.

## Training

Settings are environment variables prefixed `CLOCK_SCAN_` (see `robots/go2/clock_scan_env_cfg.py`).
Train from the repo root with `python launcher.py` -> train -> ClockScan, or directly:

    PYTHONPATH=source/clock_scan python scripts/rsl_rl/train.py --task=Go2-Clock-Scan --headless

Play on one terrain type, `CLOCK_SCAN_PLAY_TERRAIN` (default `boxes`; also `hf_pyramid_slope` and
`hf_pyramid_slope_inv`, which are not in training) at difficulty 0.6-1.0:

    PYTHONPATH=source/clock_scan python scripts/rsl_rl/play.py --task=Go2-Clock-Scan

## Evaluation and deploy

The MuJoCo eval and the C++ policy runner do not give a height scan, so they cannot run this policy.
Use the Isaac sweep:

    PYTHONPATH=source/clock_scan python scripts/rsl_rl/sweep.py --task=Go2-Clock-Scan --headless \
        --checkpoint logs/rsl_rl/go2_clock_scan/<run>/model_<n>.pt
