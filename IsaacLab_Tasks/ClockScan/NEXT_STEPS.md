# Next steps (after the safety-phase runs, 2026-11-10)

Status: the 360 degree lidar actor, trained with the safety phases and no operator at the end, climbs 15-22 cm
pyramid stairs and the 0.8 m open narrow flight up and down (stair score 0.99), and with any command switched in
on the platform, the steps or the top it falls at most 4% in any cell (Scan, model_35000, one run from
scratch; README, Runs). On the narrow flight it ignores sideways and turning commands, and it goes backward
slowly on steps.

## 1. Final policy

- Done: one run from scratch with the final config (Scan, all safety phases done at 18000); model_35000 is the
  pick (final tests with 256 robots per height and 128 per command cell, and MuJoCo; README, Runs). It is better
  than Lidar4's model_30000 on the narrow flight and the command test.
- Open, smaller: turning at low rates (50-65% of the command up to 0.5 rad/s); forward at 0.6 m/s on the narrow
  flight still falls up to 6%; robots on the upper floor of a flight going down are not stopped at the stairwell edge
  by the reward command (only by the fall cost).

## 2. Sim2sim in MuJoCo

- Done: the C++ runner reads the lidar ranges (the fan is in the `.qpol`), MuJoCo casts the same fan
  (`quadruped_drivers/lidar_fan.py`, ground only), `terrain.py` has `stairs` and `narrow`, and
  `quadruped_drivers.stairs_mujoco` runs the stairs test with no ROS (README, Evaluation and deploy).
- Open: the same in Gazebo (its lidar is a point cloud, so it needs the binning node of step 3); the command test
  (`command_eval.py`) in MuJoCo.

## 3. Deploy

- Robot-side lidar node: the filtered L1 cloud into the 16 x 36 fan bins, the same 576 ranges as in training
  (3 m when empty, last reading held, also for bins a leg hides), at the policy rate.
- The lidar input in the `.qpol` export and the C++ runner (`quadruped_control`).
- Robot standing on flat ground: the binned real ranges against the simulated ranges, within a few cm.
- Real robot: flat ground, then one step, then a full flight, with a person or a harness beside it.
