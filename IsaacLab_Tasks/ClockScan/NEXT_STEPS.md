# Next steps (after the safety-phase runs, 2026-11-10)

Status: the 360 degree lidar actor, trained with the safety phases and no operator at the end, climbs 15-22 cm
pyramid stairs and the 0.8 m open narrow flight up and down (stair score 0.99), and with any command switched in
on the platform, the steps or the top it falls at most 4% in any cell (ScanImp, model_40000: Scan's
run from scratch with a stronger hip penalty, so the HAA wobble is about 3 times smaller; README, Runs). On the narrow flight it ignores sideways and turning commands, and it goes backward
slowly on steps.

## 1. Final policy

- Done: one run from scratch with the final config (Scan, all safety phases done at 18000); model_35000 is the
  pick (final tests with 256 robots per height and 128 per command cell, and MuJoCo; README, Runs). It is better
  than Lidar4's model_30000 on the narrow flight and the command test.
- Done: ScanImp, model_40000 (Scan's model_35000 and 5000 iterations with `W_HIP_DEV` -1.0): the same stairs and
  commands, HAA wobble about 3 times smaller.
- Done: ScanHide, model_42000 (ScanImp with the L1's hidden rays in training; section 2): better than ScanImp on
  the L1's pattern in MuJoCo. Not yet tested in Isaac.
- Open: the nose is up 2-3 deg on flat ground, with the rear legs bent more than the front.
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

- Done: `quadruped_perception`'s `lidar_fan` (the L1 cloud into the 16 x 36 fan, at 50 Hz, started by
  `real.launch.py`), `real_driver` reading `/lidar/fan_ranges`, the controller standing a lidar policy still on stale
  ranges, the `ScanImp` entry in `policies.json` with its own tilt limits and no `rom` check. Tested on the
  PC with `mock_go2`, a synthetic floor cloud and `real_driver` (README, Evaluation and deploy).
- The lidar input in the `.qpol` export and the C++ runner (`quadruped_control`). Done.
- Open: the soft target clip (`control.saturation_limit` 0.9) costs the narrow flight down at 22 cm; a per-policy
  clip or 1.0 for ScanImp.
- Open: `lidar_fan`'s time per tick on the robot (0.8 ms mean, 1.4 ms at most, on the PC, with the fill).
- Real L1 (`Mcap/Recordings/lio_bag`, 15 Hz, 3800 points per cloud after the filters, replayed through
  `lidar_fan`): the flat floor within 0.6-8 cm per row (median). Most rays are 0.1-0.2 s old; 11% (65 rays) have no point
  for 0.5 s, all behind the robot, where the body hides the floor: the far rows at -170, 160 deg and the column at
  -140 deg. Training rays go through the body. `lidar_fan` fills these from the nearest rays with points: on the
  flat floor 2.6 cm off (median; 7.4 cm at 90%), as good as the rays with points; rays with points left out and
  filled the same way are 1.7 cm off while walking (6.0 cm at 90%). Open: the fill behind the robot on stairs.
- Open: the estimator's vertical velocity on real stairs (it moves `lidar_fan`'s kept points; MuJoCo's true
  velocity was used in the tests, and they also passed with no height change at all).
- MuJoCo with the L1's own pattern (`stairs_mujoco --lidar l1`, ScanImp model_40000, 8 trials): stairs up and down
  18 cm and the narrow flight down 18 and 22 cm all across. The narrow flight up is worse than with the exact fan
  (8/8 at 18 and 22 cm): window 0.15 s 2/8 and 0/8, window 0.5 s 7/8 and 5/8 (8/8 and 3/8 with the estimator).
  The rays the body hides (behind and beside the robot) are filled from the stair surface next to them, also where
  the side of the flight drops away. `window_s` is now 0.5 s, and each ray keeps 3 points with no held range (a
  held range was 40% of the rays more than 20 cm off): narrow up 22 cm 2/8 across (4/8 with the estimator), within
  the noise of 8 trials. 23% of the rays are still more than 20 cm off there: about 57% of those are filled rays
  (most behind the robot), 35% points at a step edge in the ray's bin but on the other surface. The fill was level
  in the base frame: with the body pitched on the flight, rays that point up in the world (no return; 3 m in
  training) read the ground, in the air above the top floor. Level in the world, rays going up reading 3 m: 18%
  off, and all 8/8 with true velocity; with the estimator narrow up 18 cm 7/8 (1 fell), 22 cm 6/8 (none fell).
  Each point is now moved from its own time in the cloud (the 'time' field; before, all from the cloud's start,
  65 ms early): 15% of the rays off; with the estimator narrow up 18 and 22 cm 7/8 (1 fell each), with true
  velocity 22 cm 6/8 (none fell), the rest 8/8. Of the rays off: about 50% filled (most behind the robot), 28% old
  points that a step edge now hides from the lidar, 15% a step edge in the ray's bin, 7% motion. Now a cone with
  any new point drops its older ones and the ray takes the shortest range in the cone (in place of the point
  nearest the ray's centre): 14% of the rays off, all 8/8 with true velocity, with the estimator only narrow up
  22 cm 7/8 (1 fell). On flat ground the shortest range in a cone is 2-7% short of the centre's (the cone's
  steepest edge). Open: the hidden rays behind the robot.
- The red rays' effect (narrow up 22 cm, estimator): the policy's joint targets differ from the same policy on
  exact rays by 0.079 rad on average (0.004 for the training noise), 67% of it from the rays behind the robot.
- Done: ScanHide, model_42000 (ScanImp model_40000 resumed with `LIDAR_HIDE` to 46000: rays hidden as the L1
  misses them, `mdp/l1_hidden.npy` from MuJoCo, and filled as `lidar_fan` fills them; about 14% hidden;
  `logs/train_out/scanimp_hide.log`). MuJoCo, L1 pattern, estimator, narrow flight 22 cm, 80 trials each
  (`ScanHide/mujoco/l1_compare.md`): model_42000 up 80/80 and down 80/80 across, none fell; ScanImp model_40000
  up 76/80 (1 fell), down 78/80 (2 fell). model_42000 is also 16/16 on all six tests with the L1 and with the exact
  fan. The other checkpoints vary much at 22 cm (model_44000 up 1/16, model_45999 up 12/16, model_41500 and 42500
  up 11/16 and 10/16), so model_42000 can be a good point in a noisy run. In training the stair levels stay as
  ScanImp's, falls 2-3% (ScanImp 0.3-1.3%). Open: the Isaac final tests (`stair_eval.py`, `command_eval.py`) on
  model_42000. No LIO for a longer memory: the robot's memory is for the LLM.
- Robot standing on flat ground: the binned real ranges against the simulated ranges, within a few cm.
- Real robot: flat ground, then one step, then a full flight, with a person or a harness beside it.
