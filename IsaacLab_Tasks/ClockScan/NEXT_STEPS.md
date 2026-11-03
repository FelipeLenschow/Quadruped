# Next steps (after run Lidar2)

Status: the lidar actor climbs 15-22 cm pyramid stairs and the 0.8 m narrow flight in simulation, at 0.3-0.6 m/s
with sideways commands, with at most 9% falls (`Lidar2/wide_34000/`). Best checkpoint: Lidar2/model_37000.

## 1. Training for the real robot

- Noise on the lidar mount pose in training (about +-1 degree, +-1 cm): the mount angle was measured, not exact.
- Slopes in the terrain: the robot slides down a 20 degree pyramid slope.
- Turning at low rates: it turns only 50-65% of the command up to 0.5 rad/s.

## 2. Deploy

- Robot-side lidar node: the filtered L1 cloud into the 16 x 21 fan bins, the same 336 ranges as in training
  (3 m when empty, last reading held), at the policy rate.
- The lidar input in the `.qpol` export and the C++ runner (`quadruped_control`).
- Robot standing on flat ground: the binned real ranges against the simulated ranges, within a few cm.

## 3. Tests

- The full deploy pipeline in Gazebo or MuJoCo with a simulated lidar cloud.
- Real robot: flat ground, then one step, then a full flight, with a person or a harness beside it.

## 4. Housekeeping

- Commit the staged work (lidar actor, narrow stairs, trunk contact, alive bonus, resume options, sweep fix).
- Keep the best checkpoint as a named run, as Stairs1.
