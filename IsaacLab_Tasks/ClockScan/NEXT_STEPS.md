# Next steps (after run Lidar3)

Status: the lidar actor climbs 15-22 cm pyramid stairs and the 0.8 m narrow flight in simulation, at 0.3-0.6 m/s
with sideways commands, with at most 9% falls (`Lidar2/wide_34000/`). Lidar3/model_46500 adds the lidar mount error,
slopes and any command on 30% of the stair robots, at about the same stair score.

## 1. Training for the real robot

- Self safety: no heading operator in training. The policy alone must keep off walls, open edges and big steps taken
  sideways, whatever the command. Lidar3 with no operator: the open flight up at 18 cm fell 6% at 0.4 m/s and 15%
  at 0.6 m/s.
- Turning at low rates: it turns only 50-65% of the command up to 0.5 rad/s.
- Rear rays: the fan looks only ahead (+-60 degrees), so a backward command toward a drop is blind. The L1 sees
  360 degrees around it; from a recorded real scan, find the ground behind the robot that the body does not hide.
  If it sees ground behind the rear feet, add rear rays there to the fan.

## 2. Deploy

- Robot-side lidar node: the filtered L1 cloud into the 16 x 21 fan bins, the same 336 ranges as in training
  (3 m when empty, last reading held), at the policy rate.
- The lidar input in the `.qpol` export and the C++ runner (`quadruped_control`).
- Robot standing on flat ground: the binned real ranges against the simulated ranges, within a few cm.

## 3. Tests

- The full deploy pipeline in Gazebo or MuJoCo with a simulated lidar cloud.
- Real robot: flat ground, then one step, then a full flight, with a person or a harness beside it.

## 4. Evaluation

- A command-safety test: sideways, turning and backward commands on the narrow flight and the stairs, falls counted.
