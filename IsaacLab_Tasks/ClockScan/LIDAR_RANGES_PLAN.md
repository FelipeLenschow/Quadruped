# Plan: lidar ranges in place of the height map (actor)

Status: the Isaac Lab side is done (2026-10-30; README, Lidar ranges): 16 x 21 rays, 2 frames 5 steps apart, noise,
empty and stale bins. Not done: mount pose noise, the robot-side node, the C++ runner and `.qpol` export, the tests on
the real robot.

## Why

The actor's height map (`map_scanner`) uses vertical rays from above. It sees all of the ground, also where the
real lidar cannot: below a tall step, the top of the step is hidden behind its riser. A map built from the real
lidar would have holes there, and the policy never trained with holes.

The fix: give the actor what the L1 measures, the distance along a fixed set of directions from the lidar.
A ray that hits a riser gives the distance to the riser. Nothing is hidden or made up, so there is no
"unknown cell" to fill.

## Design

- **Rays:** a fixed fan from the real mount pose, about 16 elevation rows x 21 azimuth columns = 336 rays (about
  the size of the current 323-ray map). Aim them at the ground from just ahead of the front feet to about 1.5 m.
  Choose the exact angles so the fan covers the same ground as the current map (x up to 1.0 m, y +-0.5 m) on
  flat ground, and more ahead.
- **Mount pose:** `src/quadruped_bringup/config/config.yaml`, `real_lidar`: xyz [0.28945, 0.0, -0.046825],
  rpy [0.2171, 3.2680, 2.0766] in the base frame. The sensor must turn with the full base orientation (not
  yaw only), as the real lidar does.
- **Value:** range in m, clipped to 3.0 m. No return (sky, past max range) = 3.0 m.
- **History:** the last 2 frames, about 0.1 s apart, so the policy can remember ground that went under the
  body. The lidar cannot see under the body or the rear feet, and the current MLP has no memory. If 2 frames
  is not enough, try 3, or a recurrent actor.
- **Critic:** keep the exact height map (323 rays) and the height grid (187 rays). Only the actor gets the
  real-sensor input.

## Training noise (to match the real L1)

- Range noise about +-0.02 m per ray.
- The L1 scan pattern does not repeat and is sparse: in one policy step only some bins get a new point. Copy
  this: per ray, a random share (start at 10-20%) is empty (= 3.0 m), and a random share keeps its value from
  an earlier step (up to about 0.2 s old).
- The L1 reads ranges 0.12 m too long; `real_sensors` corrects this (`range_offset`), so the training
  needs no offset.
- Small random error on the mount pose (about +-1 degree, +-1 cm), because the tilt was measured.

## Isaac Lab side

1. A `RayCasterCfg` on the base (or `Head_lower`) at the mount pose, with a custom pattern of fixed directions
   (Isaac Lab's `LidarPatternCfg`, or a pattern function like `mdp.FootprintPatternCfg`), alignment to the full
   base orientation, mesh `/World/ground` only (the robot's own body is not hit, as the real filter removes it).
2. An observation term: range = |hit - ray start|, clipped, with the noise, empty bins and old values above,
   and a short history buffer.
3. Replace `height_map` in the actor group with it. Keep `height_map` and `height_scan` in the critic.
4. Mirror rule for the symmetry augmentation (`mdp/symmetry.py`): the fan must be symmetric in azimuth, and the
   mirror swaps the columns left and right. Note: the real mount yaw is not exactly 0; check that the fan is
   still symmetric about the base's x axis after the mount rotation.
5. Resume from the last stairs checkpoint. The actor's first layer gets new inputs: start their weights at zero
   (as `cli_args.load_checkpoint` did when the critic was widened). Expect a drop while the actor learns the new
   input.

## Robot side (deploy)

- A node that takes the filtered L1 cloud (`/lidar/points_filtered`, after `lidar_filter` and the
  `range_offset` fix), turns each point into the lidar frame, and puts it in the bin of its direction.
- Each bin keeps its newest return from the last ~0.2 s. An empty bin reads 3.0 m. Older points can be moved
  into the current lidar frame with the odometry, or simply dropped.
- The node gives the policy the same 336 x 2 values at the policy rate. The C++ runner and the `.qpol` export
  need this new input.

## Tests before training

- On flat ground and on 18 cm stairs in Isaac Sim: draw the ray hits and check the fan covers the planned ground.
- On the real robot standing still: compare the binned real ranges with the simulated ranges for the same pose
  on flat ground. They should agree within a few cm.
- Check how many bins the real L1 fills per policy step (sets the empty-bin share for training).

## Open questions

- Number of rows and columns, and the elevation limits.
- 2 or 3 frames of history, or a recurrent actor.
- Keep part of the old height map as a fallback input, or not.
