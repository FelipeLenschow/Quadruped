"""ClockScan's actor lidar in MuJoCo: ranges along the policy's fixed fan of rays (quadruped_control.LidarFan).

The rays start at the fan's pos in the base frame and turn with the whole base, rows by depression and columns by
azimuth (the order Isaac Lab's fan_pattern makes). They hit only geom group 0 (the floor and terrain), as the
training rays hit only the ground; no hit within max_range reads max_range.
"""

import numpy as np
import mujoco


class LidarFan:
    def __init__(self, model, fan, body="base"):
        self.model = model
        self.body = model.body(body).id
        self.pos = np.asarray(fan.pos, dtype=np.float64)
        self.max_range = float(fan.max_range)
        dep = np.radians(np.asarray(fan.depressions, dtype=np.float64))
        az = np.radians(np.asarray(fan.azimuths, dtype=np.float64))
        d, a = np.meshgrid(dep, az, indexing="ij")
        self.dirs = np.stack([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), -np.sin(d)], axis=-1).reshape(-1, 3)
        self.n = len(self.dirs)
        self.group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        self.geomid = np.zeros(self.n, dtype=np.int32)
        self.dist = np.zeros(self.n, dtype=np.float64)
        self.origin = np.zeros(3)
        self.hits = np.zeros((self.n, 3))

    def ranges(self, data):
        rot = data.xmat[self.body].reshape(3, 3)
        self.origin = data.xpos[self.body] + rot @ self.pos
        vec = (self.dirs @ rot.T).ravel()
        args = (self.model, data, self.origin, vec, self.group, 1, -1, self.geomid, self.dist)
        # No cutoff: it drops geoms by the distance to their centre, and the floor plane's is the world origin.
        try:
            mujoco.mj_multiRay(*args, None, self.n, mujoco.mjMAXVAL)
        except TypeError:  # MuJoCo before the normal argument
            mujoco.mj_multiRay(*args, self.n, mujoco.mjMAXVAL)
        out = np.where(self.dist < 0.0, self.max_range, np.minimum(self.dist, self.max_range))
        self.hits = self.origin + vec.reshape(-1, 3) * out[:, None]
        return out.astype(np.float32)
