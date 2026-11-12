"""The lidar policy's fan ranges as spheres in the MuJoCo viewer: one per ray, at the range the policy reads.

Green: a ray with a lidar point. Blue: a filled ray (no point in its cone). Red: more than 20 cm off the exact ray.
"""

import numpy as np
import mujoco

POINT = np.array([0.1, 0.9, 0.1, 1.0], np.float32)
FILLED = np.array([0.2, 0.5, 1.0, 1.0], np.float32)
OFF = np.array([1.0, 0.1, 0.1, 1.0], np.float32)


class FanOverlay:
    SIZE = 0.018
    OFF_M = 0.2

    def __init__(self):
        self.ends = None

    def update(self, origin, dirs_world, ranges, exact, filled=None):
        self.ends = origin + dirs_world * np.asarray(ranges, np.float64)[:, None]
        col = np.tile(POINT, (len(ranges), 1))
        if filled is not None:
            col[filled] = FILLED
        col[np.abs(np.asarray(ranges) - exact) > self.OFF_M] = OFF
        self.rgba = col

    def draw(self, scn):
        if scn is None or self.ends is None:
            return
        size, eye = np.array([self.SIZE, 0.0, 0.0]), np.eye(3).ravel()
        for p, c in zip(self.ends, self.rgba):
            if scn.ngeom >= scn.maxgeom:
                return
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, size, p, eye, c)
            g.emission = 0.5
            scn.ngeom += 1
