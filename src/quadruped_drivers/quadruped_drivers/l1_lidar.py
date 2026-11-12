"""The Go2's L1 lidar in MuJoCo, as the robot sends it on rt/utlidar/cloud (test/mock_go2.cpp has the same).

The pattern is fitted to a real recording (Mcap/Recordings/lio_bag): one beam on a mirror turning once per PERIOD,
along a great circle TILT off the lidar's z axis, whose plane turns about z at AZ_RATE; a return every
PERIOD / SLOTS, only on the half above the lidar's xy plane. A cloud is CLOUD_SLOTS slots (15.4 Hz, about 4200
points) in the lidar's frame at each point's own time, with its time after the cloud's stamp. The ranges are
range_offset too long (real_lidar.range_offset: real_sensors takes it off) plus noise, and the rays hit the robot's
own body (geom groups 0 and 2: terrain and the visual meshes), as the real L1 does. Rays start MIN_RANGE out, and
hits on the base nearer than HOUSING are dropped: the lidar sits inside the head's mesh, and the real L1 sees nothing
of it (nothing nearer than 0.2 m in the recording).
"""

import types

import numpy as np
import mujoco

PERIOD = 4.5585e-3
SLOTS = 591
AZ_RATE = np.radians(1567.45)
TILT = np.radians(2.2)
MIN_EL = np.radians(1.0)
CLOUD_SLOTS = 8400
MAX_RANGE = 30.0
MIN_RANGE = 0.05
HOUSING = 0.15


def rpy_matrix(rpy):
    cr, sr, cp, sp, cy, sy = (f(a) for a in rpy for f in (np.cos, np.sin))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def directions(t):
    """Beam directions in the lidar frame at times t after the scan's start; NaN rows on the lower half."""
    psi = 2.0 * np.pi * t / PERIOD
    phi = AZ_RATE * t
    c, s = np.cos(psi), np.sin(psi)
    u = np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], 1)
    n = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], 1)
    d = c[:, None] * u + s[:, None] * (np.cos(TILT) * np.array([0.0, 0.0, 1.0]) + np.sin(TILT) * n)
    d[d[:, 2] < np.sin(MIN_EL)] = np.nan
    return d


def correct(pts, times, range_offset, mount_R, mount_t, body_box):
    """real_sensors' correction: ranges pulled back by range_offset, the robot's own points (body_box) dropped."""
    r = np.linalg.norm(pts, axis=1)
    keep = r - range_offset >= MIN_RANGE
    pts, times = pts[keep] * (1.0 - range_offset / r[keep])[:, None], times[keep]
    if body_box:
        b = pts @ mount_R.T + mount_t
        lo = np.array([body_box[k][0] for k in "xyz"])
        hi = np.array([body_box[k][1] for k in "xyz"])
        keep = ~((b >= lo) & (b <= hi)).all(axis=1)
        pts, times = pts[keep], times[keep]
    return pts, times


class L1Lidar:
    def __init__(self, model, xyz, rpy, body="base", range_offset=0.12, noise=0.015, seed=0):
        self.model = model
        self.body = model.body(body).id
        self.mount_t = np.asarray(xyz, dtype=np.float64)
        self.mount_R = rpy_matrix(rpy)
        self.range_offset, self.noise = range_offset, noise
        self.rng = np.random.default_rng(seed)
        self.group = np.array([1, 0, 1, 0, 0, 0], dtype=np.uint8)
        self.dt = PERIOD / SLOTS
        self.reset(0.0)

    def reset(self, t0):
        self.t0, self.slot = t0, 0
        self.pts, self.times = [], []

    def step(self, data, t):
        """Casts every slot up to time t from the current pose. Returns the clouds finished: (stamp, xyz, time)."""
        end = int((t - self.t0) / self.dt) + 1
        out = []
        while self.slot < end:
            stop = min(end, (self.slot // CLOUD_SLOTS + 1) * CLOUD_SLOTS)
            self._cast(data, np.arange(self.slot, stop))
            self.slot = stop
            if stop % CLOUD_SLOTS == 0:
                start = (stop - CLOUD_SLOTS) * self.dt
                pts = np.concatenate(self.pts) if self.pts else np.zeros((0, 3))
                times = np.concatenate(self.times) - start if self.times else np.zeros(0)
                out.append((self.t0 + start, pts.astype(np.float32), times.astype(np.float32)))
                self.pts, self.times = [], []
        return out

    def _cast(self, data, slots):
        ts = slots * self.dt
        d = directions(ts)
        ok = np.isfinite(d[:, 0])
        d, ts = d[ok], ts[ok]
        n = len(d)
        if not n:
            return
        rot = data.xmat[self.body].reshape(3, 3)
        origin = data.xpos[self.body] + rot @ self.mount_t
        world = d @ (rot @ self.mount_R).T
        geomid = np.zeros(n, dtype=np.int32)
        dist = np.zeros(n)
        for i in range(n):
            dist[i] = mujoco.mj_ray(self.model, data, origin + MIN_RANGE * world[i], world[i], self.group, 1, -1,
                                    geomid[i:i + 1])
        hit = (dist >= 0.0) & (dist < MAX_RANGE - MIN_RANGE)
        hit &= (dist >= HOUSING - MIN_RANGE) | (self.model.geom_bodyid[geomid] != self.body)
        r = dist[hit] + MIN_RANGE + self.range_offset + self.rng.normal(0.0, self.noise, hit.sum())
        self.pts.append(d[hit] * r[:, None])
        self.times.append(ts[hit])


class L1Fan:
    """The L1 through real_sensors' correction and lidar_fan's binning (FanInput): the fan's ranges as on the robot."""

    def __init__(self, model, fan, lidar_cfg, fan_cfg, body="base"):
        off = float(lidar_cfg.get("range_offset", 0.0))
        self.l1 = L1Lidar(model, lidar_cfg["xyz"], lidar_cfg["rpy"], body, off)
        self.corr = (off, self.l1.mount_R, self.l1.mount_t, lidar_cfg.get("body_box"))
        self.fan, self.cfg = fan, fan_cfg
        self.reset(0.0)

    def reset(self, t0):
        from quadruped_perception.lidar_fan import FanInput

        self.l1.reset(t0)
        self.input = FanInput(self.fan, self.cfg)

    def cast(self, data):
        for stamp, pts, times in self.l1.step(data, data.time):
            pts, times = correct(pts.astype(np.float64), times, *self.corr)
            self.input.cloud(stamp, pts @ self.l1.mount_R.T + self.l1.mount_t, self.l1.mount_t, data.time, times)

    def ranges(self, data, vel_b, joint_names, q):
        from quadruped_perception.lidar_filter import leg_links

        f = self.input
        self.cast(data)
        f.motion(data.time, data.xmat[self.l1.body].reshape(3, 3).copy(), vel_b, data.time)
        f.links = leg_links(types.SimpleNamespace(name=list(joint_names), position=np.asarray(q)))
        r = f.ranges(data.time)
        return (f.bins.prior if r is None else r).copy()

    @property
    def filled(self):
        return self.input.tracker.stale
