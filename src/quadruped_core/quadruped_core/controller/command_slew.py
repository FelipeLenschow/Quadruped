import numpy as np


class CommandSlewLimiter:
    """Per-axis acceleration limit on the velocity command [vx, vy, wz, ...].

    Speeding up is limited to accel and slowing down (toward zero, or through it on a reversal) to
    decel, so releasing the deadman still stops the robot quickly. A limit of 0 passes that axis
    through. Extra command elements are passed through untouched.
    """

    def __init__(self, lin_accel=0.0, lin_decel=0.0, yaw_accel=0.0, yaw_decel=0.0):
        self.accel = np.array([lin_accel, lin_accel, yaw_accel], dtype=np.float64)
        self.decel = np.array([lin_decel, lin_decel, yaw_decel], dtype=np.float64)
        self.enabled = bool((self.accel > 0).any() or (self.decel > 0).any())
        self.current = np.zeros(3)

    @classmethod
    def from_config(cls, cfg):
        cfg = cfg or {}
        return cls(*(float(cfg.get(k, 0.0)) for k in ("lin_accel", "lin_decel", "yaw_accel", "yaw_decel")))

    def reset(self):
        self.current[:] = 0.0

    def __call__(self, cmd_vel, dt):
        cmd = np.asarray(cmd_vel, dtype=np.float64).ravel()
        if not self.enabled or cmd.size < 3:
            return cmd
        target = cmd[:3]
        cur = self.current
        reversing = cur * target < 0
        slowing = reversing | (np.abs(target) < np.abs(cur))
        goal = np.where(reversing, 0.0, target)
        limit = np.where(slowing, self.decel, self.accel)
        step = np.clip(goal - cur, -limit * dt, limit * dt)
        self.current = np.where(limit > 0, cur + step, target)
        out = cmd.copy()
        out[:3] = self.current
        return out
