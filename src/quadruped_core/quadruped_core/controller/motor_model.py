import numpy as np

# The Go2 knee is the hip/thigh motor behind an extra 45.43 / 23.7 reduction: joint torque up, speed down.
GO2_KNEE_RATIO = 45.43 / 23.7


class Go2MotorModel:
    """Go2 torque-speed curve, the one training uses (UnitreeActuatorCfg_Go2HV with the knee reduction).

    Peak torque is Y1 when torque and speed point the same way and Y2 when they oppose, held up to X1
    rad/s and falling linearly to zero at X2: hip and thigh 20.2 / 23.4 N*m, 13.5 / 30 rad/s; calf
    38.7 / 44.9 N*m, 7.04 / 15.65 rad/s.
    """

    def __init__(self, joint_names):
        ratio = np.array([GO2_KNEE_RATIO if "calf" in n else 1.0 for n in joint_names])
        self.y1 = 20.2 * ratio
        self.y2 = 23.4 * ratio
        self.x1 = 13.5 / ratio
        self.x2 = 30.0 / ratio

    def clip(self, torque, vel):
        torque = np.asarray(torque, dtype=np.float64)
        speed = np.abs(vel)
        peak = np.where(vel * torque > 0, self.y1, self.y2)
        fade = np.clip(peak * (1.0 - (speed - self.x1) / (self.x2 - self.x1)), 0.0, None)
        limit = np.where(speed < self.x1, peak, fade)
        return np.clip(torque, -limit, limit)
