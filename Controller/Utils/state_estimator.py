"""
state_estimator.py
------------------
Body-state estimation for real-robot deployment.

Currently implements a contact-aided linear velocity estimator based on
IMU accelerometer integration + no-slip contact decay.

Future extensions:
  - EKF-based estimator
  - Leg odometry (foot velocity from forward kinematics)
  - Height estimation from foot contact + joint encoders
  - Terrain-aware inclination estimation
"""

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rot_from_quat(quat_wxyz) -> np.ndarray:
    """Return 3×3 rotation matrix (body → world) from quaternion [w, x, y, z]."""
    w, x, y, z = quat_wxyz
    return np.array([
        [1 - 2*y**2 - 2*z**2,  2*x*y - 2*w*z,       2*x*z + 2*w*y],
        [2*x*y + 2*w*z,         1 - 2*x**2 - 2*z**2,  2*y*z - 2*w*x],
        [2*x*z - 2*w*y,         2*y*z + 2*w*x,        1 - 2*x**2 - 2*y**2],
    ])


# ---------------------------------------------------------------------------
# Contact-aided IMU velocity estimator
# ---------------------------------------------------------------------------

class StateEstimator:
    """
    Contact-aided linear velocity estimator for real-robot deployment.

    Algorithm
    ---------
    1. Gravity compensation:
         a_linear_body = f_imu + R.T @ g_world
       where f_imu is the IMU specific force (accelerometer reading) and
       g_world = [0, 0, -9.81]. When upright and stationary this yields 0.

    2. Velocity integration:
         v += a_linear_body * dt

    3. Contact-based decay (no-slip assumption):
       When feet are on the ground the foot velocity ≈ 0, so the body
       velocity should decay toward zero. Decay rate scales with number
       of grounded feet.

    Contact decay table (per 50 Hz step):
      0 feet  → 0.999  (airborne — almost no decay)
      1 foot  → 0.995
      2 feet  → 0.980  (trot support phase)
      3 feet  → 0.950
      4 feet  → 0.900  (standing — decays to ~0 in 0.2 s)

    Limitations
    -----------
    - IMU bias and noise cause long-term velocity drift, mitigated by the
      contact decay.
    - No-slip assumption breaks during slip events or on icy surfaces.
    - For precise tasks, upgrade to EKF or leg-odometry estimator.
    """

    _CONTACT_DECAY = {0: 0.999, 1: 0.995, 2: 0.98, 3: 0.95, 4: 0.90}

    def __init__(self, dt: float = 0.02):
        """
        Args:
            dt: Control timestep in seconds (default 0.02 = 50 Hz).
        """
        self.dt = dt
        self._v_body  = np.zeros(3, dtype=np.float64)
        self._g_world = np.array([0.0, 0.0, -9.81])

    # ------------------------------------------------------------------
    def reset(self):
        """Reset velocity estimate to zero (call on robot restart / fall)."""
        self._v_body[:] = 0.0

    # ------------------------------------------------------------------
    def update(self, quat_wxyz, accel_body, feet_contact) -> np.ndarray:
        """
        Perform one estimation step.

        Args:
            quat_wxyz   : Orientation quaternion [w, x, y, z] (body → world).
            accel_body  : IMU accelerometer reading in body frame [ax, ay, az]
                          (specific force ≈ [0, 0, +9.81] when upright & still).
            feet_contact: [FL, FR, RL, RR] binary contact flags (1.0 = contact).

        Returns:
            Estimated linear velocity in body frame (np.ndarray, shape [3]).
        """
        R       = rot_from_quat(quat_wxyz)
        g_body  = R.T @ self._g_world                          # gravity in body frame
        a_lin   = np.array(accel_body, dtype=np.float64) + g_body  # gravity-compensated

        # Integrate
        self._v_body += a_lin * self.dt

        # Contact decay
        n = int(sum(1 for c in feet_contact if c > 0.5))
        self._v_body *= self._CONTACT_DECAY.get(n, 0.999)

        return self._v_body.copy()

    # ------------------------------------------------------------------
    @property
    def velocity(self) -> np.ndarray:
        """Current velocity estimate (body frame)."""
        return self._v_body.copy()
