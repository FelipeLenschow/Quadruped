"""
estimator.py
------------
Body-state estimation for real-robot deployment.

Implements a 6-state Linear Kalman Filter (LKF) that fuses:
  - Prediction : IMU accelerometer integration (gravity-compensated)
  - Correction : Leg odometry from forward kinematics (no-slip assumption)

State vector:  x = [vx, vy, vz, bax, bay, baz]
                    ← body velocity →  ← accel bias →

The accelerometer bias is co-estimated and subtracted during prediction,
eliminating the long-term integration drift that plagued the old heuristic
contact-decay estimator.

References
----------
  Bloesch et al., "State Estimation for Legged Robots — Consistent Fusion
  of Leg Kinematics and IMU", RSS 2013.
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# Shared geometry utilities (used by TelemetryManager and SupervisorNode)
# ---------------------------------------------------------------------------

_GRAVITY_WORLD = np.array([0.0, 0.0, -9.81])


def rot_from_quat(quat_wxyz) -> np.ndarray:
    """Return 3×3 rotation matrix R (body → world) from quaternion [w, x, y, z]."""
    w, x, y, z = quat_wxyz
    return np.array([
        [1 - 2*y**2 - 2*z**2,  2*x*y - 2*w*z,       2*x*z + 2*w*y],
        [2*x*y + 2*w*z,         1 - 2*x**2 - 2*z**2,  2*y*z - 2*w*x],
        [2*x*z - 2*w*y,         2*y*z + 2*w*x,        1 - 2*x**2 - 2*y**2],
    ], dtype=np.float64)


def projected_gravity_b(quat_wxyz) -> np.ndarray:
    """
    Return the gravity vector expressed in the robot body frame.

    Mirrors Isaac Lab's ``projected_gravity_b`` observation:
        projected_gravity = R^T @ g_world
    where R = rot_from_quat(quat_wxyz)  (body → world).

    Typical values
    --------------
    Upright robot : [~0,  ~0, -9.81]   z-component ≈ -9.81
    On its side   : z-component ≈  0
    Upside down   : z-component ≈ +9.81

    Tilt cosine (used for fall detection)
    --------------------------------------
        base_tilt_cos = -projected_gravity_b(q)[2] / 9.81
        → 1.0  perfectly upright
        → 0.0  robot on its side
        → 0.7  ≈ 45° (training termination default)
    """
    R = rot_from_quat(quat_wxyz)
    return R.T @ _GRAVITY_WORLD


# ---------------------------------------------------------------------------
# Linear Kalman Filter — 6-state body velocity + accelerometer bias
# ---------------------------------------------------------------------------

class StateEstimator:
    """
    6-state Linear Kalman Filter for body velocity estimation.

    State vector
    ------------
        x = [vx, vy, vz, bax, bay, baz]   (all in body frame)
            ← body velocity (m/s) → ← IMU accel bias (m/s²) →

    Predict step (50 Hz, driven by IMU)
    ------------------------------------
        a_corrected = accel_body - bias         # remove estimated bias
        a_linear    = a_corrected + R^T g_world  # gravity compensation
        v_new       = v + a_linear * dt
        bias stays constant (random-walk model)
        P = F P Fᵀ + Q

    Update step (for each grounded foot, from FK leg odometry)
    -----------------------------------------------------------
    The no-slip constraint gives:

        v_foot_world ≈ 0
        ⇒  v_body + ω × r_foot + J @ dq = 0  (in body frame)
        ⇒  v_body = -J @ dq - ω × r_foot

    Each grounded foot provides a 3D velocity measurement z_i.
    Sequential scalar updates are used for numerical stability.

    Parameters
    ----------
    dt : float
        Control timestep in seconds (default 0.02 = 50 Hz).
    q_vel : float
        Process noise variance for velocity states.
    q_bias : float
        Process noise variance for accelerometer bias states.
    r_meas : float
        Measurement noise variance for each leg-odometry velocity component.
    """

    def __init__(
        self,
        dt: float = 0.02,
        q_vel: float = 0.01,
        q_bias: float = 1e-3,
        r_meas: float = 0.05,
    ):
        self.dt     = dt
        self._n     = 6   # state dimension

        # ── Process noise covariance Q ─────────────────────────────────────
        self._Q = np.diag([q_vel, q_vel, q_vel,
                           q_bias, q_bias, q_bias])

        # ── Measurement noise variance (scalar, each component independent) ─
        self._r = r_meas

        # ── State transition matrix F ───────────────────────────────────────
        # v_new = v + (a_imu - bias + g_body) * dt
        # b_new = b
        #
        # Linearised:  [ v ]   [ I   -dt·I ] [ v ]   [ (a_imu+g_body)·dt ]
        #              [ b ] = [ 0     I   ] [ b ] + [         0          ]
        #
        # The off-diagonal block (-dt·I) is the key: it makes the covariance
        # cross-term P[v,b] grow during predict, so Kalman updates on velocity
        # can back-propagate corrections into the bias estimate.
        self._F = np.eye(6)
        self._F[:3, 3:] = -dt * np.eye(3)   # ∂v_new/∂b = -dt·I

        # ── Observation matrix H — measures velocity, not bias ─────────────
        # One full 3D measurement: H = [I₃ | 0₃]
        self._H = np.zeros((3, 6))
        self._H[:, :3] = np.eye(3)

        # ── Initial state & covariance ──────────────────────────────────────
        self._x = np.zeros(6, dtype=np.float64)  # [vx vy vz bax bay baz]
        self._P = np.eye(6, dtype=np.float64) * 1.0

        # Import FK here (lazy) to avoid import cycle during testing
        self._kin = None

    # ------------------------------------------------------------------
    def reset(self):
        """Reset state to zero (call on robot restart or fall recovery)."""
        self._x[:] = 0.0
        self._P[:] = np.eye(6) * 1.0

    # ------------------------------------------------------------------
    def _get_kin(self):
        """Lazy-load kinematics to avoid circular imports."""
        if self._kin is None:
            from .kinematics import Go2Kinematics
            self._kin = Go2Kinematics()
        return self._kin

    # ------------------------------------------------------------------
    def update(
        self,
        quat_wxyz,
        accel_body,
        feet_contact,
        joint_pos=None,
        joint_vel=None,
        gyro_body=None,
    ) -> np.ndarray:
        """
        Perform one estimation step (predict + update).

        Parameters
        ----------
        quat_wxyz   : [w, x, y, z]  orientation quaternion (body → world).
        accel_body  : [ax, ay, az]  IMU specific force in body frame.
                      Upright & still ≈ [0, 0, +9.81].
        feet_contact: [FL, FR, RL, RR]  binary contact flags (1.0 = contact).
        joint_pos   : (12,) joint positions [FL_hip, FL_thigh, FL_calf, FR…].
                      Required for leg-odometry correction.
        joint_vel   : (12,) joint velocities.  Required for leg-odometry.
        gyro_body   : [wx, wy, wz] angular velocity in body frame (rad/s).
                      Used to account for rotation in no-slip equation.

        Returns
        -------
        v_body : np.ndarray, shape (3,)  — estimated linear velocity in body frame.
        """
        quat_wxyz   = np.asarray(quat_wxyz,   dtype=np.float64)
        accel_body  = np.asarray(accel_body,  dtype=np.float64)
        feet_contact = np.asarray(feet_contact, dtype=np.float64)

        R = rot_from_quat(quat_wxyz)

        # ── 1. Predict ─────────────────────────────────────────────────────
        v   = self._x[:3]
        b   = self._x[3:]

        # Gravity-compensated acceleration (bias-corrected)
        g_body   = R.T @ _GRAVITY_WORLD          # gravity in body frame
        a_linear = (accel_body - b) + g_body     # remove bias, then compensate gravity

        x_pred = np.empty(6)
        x_pred[:3] = v + a_linear * self.dt
        x_pred[3:] = b                            # bias random walk: no change

        P_pred = self._F @ self._P @ self._F.T + self._Q

        self._x = x_pred
        self._P = P_pred

        # ── 2. Update — leg odometry ────────────────────────────────────────
        if (joint_pos is not None and joint_vel is not None
                and np.any(feet_contact > 0.5)):

            kin  = self._get_kin()
            omega = (np.asarray(gyro_body, dtype=np.float64)
                     if gyro_body is not None
                     else np.zeros(3))

            # joint_pos / joint_vel layout (ISAAC order): 
            # [FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, RL_thigh, RR_thigh, FL_calf, FR_calf, RL_calf, RR_calf]
            for leg_idx in range(4):
                if feet_contact[leg_idx] <= 0.5:
                    continue

                idx = [leg_idx, leg_idx + 4, leg_idx + 8]
                q_leg  = np.asarray([joint_pos[i] for i in idx], dtype=np.float64)
                dq_leg = np.asarray([joint_vel[i] for i in idx], dtype=np.float64)

                # FK: foot position and Jacobian in body frame
                r_foot = kin.foot_position_body(leg_idx, q_leg)
                J      = kin.foot_jacobian_body(leg_idx, q_leg)

                # No-slip: v_body = -J @ dq - ω × r_foot
                z = -J @ dq_leg - np.cross(omega, r_foot)

                # Sequential Kalman update (one 3D measurement)
                # Innovation
                y = z - self._H @ self._x
                S = self._H @ self._P @ self._H.T + self._r * np.eye(3)
                K = self._P @ self._H.T @ np.linalg.inv(S)

                self._x = self._x + K @ y
                self._P = (np.eye(6) - K @ self._H) @ self._P

        return self._x[:3].copy()

    # ------------------------------------------------------------------
    @property
    def velocity(self) -> np.ndarray:
        """Current velocity estimate (body frame)."""
        return self._x[:3].copy()

    @property
    def accel_bias(self) -> np.ndarray:
        """Current accelerometer bias estimate (body frame, m/s²)."""
        return self._x[3:].copy()

    @property
    def covariance(self) -> np.ndarray:
        """Full 6×6 state covariance matrix."""
        return self._P.copy()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Synthetic test: robot accelerates at 1 m/s² forward for 1 second (50 steps).
    With leg odometry off (no joint data), the LKF is driven purely by IMU.
    With a known bias, check that bias is estimated and velocity is corrected.
    """
    import numpy as np

    est = StateEstimator(dt=0.02, q_vel=0.01, q_bias=1e-4, r_meas=0.02)

    q_upright = np.array([1.0, 0.0, 0.0, 0.0])
    true_bias  = np.array([0.05, 0.0, 0.0])  # 0.05 m/s² forward bias

    print("=== StateEstimator LKF Self-Test ===")
    print("Scenario: 1 m/s² forward acceleration for 1 s (50 Hz), bias=[0.05,0,0]")
    print()

    # ── Test 1: IMU-only (no leg odometry) ─────────────────────────────────
    v_true = 0.0
    for i in range(50):
        v_true += 1.0 * 0.02  # true velocity accumulation

        # IMU reading: specific force = accel + gravity - in upright body frame
        # gravity component in body frame for upright = [0, 0, +9.81]
        # forward accel = 1 m/s^2 → ax = 1.0, with bias added
        accel = np.array([1.0 + true_bias[0], true_bias[1], 9.81 + true_bias[2]])
        v_est = est.update(q_upright, accel, [0, 0, 0, 0])

    print(f"  IMU-only after 1s:")
    print(f"    True velocity:  {v_true:.3f} m/s")
    print(f"    Est. velocity:  {v_est[0]:.3f} m/s  (bias={true_bias[0]:.3f}, no correction)")
    print(f"    Est. bias:      {est.accel_bias[0]:.4f} m/s² (no FK → bias not estimated)")

    # ── Test 2: Stationary with 4-feet contact and FK ──────────────────────
    print()
    print("Scenario: stationary robot, all feet in contact, bias injected")
    est.reset()

    # Simulate the Go2 'home' joint positions and zero velocities
    q_home = np.array([0.0, 0.9, -1.8] * 4)
    dq_zero = np.zeros(12)
    gyro_zero = np.zeros(3)

    for i in range(100):
        # Upright, stationary: accel = [0, 0, 9.81] + bias
        accel = np.array([true_bias[0], true_bias[1], 9.81 + true_bias[2]])
        v_est = est.update(q_upright, accel, [1, 1, 1, 1],
                           joint_pos=q_home, joint_vel=dq_zero, gyro_body=gyro_zero)

    print(f"  Stationary + FK correction after 2s (100 steps):")
    print(f"    Est. velocity:  {v_est}  (should be ≈ [0, 0, 0])")
    print(f"    Est. bias:      {est.accel_bias}  (converging to [{true_bias[0]:.3f}, 0, 0])")
    print()

    bias_err = abs(est.accel_bias[0] - true_bias[0])
    vel_err  = np.linalg.norm(v_est)
    print(f"  Bias estimation error: {bias_err:.4f} m/s²")
    print(f"  Velocity norm (should be ~0): {vel_err:.4f} m/s")
    print(f"  Status: {'OK' if bias_err < 0.02 and vel_err < 0.02 else 'NEEDS TUNING'}")
