"""
kinematics.py
-------------
Forward kinematics and geometric Jacobians for the Unitree Go2 quadruped.

Geometry is loaded from Configs/go2_kinematics.yaml, which was extracted from
the official MuJoCo model (Mujoco/mujoco_menagerie/unitree_go2/go2.xml).

Joint convention (per leg, in Isaac Lab / SDK2 order)
------------------------------------------------------
  index 0 — hip abduction  (rotation about body-frame X)
  index 1 — thigh          (rotation about body-frame Y)
  index 2 — calf / knee    (rotation about body-frame Y)

Leg index mapping
-----------------
  0 → FL,  1 → FR,  2 → RL,  3 → RR

All quantities are expressed in the **robot body frame** unless noted.
"""

from __future__ import annotations
import os
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Load geometry from YAML config
# ---------------------------------------------------------------------------

def _load_kin_config() -> dict:
    """Return the go2_kinematics sub-dict from Configs/go2_kinematics.yaml."""
    base = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(base, "..", "Configs", "go2_kinematics.yaml")
    cfg_path = os.path.normpath(cfg_path)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"[Kinematics] Config not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)["go2_kinematics"]


_cfg = _load_kin_config()

_LEG_NAMES   = ["FL", "FR", "RL", "RR"]

# Hip origins in body frame — shape (4, 3)
_HIP_ORIGINS: np.ndarray = np.array(
    [_cfg["hip_origins"][n] for n in _LEG_NAMES], dtype=np.float64
)

# Y-offset from hip joint to thigh joint (abduction axis end-point)
_THIGH_Y: np.ndarray = np.array(
    [_cfg["thigh_offset_y"][n] for n in _LEG_NAMES], dtype=np.float64
)

_L_THIGH:    float = float(_cfg["thigh_length"])
_L_CALF:     float = float(_cfg["calf_length"])
_FOOT_X_OFF: float = float(_cfg["foot_x_offset"])  # small forward offset


# ---------------------------------------------------------------------------
# Elementary rotation matrices
# ---------------------------------------------------------------------------

def _Rx(a: float) -> np.ndarray:
    """3×3 rotation about X by angle a (radians)."""
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=np.float64)


def _Ry(a: float) -> np.ndarray:
    """3×3 rotation about Y by angle a (radians)."""
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Forward Kinematics
# ---------------------------------------------------------------------------

class Go2Kinematics:
    """
    Forward kinematics and geometric Jacobians for the Unitree Go2.

    Geometry is loaded at import time from Configs/go2_kinematics.yaml.

    Usage
    -----
    kin = Go2Kinematics()
    q_leg = np.array([q_hip, q_thigh, q_calf])   # radians

    # Foot position in body frame
    p_foot = kin.foot_position_body(leg_idx=0, q_leg=q_leg)

    # 3×3 geometric Jacobian  (v_foot_body = J @ dq_leg  +  correction term)
    J = kin.foot_jacobian_body(leg_idx=0, q_leg=q_leg)
    """

    def foot_position_body(self, leg_idx: int, q_leg: np.ndarray) -> np.ndarray:
        """
        Compute foot position in the body frame via 3-DOF FK.

        Chain:
          body → (translate to hip) → R_x(q0) → (translate along Y by thigh_y)
               → R_y(q1) → (translate [0,0,-L_thigh])
               → R_y(q2) → (translate [foot_x_off, 0, -L_calf])

        Parameters
        ----------
        leg_idx : int  — 0=FL, 1=FR, 2=RL, 3=RR
        q_leg   : array of shape (3,)  — [q_hip, q_thigh, q_calf] in radians

        Returns
        -------
        p_foot : np.ndarray, shape (3,)  — foot position in body frame
        """
        q0, q1, q2 = float(q_leg[0]), float(q_leg[1]), float(q_leg[2])

        # 1. Start at hip origin in body frame
        p = _HIP_ORIGINS[leg_idx].copy()

        # 2. Abduction (Rx about hip X-axis)
        R_ab = _Rx(q0)

        # 3. Translate to thigh joint (along abduction-rotated Y)
        thigh_offset = R_ab @ np.array([0.0, _THIGH_Y[leg_idx], 0.0])
        p = p + thigh_offset

        # 4. Thigh rotation (Ry)
        R_th = R_ab @ _Ry(q1)

        # 5. Translate to calf joint
        p = p + R_th @ np.array([0.0, 0.0, -_L_THIGH])

        # 6. Calf rotation (Ry)
        R_ca = R_th @ _Ry(q2)

        # 7. Translate to foot contact point
        p = p + R_ca @ np.array([_FOOT_X_OFF, 0.0, -_L_CALF])

        return p

    def foot_jacobian_body(self, leg_idx: int, q_leg: np.ndarray) -> np.ndarray:
        """
        Compute the 3×3 geometric Jacobian mapping joint velocities to
        foot linear velocity in the body frame:

            v_foot_body = J(q) @ dq_leg

        Derived analytically from the FK chain using the cross-product rule:
            J_i = z_i × (p_foot - p_i)
        where z_i is the joint axis in body frame and p_i is the joint origin.

        Parameters
        ----------
        leg_idx : int
        q_leg   : array of shape (3,)

        Returns
        -------
        J : np.ndarray, shape (3, 3)
        """
        q0, q1, q2 = float(q_leg[0]), float(q_leg[1]), float(q_leg[2])

        # --- Compute all joint origins and axes in body frame ---

        # Joint 0 — hip abduction (Rx)
        p0 = _HIP_ORIGINS[leg_idx].copy()
        z0 = np.array([1.0, 0.0, 0.0])   # X-axis in body frame (abduction)

        # Joint 1 — thigh (Ry, after abduction)
        R_ab = _Rx(q0)
        p1 = p0 + R_ab @ np.array([0.0, _THIGH_Y[leg_idx], 0.0])
        z1 = R_ab @ np.array([0.0, 1.0, 0.0])   # rotated Y-axis

        # Joint 2 — calf (Ry, after abduction+thigh)
        R_th = R_ab @ _Ry(q1)
        p2 = p1 + R_th @ np.array([0.0, 0.0, -_L_THIGH])
        z2 = R_th @ np.array([0.0, 1.0, 0.0])   # rotated Y-axis

        # Foot position
        R_ca = R_th @ _Ry(q2)
        p_foot = p2 + R_ca @ np.array([_FOOT_X_OFF, 0.0, -_L_CALF])

        # --- Jacobian columns: J_i = z_i × (p_foot - p_i) ---
        J = np.column_stack([
            np.cross(z0, p_foot - p0),
            np.cross(z1, p_foot - p1),
            np.cross(z2, p_foot - p2),
        ])
        return J


# Module-level singleton for convenience
_kin = Go2Kinematics()

def foot_position_body(leg_idx: int, q_leg: np.ndarray) -> np.ndarray:
    """Module-level convenience wrapper for Go2Kinematics.foot_position_body."""
    return _kin.foot_position_body(leg_idx, q_leg)

def foot_jacobian_body(leg_idx: int, q_leg: np.ndarray) -> np.ndarray:
    """Module-level convenience wrapper for Go2Kinematics.foot_jacobian_body."""
    return _kin.foot_jacobian_body(leg_idx, q_leg)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Sanity checks using the Go2 'home' keyframe from go2.xml:
        qpos home = ... 0 0.9 -1.8   (hip=0, thigh=0.9, calf=-1.8) per leg
    Expected: feet should be ~0.28–0.30 m below body origin (height ≈ 0.27 m)
    """
    kin = Go2Kinematics()
    q_home = np.array([0.0, 0.9, -1.8])

    print("=== Go2 Kinematics Self-Test (home pose) ===")
    print(f"  thigh={_L_THIGH:.3f}m  calf={_L_CALF:.3f}m")
    print()

    for i, name in enumerate(_LEG_NAMES):
        p = kin.foot_position_body(i, q_home)
        J = kin.foot_jacobian_body(i, q_home)
        print(f"  {name}  foot_pos = [{p[0]:+.4f},  {p[1]:+.4f},  {p[2]:+.4f}]  "
              f"  |J| = {np.linalg.norm(J):.3f}")

    # Numerical Jacobian check for FL
    print("\n=== Numerical vs. Analytical Jacobian (FL) ===")
    eps = 1e-6
    J_ana = kin.foot_jacobian_body(0, q_home)
    J_num = np.zeros((3, 3))
    for j in range(3):
        dq = np.zeros(3); dq[j] = eps
        J_num[:, j] = (kin.foot_position_body(0, q_home + dq) -
                       kin.foot_position_body(0, q_home - dq)) / (2 * eps)
    err = np.max(np.abs(J_ana - J_num))
    print(f"  Max elementwise error: {err:.2e}  ({'OK' if err < 1e-9 else 'FAIL'})")
