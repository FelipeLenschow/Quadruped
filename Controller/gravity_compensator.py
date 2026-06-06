import os
import numpy as np

try:
    import pinocchio as pin
except ImportError as e:
    raise ImportError(
        "[GravityCompensator] Pinocchio (pin) is required for gravity compensation. "
        "Install it with: pip install pin"
    ) from e


class GravityCompensator:
    """
    Pinocchio-based gravity compensation for the Go2 quadruped.

    Implements the Hybrid Approach (Tier 2):
      - Always: link-mass gravity torques via computeGeneralizedGravity
      - Stance legs: add GRF support torques via Jacobian transpose (J^T * F)

    The robot is modelled as a 19-DoF floating-base system:
      - 7 DoF for the free-flyer base (3 position + 4 quaternion [x,y,z,w])
      - 12 DoF for the joints

    Quaternion convention: [x, y, z, w] (Pinocchio standard).
    """

    # Foot frame names as they appear in the Go2 URDF
    FOOT_FRAME_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

    # Pipeline joint order (type-grouped: 4 hips, 4 thighs, 4 calves)
    PIPELINE_JOINT_NAMES = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    ]

    def __init__(self, urdf_path: str):
        """
        Load the Go2 URDF and build index mappings.

        Args:
            urdf_path: Absolute or project-relative path to the Go2 URDF.
        """
        # Resolve relative paths from project root
        if not os.path.isabs(urdf_path):
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), ".."))
            urdf_path = os.path.join(project_root, urdf_path)

        if not os.path.exists(urdf_path):
            raise FileNotFoundError(
                f"[GravityCompensator] URDF not found: {urdf_path}")

        # Build floating-base model (7 base DoF + 12 joint DoF = 19 DoF)
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data = self.model.createData()

        # Verify model dimensions
        assert self.model.nq == 19, (
            f"Expected 19 generalized coords (7 base + 12 joints), got {self.model.nq}")
        assert self.model.nv == 18, (
            f"Expected 18 velocity DoF (6 base + 12 joints), got {self.model.nv}")

        # Build pipeline-order → Pinocchio-order joint index mapping
        self._pipeline_to_pin_q = []   # indices into q_full[7:]
        self._pipeline_to_pin_v = []   # indices into v_full[6:]
        for name in self.PIPELINE_JOINT_NAMES:
            jid = self.model.getJointId(name)
            if jid >= self.model.njoints:
                raise RuntimeError(
                    f"[GravityCompensator] Joint '{name}' not found in URDF. "
                    f"Available: {[self.model.names[i] for i in range(self.model.njoints)]}")
            # q index offset by 7 (FreeFlyer uses 7 coords: x,y,z,qx,qy,qz,qw)
            self._pipeline_to_pin_q.append(self.model.joints[jid].idx_q - 7)
            # v index offset by 6 (FreeFlyer uses 6 velocity DoF)
            self._pipeline_to_pin_v.append(self.model.joints[jid].idx_v - 6)

        # Resolve foot frame IDs for Jacobian computation
        self._foot_frame_ids = []
        for fname in self.FOOT_FRAME_NAMES:
            fid = self.model.getFrameId(fname)
            if fid >= len(self.model.frames):
                raise RuntimeError(
                    f"[GravityCompensator] Frame '{fname}' not found in URDF.")
            self._foot_frame_ids.append(fid)

        # Total robot mass (for GRF calculation)
        self._total_mass = sum(
            self.model.inertias[i].mass for i in range(self.model.njoints))
        self._gravity_magnitude = np.linalg.norm(self.model.gravity.linear)

        print(f"[GravityCompensator] Loaded Go2 model: "
              f"{self.model.njoints-1} joints, "
              f"total mass = {self._total_mass:.2f} kg, "
              f"|g| = {self._gravity_magnitude:.2f} m/s²")

    def _build_q_full(self, quat_xyzw, joint_pos):
        """
        Construct the 19-dim generalized coordinate vector.

        Args:
            quat_xyzw: [x,y,z,w] base orientation from IMU
            joint_pos: 12-dim joint positions in pipeline order

        Returns:
            q_full: 19-dim vector [x,y,z, qx,qy,qz,qw, j1..j12]
        """
        q_full = pin.neutral(self.model)  # Safe default (normalized quat)

        # Base position: [0,0,0] — gravity comp is translation-invariant
        q_full[0:3] = 0.0

        # Base quaternion: already in [x,y,z,w] — Pinocchio format
        q_full[3:7] = quat_xyzw

        # Joint positions: reorder from pipeline order to Pinocchio order
        for i, pin_idx in enumerate(self._pipeline_to_pin_q):
            q_full[7 + pin_idx] = joint_pos[i]

        return q_full

    def _reorder_pin_to_pipeline(self, tau_joints_pin):
        """Reorder 12-dim Pinocchio joint torques to pipeline order."""
        tau_ff = np.zeros(12, dtype=np.float64)
        for i, pin_idx in enumerate(self._pipeline_to_pin_v):
            tau_ff[i] = tau_joints_pin[pin_idx]
        return tau_ff

    def compute(self, quat_xyzw, joint_pos, contact_flags):
        """
        Compute hybrid gravity compensation torques (Tier 2).

        Tier 2 Hybrid WBC:
          1. Compute link-mass gravity torques for ALL joints (baseline)
          2. For stance legs, add GRF support torques via J^T * F

        Args:
            quat_xyzw:     [x,y,z,w] base orientation from IMU
            joint_pos:     12-dim joint positions in pipeline order
            contact_flags: [FL, FR, RL, RR] binary contact (1.0 = stance)

        Returns:
            tau_ff: 12-dim feedforward torques (Nm) in pipeline order
        """
        quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
        joint_pos = np.asarray(joint_pos, dtype=np.float64)
        contact_flags = np.asarray(contact_flags, dtype=np.float64)

        # 1. Build generalized coordinate vector
        q_full = self._build_q_full(quat_xyzw, joint_pos)

        # 2. Compute full-body gravity torques (link-mass compensation)
        pin.computeGeneralizedGravity(self.model, self.data, q_full)
        tau_gravity_full = self.data.g.copy()  # (nv,) = 18 (6 base + 12 joints)

        # 3. Extract joint gravity torques (skip 6 base DoF)
        tau_joints_pin = tau_gravity_full[6:]  # 12-dim in Pinocchio order

        # Reorder to pipeline order
        tau_ff = self._reorder_pin_to_pipeline(tau_joints_pin)

        # 4. Add GRF support for stance legs (Hybrid WBC)
        n_stance = int(np.sum(contact_flags > 0.5))
        if n_stance > 0:
            # Update kinematics for Jacobian computation
            pin.forwardKinematics(self.model, self.data, q_full)
            pin.updateFramePlacements(self.model, self.data)

            # Static weight distribution: F = Mg / n_stance per foot (vertical)
            # (CoM-based distribution is a future enhancement — Tier 2+)
            F_per_foot = (self._total_mass * self._gravity_magnitude) / n_stance
            F_world = np.array([0.0, 0.0, F_per_foot])  # Vertical support force

            for leg_idx in range(4):
                if contact_flags[leg_idx] <= 0.5:
                    continue

                fid = self._foot_frame_ids[leg_idx]

                # Get 6×nv Jacobian in LOCAL_WORLD_ALIGNED frame
                J = pin.computeFrameJacobian(
                    self.model, self.data, q_full, fid,
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )

                # Extract linear part (3×nv), joint columns only (skip 6 base DoF)
                J_lin_joints = J[:3, 6:]  # (3, 12) in Pinocchio order

                # tau_contact = J^T @ F (for this leg's joints in Pinocchio order)
                tau_contact_pin = J_lin_joints.T @ F_world  # (12,)

                # Reorder and add to pipeline tau_ff
                tau_ff += self._reorder_pin_to_pipeline(tau_contact_pin)

        return tau_ff.astype(np.float32)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    urdf = os.path.join(
        os.path.dirname(__file__), "..", "Configs", "go2_model", "go2.urdf")

    gc = GravityCompensator(urdf)

    # Identity quaternion (upright) in [x,y,z,w]
    quat = np.array([0.0, 0.0, 0.0, 1.0])

    # Standing configuration
    joint_pos = np.array([
        0.1, -0.1, 0.1, -0.1,   # hips
        0.8, 0.8, 1.0, 1.0,     # thighs
        -1.5, -1.5, -1.5, -1.5  # calves
    ])

    # Test 1: No contact (link-mass only)
    contact_none = np.array([0.0, 0.0, 0.0, 0.0])
    tau_no_contact = gc.compute(quat, joint_pos, contact_none)
    print("\n=== Gravity Compensation Self-Test ===")
    print(f"  Link-mass only (no contact):")
    print(f"    Hip torques:   {tau_no_contact[:4]}")
    print(f"    Thigh torques: {tau_no_contact[4:8]}")
    print(f"    Calf torques:  {tau_no_contact[8:12]}")

    # Test 2: All feet in contact (hybrid WBC)
    contact_all = np.array([1.0, 1.0, 1.0, 1.0])
    tau_all_contact = gc.compute(quat, joint_pos, contact_all)
    print(f"\n  Hybrid WBC (4 feet contact):")
    print(f"    Hip torques:   {tau_all_contact[:4]}")
    print(f"    Thigh torques: {tau_all_contact[4:8]}")
    print(f"    Calf torques:  {tau_all_contact[8:12]}")

    print(f"\n  GRF contribution (stance - no_contact):")
    delta = tau_all_contact - tau_no_contact
    print(f"    Hip:   {delta[:4]}")
    print(f"    Thigh: {delta[4:8]}")
    print(f"    Calf:  {delta[8:12]}")
    print(f"\n  Total mass: {gc._total_mass:.2f} kg")
    print(f"  Expected GRF per foot: {gc._total_mass * 9.81 / 4:.2f} N")
