import os
import mujoco
import numpy as np

class Kinematics:
    def __init__(self, robot_type="go2"):
        """
        Initializes the MuJoCo model for kinematics.
        """
        self.robot_type = robot_type.lower()
        
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        robot_folder = f"unitree_{self.robot_type}"
        self.mjcf_path = os.path.join(base_dir, "Mujoco", "mujoco_menagerie", robot_folder, "scene.xml")
        
        if not os.path.exists(self.mjcf_path):
            # Fallback for older repo structure if needed
            self.mjcf_path = os.path.join(base_dir, "Mujoco", "scene.xml")
            
        if not os.path.exists(self.mjcf_path):
            raise FileNotFoundError(f"MJCF not found at: {self.mjcf_path}")

        self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self.data = mujoco.MjData(self.model)
        
        self.foot_names = ["FL", "FR", "RL", "RR"]
        self.foot_ids = []
        for fn in self.foot_names:
            try:
                self.foot_ids.append(self.model.geom(fn).id)
            except KeyError:
                print(f"[WARNING] Foot geom '{fn}' not found in MJCF!")
                self.foot_ids.append(-1)
                
        try:
            self.base_id = self.model.body("base").id
        except KeyError:
            self.base_id = 0
                
    def get_foot_positions(self, q: np.ndarray) -> np.ndarray:
        """
        Computes forward kinematics and returns the [X, Y, Z] positions
        of all 4 feet relative to the base frame.
        q: (12,) joint angles in order [FL, FR, RL, RR]
        """
        assert q.shape == (12,), f"Expected q shape (12,), got {q.shape}"
        
        # In MuJoCo, a floating base robot's qpos is usually 7 (base) + 12 (joints) = 19
        # Assuming the first 7 are base pose and we just leave them at 0
        if self.model.nq >= 19:
            self.data.qpos[7:19] = q
        else:
            # Fallback if no free joint
            self.data.qpos[:] = q
            
        mujoco.mj_kinematics(self.model, self.data)
        
        base_pos = self.data.xpos[self.base_id]
        
        foot_positions = np.zeros((4, 3))
        for i, f_id in enumerate(self.foot_ids):
            if f_id != -1:
                foot_positions[i] = self.data.geom_xpos[f_id] - base_pos
                
        return foot_positions
        
    def get_gravity_compensation(self, q: np.ndarray) -> np.ndarray:
        """
        Returns zero torques for now to avoid computing full inverse dynamics.
        Can be implemented using mj_rne if needed later.
        """
        return np.zeros(12)

if __name__ == "__main__":
    # Test the module
    kin = Kinematics("go2")
    q_test = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5])
    print("Testing get_foot_positions:")
    feet_pos = kin.get_foot_positions(q_test)
    for i, name in enumerate(["FL", "FR", "RL", "RR"]):
        print(f"  {name}: {feet_pos[i]}")
        
    print("\nTesting get_gravity_compensation:")
    tau_g = kin.get_gravity_compensation(q_test)
    print(tau_g)
