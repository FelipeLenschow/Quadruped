import os
import sys
import time
import threading
import numpy as np

import mujoco
import mujoco.viewer

class MujocoTwin:
    """
    Passive MuJoCo Digital Twin Renderer.
    Only responsible for initializing MuJoCo and updating the visualization.
    """
    def __init__(self, robot_type="go2"):
        self.robot_type = robot_type

        # 1. Load MuJoCo Model
        # Needs to point to the correct path where mujoco_menagerie is.
        # Since it's in DigitalTwin now, the menagerie is in Mujoco/mujoco_menagerie or we use absolute path.
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Mujoco"))
        mjcf_path = os.path.join(
            base_dir, "mujoco_menagerie", "unitree_go2", "scene.xml"
        )
        if not os.path.exists(mjcf_path):
            mjcf_path = os.path.join(base_dir, "scene.xml")

        print(f"[MujocoTwin] Loading MuJoCo Twin Model from {mjcf_path}")
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)

        # Disable Gravity and Collisions
        self.model.opt.gravity[:] = 0.0
        self.model.geom_conaffinity[:] = 0
        self.model.geom_contype[:] = 0

        # Resolve joint indices
        self.isaac_names = [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
        ]
        self.qpos_addr = np.zeros(12, dtype=int)
        for i, name in enumerate(self.isaac_names):
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j_id != -1:
                self.qpos_addr[i] = self.model.jnt_qposadr[j_id]

        # 2. State Variables
        self.base_pos = np.array([0.0, 0.0, 0.35], dtype=np.float64)
        self.base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.joint_pos = np.zeros(12, dtype=np.float64)

        self.lock = threading.Lock()
        self.running = True

        # 3. Start Viewer Thread
        self.viewer_thread = threading.Thread(target=self._viewer_loop, daemon=True)
        self.viewer_thread.start()

    def update_state(self, joint_pos: np.ndarray, base_quat: np.ndarray, base_pos: np.ndarray):
        """
        Updates the internal state to be rendered in the next frame.
        """
        with self.lock:
            self.joint_pos = np.copy(joint_pos)
            self.base_quat = np.copy(base_quat)
            self.base_pos = np.copy(base_pos)

    def stop(self):
        self.running = False

    def _viewer_loop(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            track_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
            if track_id == -1:
                track_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
            if track_id != -1:
                viewer.cam.trackbodyid = track_id

            while self.running and viewer.is_running():
                with self.lock:
                    self.data.qpos[0:3] = self.base_pos
                    self.data.qpos[3:7] = self.base_quat
                    for i, addr in enumerate(self.qpos_addr):
                        if addr > 0:
                            self.data.qpos[addr] = self.joint_pos[i]

                # Kinematic Forward (NO mj_step!)
                mujoco.mj_forward(self.model, self.data)
                
                # Update Viewer
                viewer.sync()
                
                # Render at ~60 Hz
                time.sleep(1.0 / 60.0)
