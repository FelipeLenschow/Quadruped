import os
import sys

# Ensure absolute path of the repository is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import json
import numpy as np

import mujoco
from mujoco import viewer
import rclpy
from rclpy.node import Node
import argparse
import threading
from pipeline import LocomotionPipeline
from Configs.config_loader import load_config

class MujocoEvaluator(Node):
    def __init__(self, robot_type="go2", checkpoint=None, obs_dim=49, use_estimator=False, headless=True):
        super().__init__("mujoco_evaluator_node")
        self.robot_type = robot_type
        self.cmd_vel = [0.0, 0.0, 0.0, 0.0]
        self.headless = headless

        # 0. Load Central Config
        self.config = load_config()
        self.ctrl_cfg = self.config.get("control", {})
        self.motor_cfg = self.config.get("motor", {})
        self.kp = float(self.ctrl_cfg.get("kp", 0.0))
        self.kd = float(self.ctrl_cfg.get("kd", 0.0))

        # 1. Load MuJoCo Scene
        robot_folder = f"unitree_{self.robot_type.lower()}"
        mjcf_path = os.path.join(
            os.path.dirname(__file__), "mujoco_menagerie", robot_folder, "scene.xml"
        )
        if not os.path.exists(mjcf_path):
            mjcf_path = os.path.join(os.path.dirname(__file__), "scene.xml")

        print(f"[MujocoEvaluator] Initializing for {self.robot_type.upper()}. Model: {mjcf_path}")
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.001

        self._init_physics()

        # 2. Locomotion Pipeline
        self.pipeline = LocomotionPipeline(
            node=self,
            robot_type=robot_type,
            checkpoint=checkpoint,
            obs_dim=obs_dim,
            use_estimator=use_estimator,
            joint_names=self.isaac_names,
            sim_dt=0.001
        )

        # 4. Physics Thread
        self.physics_thread = threading.Thread(target=self._evaluation_loop, daemon=True)
        self.physics_thread.start()

    def _init_physics(self):
        """Initialize MuJoCo physics and resolve joint addresses."""
        self.PD_DECIMATION = 1

        self.isaac_names = [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
        ]
        self.isaac_qpos_addr = np.zeros(12, dtype=int)
        self.isaac_qvel_addr = np.zeros(12, dtype=int)
        self.isaac_ctrl_idx = np.zeros(12, dtype=int)

        for i, name in enumerate(self.isaac_names):
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.isaac_qpos_addr[i] = self.model.jnt_qposadr[j_id]
            self.isaac_qvel_addr[i] = self.model.jnt_dofadr[j_id]
            act_name = name.replace("_joint", "")
            self.isaac_ctrl_idx[i] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name
            )

        self.desired_qpos = np.array([
            0.1, -0.1, 0.1, -0.1,  # hips
            0.8, 0.8, 1.0, 1.0,  # thighs
            -1.5, -1.5, -1.5, -1.5,  # calves
        ], dtype=np.float32)

        self.current_targets = self.desired_qpos.copy()

        for i in range(self.model.nu):
            self.model.actuator_gainprm[i, 0] = 1.0
            self.model.actuator_biasprm[i, 1] = 0.0
            self.model.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_NONE
            self.model.actuator_ctrllimited[i] = 0

        for i in range(self.model.njnt):
            if self.model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
                self.model.dof_damping[self.model.jnt_dofadr[i]] = 0.0
                self.model.dof_frictionloss[self.model.jnt_dofadr[i]] = 0.01

    def _get_raw_sensor_data(self):
        """Extracts raw state vectors from MuJoCo data."""
        q = self.data.qpos[self.isaac_qpos_addr]
        dq = self.data.qvel[self.isaac_qvel_addr]
        quat = self.data.qpos[3:7]  # [w, x, y, z]
        pos = self.data.qpos[:3]

        w, x, y, z = quat
        R = np.array([
            [1-2*y**2-2*z**2, 2*x*y-2*w*z,      2*x*z+2*w*y],
            [2*x*y+2*w*z,     1-2*x**2-2*z**2,  2*y*z-2*w*x],
            [2*x*z-2*w*y,     2*y*z+2*w*x,      1-2*x**2-2*y**2],
        ])

        global_ang_vel = self.data.cvel[1][:3]
        gyro = R.T @ global_ang_vel
        vel_b = R.T @ self.data.qvel[:3]

        try:
            accel = self.data.sensor('accelerometer').data.copy()
        except KeyError:
            accel = np.array([0.0, 0.0, 9.81])

        contact = [0.0, 0.0, 0.0, 0.0]
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = con.geom1, con.geom2
            if g1 == 0 or g2 == 0:
                for foot_idx, foot_name in enumerate(["FL", "FR", "RL", "RR"]):
                    name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g1)
                    name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g2)
                    is_match1 = name1 and name1.lower() != "floor" and foot_name.lower() in name1.lower()
                    is_match2 = name2 and name2.lower() != "floor" and foot_name.lower() in name2.lower()
                    if is_match1 or is_match2:
                        contact[foot_idx] = 1.0

        return {
            'q': q, 'dq': dq, 'quat': quat, 'gyro': gyro, 
            'accel': accel, 'pos': pos, 'vel': vel_b, 'contact': contact
        }

    def _reset_robot(self):
        mujoco.mj_resetData(self.model, self.data)
        for i, addr in enumerate(self.isaac_qpos_addr):
            self.data.qpos[addr] = self.desired_qpos[i]
        self.data.qpos[2] = 0.50
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)

    def _pd_torques(self, targets):
        """Compute DCMotor PD torques matching training (Kp=25, Kd=0.5)."""
        q = self.data.qpos[self.isaac_qpos_addr]
        v = self.data.qvel[self.isaac_qvel_addr]

        pos_err = targets - q  # Target - Actual
        kp = self.kp
        kd = self.kd
        
        effort_limit = self.pipeline.safety_processor.active_max_torque
        
        if self.robot_type.lower() == "a1":
            sat_effort, vel_lim = 33.5, 21.0
        elif self.robot_type.lower() == "go1":
            sat_effort, vel_lim = 23.7, 30.0
        else: # go2
            sat_effort = float(self.motor_cfg.get("max_torque", 45.0))
            vel_lim = float(self.motor_cfg.get("max_velocity", 30.0))
        
        if effort_limit <= 0.1:
            kp = 0.0
            kd = 0.0

        torques = kp * pos_err + kd * (0 - v)
        vel_at_lim = vel_lim * (1 + effort_limit / sat_effort)
        v_clamp = np.clip(v, -vel_at_lim, vel_at_lim)
        t_top = effort_limit * (1.0 - v_clamp / vel_lim)
        t_bot = effort_limit * (-1.0 - v_clamp / vel_lim)
        return np.clip(
            torques, np.minimum(t_bot, -effort_limit), np.minimum(t_top, effort_limit)
        )

    def _evaluation_loop(self):
        speeds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.00]
        steps_per_speed = 32000  # 32 seconds at 1000Hz (30s walking + 2s warmup)
        warmup_steps = 2000     # 2 seconds standing still
        
        results = {}
        
        print("\n" + "="*60)
        print("🚀 STARTING MUJOCO POLICY EVALUATION")
        print("="*60)
        
        self.foot_geom_ids = []
        for fn in ["FL", "FR", "RL", "RR"]:
            for i in range(self.model.ngeom):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
                if name and name.upper() == fn.upper():
                    self.foot_geom_ids.append(i)
                    break
        # Fallback just in case
        if len(self.foot_geom_ids) < 4:
            self.foot_geom_ids = []
            for fn in ["FL", "FR", "RL", "RR"]:
                for i in range(self.model.ngeom):
                    name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
                    if name and fn.lower() in name.lower() and ("calf" in name.lower() or "foot" in name.lower()):
                        self.foot_geom_ids.append(i)
                        break

        class DummyViewer:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def is_running(self): return True
            def sync(self): pass

        viewer_ctx = mujoco.viewer.launch_passive(self.model, self.data) if not self.headless else DummyViewer()

        with viewer_ctx as viewer:
            if not self.headless:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                track_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
                if track_id == -1:
                    track_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
                viewer.cam.trackbodyid = track_id

            for speed in speeds:
                print(f"\n[EVAL] Testing forward velocity: {speed} m/s")
                self._reset_robot()
                
                max_foot_heights = [0.0, 0.0, 0.0, 0.0]
                swing_times = [[], [], [], []]
                stance_force_sum = [0.0, 0.0, 0.0, 0.0]
                stance_force_count = [0, 0, 0, 0]
                max_grf_stance_N = [0.0, 0.0, 0.0, 0.0]
                
                base_z_vels = []
                base_roll_vels = []
                base_pitch_vels = []
                actual_fwd_vels = []
                
                sync_match_count = 0
                eval_steps = 0
                
                last_contact = [0.0, 0.0, 0.0, 0.0]
                current_swing_time = [0.0, 0.0, 0.0, 0.0]
                
                for step in range(steps_per_speed):
                    if not rclpy.ok(): return
                    
                    if step < warmup_steps:
                        self.cmd_vel = [0.0, 0.0, 0.0, 0.0]
                    else:
                        self.cmd_vel = [speed, 0.0, 0.0, 0.0]
                        
                    raw_data = self._get_raw_sensor_data()
                    self.current_targets = self.pipeline.step(
                        raw_state_kwargs=raw_data,
                        cmd_vel=self.cmd_vel,
                        sim_time=self.data.time
                    )
                    
                    torques = self._pd_torques(self.current_targets)
                    for i, act_idx in enumerate(self.isaac_ctrl_idx):
                        self.data.ctrl[act_idx] = torques[i]

                    mujoco.mj_step(self.model, self.data)
                    
                    if not self.headless:
                        viewer.sync()
                        
                    if step >= warmup_steps:
                        eval_steps += 1
                        contact = raw_data['contact']
                        
                        for foot_idx in range(4):
                            is_contact = (contact[foot_idx] > 0)
                            
                            if is_contact:
                                if current_swing_time[foot_idx] > 0:
                                    swing_times[foot_idx].append(current_swing_time[foot_idx])
                                    current_swing_time[foot_idx] = 0.0
                                
                                force = np.zeros(6, dtype=np.float64)
                                foot_force = 0.0
                                for c_i in range(self.data.ncon):
                                    con = self.data.contact[c_i]
                                    g1, g2 = con.geom1, con.geom2
                                    if g1 == 0 or g2 == 0:
                                        name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g1)
                                        name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g2)
                                        fn = ["FL", "FR", "RL", "RR"][foot_idx]
                                        is_match1 = name1 and name1.lower() != "floor" and fn.lower() in name1.lower()
                                        is_match2 = name2 and name2.lower() != "floor" and fn.lower() in name2.lower()
                                        if is_match1 or is_match2:
                                            mujoco.mj_contactForce(self.model, self.data, c_i, force)
                                            foot_force += abs(force[0])
                                stance_force_sum[foot_idx] += foot_force
                                stance_force_count[foot_idx] += 1
                                max_grf_stance_N[foot_idx] = max(max_grf_stance_N[foot_idx], foot_force)
                                
                            else:
                                current_swing_time[foot_idx] += 0.001
                                if len(self.foot_geom_ids) == 4:
                                    z_height = self.data.geom_xpos[self.foot_geom_ids[foot_idx]][2]
                                    # Normalize relative to ground (floor is at 0.0). We also need to subtract nominal foot radius if we want exact clearance, but relative height is fine.
                                    max_foot_heights[foot_idx] = max(max_foot_heights[foot_idx], z_height)

                        last_contact = contact
                        
                        pair1_sync = (contact[0] == contact[3])
                        pair2_sync = (contact[1] == contact[2])
                        if pair1_sync and pair2_sync:
                            sync_match_count += 1
                            
                        base_z_vels.append(raw_data['vel'][2])
                        base_roll_vels.append(raw_data['gyro'][0])
                        base_pitch_vels.append(raw_data['gyro'][1])
                        actual_fwd_vels.append(raw_data['vel'][0])

                avg_fwd = np.mean(actual_fwd_vels)
                # Max foot height resets every speed, so the accumulated max is what we report.
                # However, since they stand during warmup, maybe max is just the highest it got.
                # Let's subtract ~0.02 (foot radius) to give true lift height above floor
                avg_foot_height_max = [max(0.0, h - 0.02) for h in max_foot_heights]
                
                valid_swing_times = [
                    [t for t in leg_times if t > 0.05] for leg_times in swing_times
                ]
                all_valid_swings = [t for leg_times in valid_swing_times for t in leg_times]
                avg_swing_time = np.mean(all_valid_swings) if all_valid_swings else 0.0
                
                eval_duration_s = max(1, eval_steps) * 0.001
                avg_step_freq = np.mean([len(swings) for swings in valid_swing_times]) / eval_duration_s
                
                avg_grf = [
                    stance_force_sum[i] / max(1, stance_force_count[i]) for i in range(4)
                ]
                
                sync_percentage = (sync_match_count / max(1, eval_steps)) * 100.0
                
                std_z = float(np.std(base_z_vels))
                std_roll = float(np.std(base_roll_vels))
                std_pitch = float(np.std(base_pitch_vels))
                
                results[str(speed)] = {
                    "commanded_speed": round(float(speed), 2),
                    "actual_speed": round(float(avg_fwd), 2),
                    "foot_lift_height_cm": [round(float(x) * 100.0, 2) for x in avg_foot_height_max],
                    "foot_swing_time_s": round(float(avg_swing_time), 2),
                    "step_frequency_hz": round(float(avg_step_freq), 2),
                    "grf_stance_N": [round(float(x), 2) for x in avg_grf],
                    "grf_peak_stance_N": [round(float(x), 2) for x in max_grf_stance_N],
                    "trot_sync_percent": round(float(sync_percentage), 2),
                    "base_oscillation": {
                        "std_z_vel": round(std_z, 2),
                        "std_roll_vel": round(std_roll, 2),
                        "std_pitch_vel": round(std_pitch, 2)
                    }
                }

                print(f"   => Actual Fwd Vel: {avg_fwd:.3f} m/s")
                print(f"   => Foot Lift Height (FL,FR,RL,RR): {[round(v, 4) for v in avg_foot_height_max]} m")
                print(f"   => Average Swing Time: {avg_swing_time:.3f} s (Freq: {avg_step_freq:.2f} Hz)")
                print(f"   => Avg Stance GRF (FL,FR,RL,RR): {[round(v, 1) for v in avg_grf]} N")
                print(f"   => Peak Stance GRF (FL,FR,RL,RR): {[round(v, 1) for v in max_grf_stance_N]} N")
                print(f"   => Trot Synchronization: {sync_percentage:.1f}%")
                print(f"   => Base Oscillation (Z_vel, Roll, Pitch): {std_z:.3f} m/s, {std_roll:.3f} rad/s, {std_pitch:.3f} rad/s")

            log_dir = "logs/skrl/eval_reports"
            os.makedirs(log_dir, exist_ok=True)
            report_path = os.path.join(log_dir, "mujoco_eval_report.json")
            with open(report_path, "w") as f:
                json.dump(results, f, indent=4)
                
            print("\n" + "="*60)
            print(f"✅ MUJOCO EVALUATION COMPLETE. Report saved to: {report_path}")
            print("="*60 + "\n")
            
            # Kill the node loop
            rclpy.shutdown()
            os._exit(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="go2")
    parser.add_argument("--internal_policy", type=str, default=None)
    parser.add_argument("--obs_dim", type=int, default=49)
    parser.add_argument("--use_estimator", action="store_true")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    rclpy.init()
    node = MujocoEvaluator(
        robot_type=args.robot, 
        checkpoint=args.internal_policy, 
        obs_dim=args.obs_dim,
        use_estimator=args.use_estimator,
        headless=args.headless
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == "__main__":
    main()
