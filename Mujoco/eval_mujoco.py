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
from Controller.robot_defaults import DEFAULT_STANCE_QPOS

class MujocoEvaluator(Node):
    def __init__(self, robot_type="go2", checkpoint=None, obs_dim=49, use_estimator=False, headless=True):
        super().__init__("mujoco_evaluator_node")
        self.robot_type = robot_type
        self.checkpoint = checkpoint
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

        self.desired_qpos = DEFAULT_STANCE_QPOS.copy()

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
        # The policy carries observation history and last_actions across mj_resetData, so without
        # this every velocity test starts with the previous test's state still inside the network.
        self.pipeline.reset()
        self.current_targets = self.desired_qpos.copy()
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
        axes_tests = {
            "x": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.00],
            "y": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50],
            "yaw": [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00]
        }
        steps_per_speed = 32000  # 32 seconds at 1000Hz (30s walking + 2s warmup)
        warmup_steps = 2000     # 2 seconds standing still
        
        results = {"x": {}, "y": {}, "yaw": {}}
        
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

            for axis, speeds in axes_tests.items():
                for speed in speeds:
                    print(f"\n[EVAL] Testing {axis} velocity: {speed}")
                    self._reset_robot()
                    
                    max_foot_heights = [0.0, 0.0, 0.0, 0.0]
                    swing_times = [[], [], [], []]
                    stance_force_sum = [0.0, 0.0, 0.0, 0.0]
                    stance_force_count = [0, 0, 0, 0]
                    max_grf_stance_N = [0.0, 0.0, 0.0, 0.0]
                    
                    base_z_vels = []
                    base_roll_vels = []
                    base_pitch_vels = []
                    actual_vels = []
                    
                    eval_steps = 0
                    
                    err_x_sum = 0.0
                    err_y_sum = 0.0
                    err_yaw_sum = 0.0
                    
                    last_contact = [0.0, 0.0, 0.0, 0.0]
                    current_swing_time = [0.0, 0.0, 0.0, 0.0]

                    # Touchdown impact speed, per foot. prev_foot_z/last_foot_vz finite-difference
                    # the foot's height while it is airborne so the landing can be charged against
                    # the last PRE-impact sample -- see the append below.
                    landing_vels = [[], [], [], []]
                    prev_foot_z = [None, None, None, None]
                    last_foot_vz = [0.0, 0.0, 0.0, 0.0]
                    
                    last_strike_time = [0.0, 0.0, 0.0, 0.0]
                    stride_duration = [0.0, 0.0, 0.0, 0.0]
                    phase_diff_front_list = []
                    phase_diff_right_list = []
                    phase_diff_diag_list = []
                    
                    for step in range(steps_per_speed):
                        if not rclpy.ok(): return
                        
                        if step < warmup_steps:
                            self.cmd_vel = [0.0, 0.0, 0.0, 0.0]
                        else:
                            if axis == "x":
                                self.cmd_vel = [speed, 0.0, 0.0, 0.0]
                            elif axis == "y":
                                self.cmd_vel = [0.0, speed, 0.0, 0.0]
                            elif axis == "yaw":
                                self.cmd_vel = [0.0, 0.0, speed, 0.0]
                            
                        raw_data = self._get_raw_sensor_data()
                        self.current_targets = self.pipeline.step(
                            raw_state_kwargs=raw_data,
                            cmd_vel=self.cmd_vel,
                            sim_time=self.data.time
                        )
                        
                        torques = self._pd_torques(self.current_targets)
                        if not np.all(np.isfinite(torques)):
                            # Feeding this to MuJoCo corrupts the model state for every test that
                            # follows, so stop this one here and say so instead.
                            print(
                                f"   !! non-finite torque at t={self.data.time:.2f}s "
                                f"({axis}={speed}); aborting this test"
                            )
                            break
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
                                        # Vertical foot speed on the last airborne sample, i.e.
                                        # BEFORE the collision is resolved. Reading it once contact
                                        # is detected reports the post-impact velocity (~0) for
                                        # exactly the hard landings this is meant to catch. Gated on
                                        # the same 0.05 s swing filter as valid_swing_times so
                                        # contact chatter does not register as a landing.
                                        if current_swing_time[foot_idx] > 0.05:
                                            landing_vels[foot_idx].append(abs(last_foot_vz[foot_idx]))
                                        current_swing_time[foot_idx] = 0.0
                                        
                                        t = self.data.time
                                        if last_strike_time[foot_idx] > 0:
                                            stride_duration[foot_idx] = t - last_strike_time[foot_idx]
                                        last_strike_time[foot_idx] = t
                                        
                                        if foot_idx == 1 and stride_duration[0] > 0.1: # FR vs FL
                                            p = ((t - last_strike_time[0]) % stride_duration[0]) / stride_duration[0]
                                            phase_diff_front_list.append(p if p <= 0.5 else 1.0 - p)
                                        if foot_idx == 3 and stride_duration[1] > 0.1: # RR vs FR
                                            p = ((t - last_strike_time[1]) % stride_duration[1]) / stride_duration[1]
                                            phase_diff_right_list.append(p if p <= 0.5 else 1.0 - p)
                                        if foot_idx == 3 and stride_duration[0] > 0.1: # RR vs FL
                                            p = ((t - last_strike_time[0]) % stride_duration[0]) / stride_duration[0]
                                            phase_diff_diag_list.append(p if p <= 0.5 else 1.0 - p)
                                    
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
                                    # Drop the airborne height history so the next swing starts
                                    # its finite difference fresh instead of across the stance.
                                    prev_foot_z[foot_idx] = None
                                    stance_force_sum[foot_idx] += foot_force
                                    stance_force_count[foot_idx] += 1
                                    max_grf_stance_N[foot_idx] = max(max_grf_stance_N[foot_idx], foot_force)
                                    
                                else:
                                    current_swing_time[foot_idx] += 0.001
                                    if len(self.foot_geom_ids) == 4:
                                        z_height = self.data.geom_xpos[self.foot_geom_ids[foot_idx]][2]
                                        # Normalize relative to ground (floor is at 0.0). We also need to subtract nominal foot radius if we want exact clearance, but relative height is fine.
                                        max_foot_heights[foot_idx] = max(max_foot_heights[foot_idx], z_height)
                                        if prev_foot_z[foot_idx] is not None:
                                            last_foot_vz[foot_idx] = (
                                                z_height - prev_foot_z[foot_idx]
                                            ) / 0.001
                                        prev_foot_z[foot_idx] = z_height
    
                            last_contact = contact
                                
                            base_z_vels.append(raw_data['vel'][2])
                            base_roll_vels.append(raw_data['gyro'][0])
                            base_pitch_vels.append(raw_data['gyro'][1])
                            
                            if axis == "x":
                                actual_vels.append(raw_data['vel'][0])
                                err_x_sum += (raw_data['vel'][0] - speed) * 0.001
                                err_y_sum += (raw_data['vel'][1] - 0.0) * 0.001
                                err_yaw_sum += (raw_data['gyro'][2] - 0.0) * 0.001
                            elif axis == "y":
                                actual_vels.append(raw_data['vel'][1])
                                err_x_sum += (raw_data['vel'][0] - 0.0) * 0.001
                                err_y_sum += (raw_data['vel'][1] - speed) * 0.001
                                err_yaw_sum += (raw_data['gyro'][2] - 0.0) * 0.001
                            elif axis == "yaw":
                                actual_vels.append(raw_data['gyro'][2])
                                err_x_sum += (raw_data['vel'][0] - 0.0) * 0.001
                                err_y_sum += (raw_data['vel'][1] - 0.0) * 0.001
                                err_yaw_sum += (raw_data['gyro'][2] - speed) * 0.001

                    avg_actual_vel = np.mean(actual_vels) if actual_vels else 0.0
                    avg_foot_height_max = [max(0.0, h - 0.02) for h in max_foot_heights]
                    
                    valid_swing_times = [
                        [t for t in leg_times if t > 0.05] for leg_times in swing_times
                    ]
                    all_valid_swings = [t for leg_times in valid_swing_times for t in leg_times]
                    avg_swing_time = np.mean(all_valid_swings) if all_valid_swings else 0.0
                    
                    eval_duration_s = max(1, eval_steps) * 0.001
                    step_freqs = [(len(swings) / eval_duration_s) for swings in valid_swing_times]
                    avg_step_freq = np.mean(step_freqs) if step_freqs else 0.0
                    
                    avg_grf = [
                        stance_force_sum[i] / max(1, stance_force_count[i]) for i in range(4)
                    ]

                    avg_landing_vel = [
                        float(np.mean(v)) if v else 0.0 for v in landing_vels
                    ]
                    
                    std_z = float(np.std(base_z_vels)) if base_z_vels else 0.0
                    std_roll = float(np.std(base_roll_vels)) if base_roll_vels else 0.0
                    std_pitch = float(np.std(base_pitch_vels)) if base_pitch_vels else 0.0
                    
                    avg_phase_front = np.mean(phase_diff_front_list) * 100.0 if phase_diff_front_list else 0.0
                    avg_phase_right = np.mean(phase_diff_right_list) * 100.0 if phase_diff_right_list else 0.0
                    avg_phase_diag = np.mean(phase_diff_diag_list) * 100.0 if phase_diff_diag_list else 0.0
                    
                    results[axis][str(speed)] = {
                        "commanded_speed": round(float(speed), 2),
                        "actual_speed": round(float(avg_actual_vel), 2),
                        "foot_lift_height_cm": [round(float(x) * 100.0, 2) for x in avg_foot_height_max],
                        "foot_swing_time_s": round(float(avg_swing_time), 2),
                        "step_frequency_hz": [round(float(f), 2) for f in step_freqs],
                        "grf_stance_N": [round(float(x), 2) for x in avg_grf],
                        "grf_peak_stance_N": [round(float(x), 2) for x in max_grf_stance_N],
                        "foot_landing_vel_ms": [round(v, 2) for v in avg_landing_vel],
                        "phase_diff_front_percent": round(float(avg_phase_front), 2),
                        "phase_diff_right_percent": round(float(avg_phase_right), 2),
                        "phase_diff_diag_percent": round(float(avg_phase_diag), 2),
                        "base_oscillation": {
                            "std_z_vel": round(std_z, 2),
                            "std_roll_vel": round(std_roll, 2),
                            "std_pitch_vel": round(std_pitch, 2)
                        },
                        "position_error": {
                            "x_m": round(err_x_sum, 4),
                            "y_m": round(err_y_sum, 4),
                            "yaw_rad": round(err_yaw_sum, 4)
                        }
                    }
    
                    print(f"   => Actual Vel: {avg_actual_vel:.3f}")
                    print(f"   => Foot Lift Height (FL,FR,RL,RR): {[round(v, 4) for v in avg_foot_height_max]} m")
                    print(f"   => Average Swing Time: {avg_swing_time:.3f} s (Freq: {avg_step_freq:.2f} Hz)")
                    print(f"   => Peak Stance GRF (FL,FR,RL,RR): {[round(v, 1) for v in max_grf_stance_N]} N")
                    print(f"   => Landing Vel (FL,FR,RL,RR): {[round(v, 2) for v in avg_landing_vel]} m/s")
                    print(f"   => Phases (Front, Right, Diag): {avg_phase_front:.1f}%, {avg_phase_right:.1f}%, {avg_phase_diag:.1f}%")

            if self.checkpoint:
                log_dir = os.path.dirname(self.checkpoint)
                basename = os.path.basename(self.checkpoint).replace(".pt", "")
                report_filename = f"mujoco_eval_report_{basename}.json"
            else:
                log_dir = "logs/skrl/eval_reports"
                report_filename = "mujoco_eval_report.json"
                
            os.makedirs(log_dir, exist_ok=True)
            report_path = os.path.join(log_dir, report_filename)
            
            final_report = {
                "metadata": {
                    "checkpoint": os.path.abspath(self.checkpoint) if self.checkpoint else "None",
                    "checkpoint_name": os.path.basename(self.checkpoint) if self.checkpoint else "None",
                    "robot_type": self.robot_type,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                },
                "results": results
            }
            
            with open(report_path, "w") as f:
                json.dump(final_report, f, indent=4)
                
            try:
                os.chmod(report_path, 0o666)
            except Exception as e:
                pass
                
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
