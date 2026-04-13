import os
import sys
import time
import numpy as np
import torch
import argparse
import threading
import mujoco

# ROS 2 Standard Imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Vector3, Pose, Twist

# Project Imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Mujoco.unitree_sdk_mock import quat_to_rot_matrix

class Ros2MujocoBridge(Node):
    def __init__(self, robot_type, scene_path):
        super().__init__('mujoco_bridge_node')
        self.robot_type = robot_type
        
        # 1. Load MuJoCo Model
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.00125
        
        # 2. Logic Mapping (Mujoco -> Isaac order)
        from Mujoco.mujoco_sim2sim import ISAAC_JOINT_NAMES
        mj_names = [self.model.joint(i).name for i in range(self.model.njnt) if self.model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
        
        def _norm(s): return s.replace("_joint", "").lower()
        mj_norm = [_norm(n) for n in mj_names]
        self.mj_to_isaac = np.zeros(12, dtype=np.int32)
        for i, name in enumerate(ISAAC_JOINT_NAMES):
            norm = _norm(name)
            try: self.mj_to_isaac[i] = mj_norm.index(norm)
            except: self.mj_to_isaac[i] = i
        
        self.isaac_to_mj = np.zeros(12, dtype=np.int32)
        for i, val in enumerate(self.mj_to_isaac): self.isaac_to_mj[i] = val
        self.mj_names = mj_names

        # 3. Control Buffers
        self.latest_command = np.zeros(12, dtype=np.float32)
        self.has_command = False
        
        # 4. ROS 2 Publishers & Subscriptions
        self.joint_pub = self.create_publisher(JointState, '/sensors/joint_states', 10)
        self.imu_pub = self.create_publisher(Imu, '/sensors/imu', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.create_subscription(JointState, '/commands/joint_commands', self.command_callback, 10)
        
        # 5. Physics Thread
        self.physics_thread = threading.Thread(target=self._physics_loop, daemon=True)
        self.physics_thread.start()
        
        print(f"[MujocoBridge] Initialized for {robot_type}. Physics running at 200Hz.")

    def reset_robot(self):
        """Resets robot to standing pose and 0.5m height."""
        # 1. Reset MuJoCo Data
        mujoco.mj_resetData(self.model, self.data)
        
        # 2. Set nominal joint positions
        desired_qpos_isaac = np.array([0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5], dtype=np.float32)
        for i, m_idx in enumerate(self.isaac_to_mj):
            # joints start at qpos[7] for a freejoint model
            self.data.qpos[7 + int(m_idx)] = desired_qpos_isaac[i]
            
        # 3. Set Base Pose (z = 0.5m)
        self.data.qpos[2] = 0.50
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0] # [w, x, y, z]
        
        mujoco.mj_forward(self.model, self.data)
        print("[MujocoBridge] Robot reset to standing pose.")

    def command_callback(self, msg):
        self.latest_command[:] = msg.position
        self.has_command = True

    def _init_physics(self):
        self.model.opt.timestep = 0.001  # Force exact physics match with mujoco_sim2sim
        
        # Set damping and friction loss to match mujoco_sim2sim
        for i in range(self.model.nu):
            self.model.actuator_gainprm[i, 0] = 1.0
            self.model.actuator_biasprm[i, 1] = 0.0
            self.model.actuator_ctrllimited[i] = 0
            self.model.actuator_forcerange[i, :2] = [-23.7, 23.7]

        for i in range(self.model.njnt):
            if self.model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
                self.model.dof_damping[self.model.jnt_dofadr[i]] = 0.15
                self.model.dof_frictionloss[self.model.jnt_dofadr[i]] = 0.05

        # Initialize step counter
        self.step_counter = 0

        # Load Actuator Net (Standard for Unitree in MuJoCo)
        self.act_net = None
        for path in [
            os.path.join(os.path.dirname(__file__), f"unitree_{self.robot_type}.pt"),
            os.path.join(os.path.dirname(__file__), "unitree_quadruped.pt"),
            os.path.join(os.path.dirname(__file__), "..", "Deployment", "unitree_quadruped.pt")
        ]:
            if os.path.exists(path):
                print(f"[MujocoBridge] Loading ActuatorNet: {path}")
                self.act_net = torch.jit.load(path, map_location="cpu").eval()
                break
        
        # History buffers for ActuatorNet (3 steps of pos/vel)
        self.pos_err_hist = np.zeros((3, 12), dtype=np.float32)
        self.vel_hist = np.zeros((3, 12), dtype=np.float32)

    def _physics_loop(self):
        self._init_physics()
        self.reset_robot()
        DECIMATION = 5 # 5 * 0.001s = 0.005s
        kp = 35.0 # Fallback
        kd = 1.0
        
        # We run the PD/ActuatorNet loop at 200Hz
        try:
            import mujoco.viewer
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                track_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
                if track_id == -1: track_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
                viewer.cam.trackbodyid = track_id
                
                next_time = time.time()
                while rclpy.ok() and viewer.is_running():
                    targets = self.latest_command if self.has_command else np.array([0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5], dtype=np.float32)
                    
                    # 1. Update Current State (Isaac Order)
                    q_all = np.array([self.data.qpos[self.model.joint(n).qposadr[0]] for n in self.mj_names])
                    v_all = np.array([self.data.qvel[self.model.joint(n).dofadr[0]] for n in self.mj_names])
                    q_isaac = q_all[self.mj_to_isaac]
                    v_isaac = v_all[self.mj_to_isaac]
                    
                    # 2. Update History
                    self.pos_err_hist = np.roll(self.pos_err_hist, 1, 0)
                    self.pos_err_hist[0] = q_isaac - targets
                    self.vel_hist = np.roll(self.vel_hist, 1, 0)
                    self.vel_hist[0] = v_isaac
                    
                    # 3. Compute Torque
                    if self.act_net:
                        net_in = torch.zeros((12, 6))
                        net_in[:, :3] = torch.from_numpy(self.pos_err_hist.T)
                        net_in[:, 3:] = torch.from_numpy(self.vel_hist.T)
                        with torch.no_grad():
                            torques = self.act_net(net_in).squeeze().numpy()
                    else:
                        torques = -kp * self.pos_err_hist[0] - kd * self.vel_hist[0]
                    
                    # 4. Apply (MuJoCo Order)
                    for i in range(12):
                        mj_idx = self.isaac_to_mj[i]
                        self.data.ctrl[mj_idx] = np.clip(torques[i], -23.7, 23.7)
                        
                    for _ in range(DECIMATION):
                        mujoco.mj_step(self.model, self.data)
                    
                    self.step_counter += 1
                    
                    # Publish sensors and sync viewer at 50Hz
                    if self.step_counter % 4 == 0:
                        self.publish_sensors()
                        viewer.sync()
                    
                    next_time += 0.005
                    sleep_dur = next_time - time.time()
                    if sleep_dur > 0:
                        time.sleep(sleep_dur)
                    else:
                        next_time = time.time()
        except Exception as e:
            print(f"[MujocoBridge] Error in physics loop: {e}")
            # ... headless fallback could be updated too but we prioritize the viewer for the user

    def publish_sensors(self):
        now = self.get_clock().now().to_msg()
        # Joint States
        js = JointState()
        js.header.stamp = now
        js.name = ["FL_HAA", "FL_HFE", "FL_KFE", "FR_HAA", "FR_HFE", "FR_KFE", 
                   "RL_HAA", "RL_HFE", "RL_KFE", "RR_HAA", "RR_HFE", "RR_KFE"]
        q_all = np.array([self.data.qpos[self.model.joint(n).qposadr[0]] for n in self.mj_names])
        dq_all = np.array([self.data.qvel[self.model.joint(n).dofadr[0]] for n in self.mj_names])
        js.position = q_all[self.mj_to_isaac].tolist()
        js.velocity = dq_all[self.mj_to_isaac].tolist()
        self.joint_pub.publish(js)

        # IMU
        imu = Imu()
        imu.header.stamp = now
        q = self.data.qpos[3:7] # [w, x, y, z]
        imu.orientation = Quaternion(w=float(q[0]), x=float(q[1]), y=float(q[2]), z=float(q[3]))
        gyro = self.data.qvel[3:6]
        imu.angular_velocity = Vector3(x=float(gyro[0]), y=float(gyro[1]), z=float(gyro[2]))
        self.imu_pub.publish(imu)

        # Odometry (for linear velocity)
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base'
        # Get world-frame linear velocity and rotate to body frame
        lin_vel_w = self.data.qvel[:3]
        from Mujoco.unitree_sdk_mock import quat_to_rot_matrix # Actually the bridge has this logic? 
        # Wait, I'll just use a simple rotation matrix here to avoid import issues
        w, x, y, z = q
        R = np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
        ])
        lin_vel_b = R.T @ lin_vel_w
        odom.twist.twist.linear = Vector3(x=float(lin_vel_b[0]), y=float(lin_vel_b[1]), z=float(lin_vel_b[2]))
        self.odom_pub.publish(odom)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='go1')
    args = parser.parse_args()

    # Determine Scene Path
    from Mujoco.mujoco_sim2sim import ensure_mjcf
    scene_path = str(ensure_mjcf(args.robot))

    rclpy.init()
    node = Ros2MujocoBridge(args.robot, scene_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
