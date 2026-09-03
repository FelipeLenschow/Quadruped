import os
import sys
import time
import threading
import numpy as np
import argparse
import re

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray

import mujoco
import mujoco.viewer

# Ensure absolute path of the repository is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Telemetry.estimator import rot_from_quat
from Mujoco.foot_contact_overlay import FootContactOverlay, FOOT_NAMES
from Mujoco.velocity_arrow_overlay import VelocityArrowOverlay
from geometry_msgs.msg import Twist

class MujocoTwinNode(Node):
    """
    Passive MuJoCo Digital Twin Node.
    Listens to ROS 2 topics and updates the MuJoCo visualization.
    """
    def __init__(self, robot_type="go2", use_estimator=False, show_ghost=True):
        super().__init__("mujoco_twin_node")
        self.robot_type = robot_type
        self.show_ghost = show_ghost

        # 1. Load MuJoCo Model
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Mujoco"))
        menagerie_dir = os.path.join(base_dir, "mujoco_menagerie", "unitree_go2")
        mjcf_path = os.path.join(menagerie_dir, "scene.xml")
        if not os.path.exists(mjcf_path):
            mjcf_path = os.path.join(base_dir, "scene.xml")
            menagerie_dir = base_dir

        # The ghost is a second, transparent green robot showing the COMMANDED
        # joint positions next to the measured ones. Optional: it doubles the
        # geometry and is only useful when comparing command vs response.
        if self.show_ghost:
            # --- Generate Ghost Robot ---
            # 1. Generate go2_ghost.xml
            with open(os.path.join(menagerie_dir, "go2.xml"), "r") as f:
                go2_xml = f.read()
        
            go2_xml = go2_xml.replace('model="go2"', 'model="go2_ghost"')
            go2_xml = go2_xml.replace('name="', 'name="cmd_')
            go2_xml = go2_xml.replace('joint="', 'joint="cmd_')
            go2_xml = go2_xml.replace('child="', 'child="cmd_')
            go2_xml = go2_xml.replace('class="', 'class="cmd_')
            # Replace material definitions with transparent rgba (Greenish)
            go2_xml = re.sub(r'material="[^"]+"', 'rgba="0.2 0.8 0.2 0.4"', go2_xml)
            # Remove asset block to prevent duplicate mesh loads
            go2_xml = re.sub(r'<asset>.*?</asset>', '', go2_xml, flags=re.DOTALL)
        
            ghost_path = os.path.join(menagerie_dir, "twin_go2_ghost.xml")
            with open(ghost_path, "w") as f:
                f.write(go2_xml)
            
            # 2. Generate scene_twin.xml
            with open(mjcf_path, "r") as f:
                scene_xml = f.read()
            
            scene_xml = scene_xml.replace(
                '<include file="go2.xml"/>',
                '<include file="go2.xml"/>\n  <include file="twin_go2_ghost.xml"/>'
            )
        
            scene_twin_path = os.path.join(menagerie_dir, "twin_scene.xml")
            with open(scene_twin_path, "w") as f:
                f.write(scene_xml)
            
            mjcf_path = scene_twin_path
            # --- End Generate Ghost Robot ---

        self.get_logger().info(
            f"Loading MuJoCo Twin Model{' with Ghost' if self.show_ghost else ''} from {mjcf_path}")
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
        self.cmd_qpos_addr = np.full(12, -1, dtype=int)
        for i, name in enumerate(self.isaac_names):
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j_id != -1:
                self.qpos_addr[i] = self.model.jnt_qposadr[j_id]
                
            cmd_j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cmd_" + name)
            if cmd_j_id != -1:
                self.cmd_qpos_addr[i] = self.model.jnt_qposadr[cmd_j_id]

        # Resolve Base Addresses
        real_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if real_body_id != -1:
            self.real_base_addr = self.model.jnt_qposadr[self.model.body_jntadr[real_body_id]]
        else:
            self.real_base_addr = 0
            
        cmd_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cmd_base")
        if cmd_body_id != -1:
            self.cmd_base_addr = self.model.jnt_qposadr[self.model.body_jntadr[cmd_body_id]]
        else:
            self.cmd_base_addr = -1

        # Foot FSR overlay: same bars, same threshold tick, same scale as the
        # MuJoCo driver's viewer - here fed from /sensors/foot_force instead of
        # from local physics, so it works against the simulator and the real
        # robot alike (both publish raw FSR counts on that topic).
        self.contact_overlay = FootContactOverlay(self.model)
        if not self.contact_overlay.available:
            self.get_logger().warn(
                "No foot sites in the twin model; contact bars disabled.")

        # 2. State Variables
        self.base_pos = np.array([0.0, 0.0, 0.50], dtype=np.float64)
        self.base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.joint_pos = np.zeros(12, dtype=np.float64)
        self.cmd_joint_pos = np.zeros(12, dtype=np.float64)
        self.base_lin_vel_body = np.zeros(3, dtype=np.float64)
        self.cmd_vel = np.zeros(4, dtype=np.float64)
        self.foot_force_raw = np.zeros(len(FOOT_NAMES), dtype=np.float64)
        self.foot_force_valid = False

        self.last_time = time.time()
        self.lock = threading.Lock()
        self.running = True

        # 3. ROS Subscriptions
        self.create_subscription(JointState, "/sensors/joint_states", self.joint_cb, 10)
        self.create_subscription(JointState, "/commands/joint_commands", self.cmd_joint_cb, 10)
        self.create_subscription(Imu, "/sensors/imu", self.imu_cb, 10)
        self.create_subscription(
            Float32MultiArray, "/sensors/foot_force", self.foot_force_cb, 10)
        self.create_subscription(Twist, "/cmd_vel", self.cmd_vel_cb, 10)
        
        self.vel_overlay = VelocityArrowOverlay(self.model)
        
        # Until the first FSR message the feet are bare, which looks identical
        # to a broken overlay. Say so, once every 5 s, until data shows up.
        self.fsr_status_timer = self.create_timer(5.0, self.fsr_status_cb)
        
        self.get_logger().info("MuJoCo Twin subscribing to both /odom/state_estimator and /odom/state_simulator")
        self.create_subscription(Odometry, "/odom/state_estimator", self.odom_cb, 10)
        self.create_subscription(Odometry, "/odom/state_simulator", self.odom_cb, 10)

        # 4. Start Viewer Thread
        self.viewer_thread = threading.Thread(target=self._viewer_loop, daemon=True)
        self.viewer_thread.start()

    def joint_cb(self, msg: JointState):
        with self.lock:
            for i, name in enumerate(msg.name):
                try:
                    idx = self.isaac_names.index(name)
                    self.joint_pos[idx] = msg.position[i]
                except ValueError:
                    pass

    def cmd_joint_cb(self, msg: JointState):
        with self.lock:
            for i, name in enumerate(msg.name):
                try:
                    idx = self.isaac_names.index(name)
                    self.cmd_joint_pos[idx] = msg.position[i]
                except ValueError:
                    pass

    def foot_force_cb(self, msg: Float32MultiArray):
        n = len(self.foot_force_raw)
        if len(msg.data) < n:
            return
        with self.lock:
            self.foot_force_raw[:] = msg.data[:n]
            first = not self.foot_force_valid
            self.foot_force_valid = True
        if first:
            self.get_logger().info(
                "Foot FSR contact bars live: first /sensors/foot_force message "
                f"{np.round(self.foot_force_raw, 0)} (threshold "
                f"{self.contact_overlay.contact_threshold:.0f} raw).")

    def cmd_vel_cb(self, msg: Twist):
        with self.lock:
            self.cmd_vel[0] = msg.linear.x
            self.cmd_vel[1] = msg.linear.y
            self.cmd_vel[2] = msg.angular.z

    def fsr_status_cb(self):
        if self.foot_force_valid:
            self.fsr_status_timer.cancel()
            return
        self.get_logger().warn(
            "No /sensors/foot_force yet - foot contact bars are not being drawn. "
            "Is a driver or an MCAP replay publishing that topic?")

    def imu_cb(self, msg: Imu):
        with self.lock:
            q = msg.orientation
            self.base_quat = np.array([q.w, q.x, q.y, q.z])

    def odom_cb(self, msg: Odometry):
        with self.lock:
            p = msg.pose.pose.position
            self.base_pos = np.array([p.x, p.y, p.z])
            v = msg.twist.twist.linear
            self.base_lin_vel_body = np.array([v.x, v.y, v.z])

    def _viewer_loop(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            track_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
            if track_id == -1:
                track_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
            if track_id != -1:
                viewer.cam.trackbodyid = track_id

            while self.running and rclpy.ok() and viewer.is_running():
                current_time = time.time()
                dt = current_time - self.last_time
                self.last_time = current_time

                with self.lock:
                    # Update MuJoCo Data
                    if self.real_base_addr >= 0:
                        self.data.qpos[self.real_base_addr:self.real_base_addr+3] = self.base_pos
                        self.data.qpos[self.real_base_addr+3:self.real_base_addr+7] = self.base_quat
                    else:
                        self.data.qpos[0:3] = self.base_pos
                        self.data.qpos[3:7] = self.base_quat

                    if self.cmd_base_addr >= 0:
                        self.data.qpos[self.cmd_base_addr:self.cmd_base_addr+3] = self.base_pos
                        self.data.qpos[self.cmd_base_addr+3:self.cmd_base_addr+7] = self.base_quat

                    for i, addr in enumerate(self.qpos_addr):
                        if addr >= 0:
                            self.data.qpos[addr] = self.joint_pos[i]
                            
                    for i, addr in enumerate(self.cmd_qpos_addr):
                        if addr >= 0:
                            self.data.qpos[addr] = self.cmd_joint_pos[i]

                    foot_force = self.foot_force_raw.copy()
                    have_force = self.foot_force_valid

                mujoco.mj_forward(self.model, self.data)

                if viewer.user_scn is not None:
                    with viewer.lock():
                        contact_drawn = False
                        if have_force and self.contact_overlay.available:
                            self.contact_overlay.draw(
                                viewer.user_scn, self.data, foot_force)
                            contact_drawn = True

                        if self.vel_overlay.available:
                            # contact_overlay clears the scene, so if it drew, we append.
                            # Otherwise we must clear the scene ourselves.
                            self.vel_overlay.draw(
                                viewer.user_scn, self.data,
                                self.cmd_vel, self.base_lin_vel_body, reset=not contact_drawn)

                viewer.sync()
                time.sleep(1.0 / 60.0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="go2")
    parser.add_argument("--use_estimator", action="store_true", help="Use estimated odometry instead of ground truth")
    parser.add_argument("--no_ghost", action="store_true",
                        help="Hide the green ghost robot showing commanded joint positions")
    args = parser.parse_args()

    rclpy.init()
    node = MujocoTwinNode(robot_type=args.robot, use_estimator=args.use_estimator,
                          show_ghost=not args.no_ghost)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.running = False
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
