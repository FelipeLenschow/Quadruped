import os
import sys
import yaml
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist, Vector3
import time

# Import Kinematics from same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from kinematics import Kinematics

class RewardEstimatorNode(Node):
    def __init__(self):
        super().__init__('reward_estimator_node')
        
        # Declare parameters
        self.declare_parameter('robot_type', 'go2')
        self.declare_parameter('config_path', '')
        self.declare_parameter('phase', 'phase1')
        
        self.robot_type = self.get_parameter('robot_type').value
        config_path = self.get_parameter('config_path').value
        self.phase = self.get_parameter('phase').value
        
        # Fallback config path if not provided
        if not config_path:
            base_dir = os.path.abspath(os.path.join(current_dir, ".."))
            config_path = os.path.join(
                base_dir, "IsaacLab_Tasks", "Walk", "source", "Quadruped", 
                "Quadruped", "tasks", "direct", "quadruped", "training_phases.yaml"
            )
            
        self.get_logger().info(f"Loading reward config from: {config_path} (Phase: {self.phase})")
        
        if not os.path.exists(config_path):
            self.get_logger().error(f"Config file not found: {config_path}")
            sys.exit(1)
            
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
            
        # Parse phase config (inherits logic simplified for this node)
        self.cfg = full_config.get(self.phase, full_config.get('default', {}))
        if 'rewards' not in self.cfg:
            self.cfg['rewards'] = full_config.get('default', {}).get('rewards', {})
        if 'commands' not in self.cfg:
            self.cfg['commands'] = full_config.get('default', {}).get('commands', {})
            
        self.rew_cfg = self.cfg['rewards']
        self.cmd_cfg = self.cfg['commands']
        
        # Initialize Kinematics
        self.kin = Kinematics(self.robot_type)
        
        # ROS 2 Pub/Sub
        self.sub_cmd     = self.create_subscription(Twist, '/cmd_vel', self.cmd_cb, 10)
        self.sub_joints  = self.create_subscription(JointState, '/sensors/joint_states', self.joint_cb, 10)
        self.sub_imu     = self.create_subscription(Imu, '/sensors/imu', self.imu_cb, 10)
        self.sub_vel     = self.create_subscription(Vector3, '/estimator/base_lin_vel', self.vel_cb, 10)
        self.sub_contact = self.create_subscription(Float32MultiArray, '/estimator/feet_contact', self.contact_cb, 10)
        
        self.pub_rewards = self.create_publisher(JointState, '/policy_rewards', 10)
        self.timer = self.create_timer(0.02, self.timer_cb) # 50Hz evaluation
        
        # State tracking (Latest values)
        self.commands = np.zeros(3) # vx, vy, wz
        self.q = np.zeros(12)
        self.dq = np.zeros(12)
        self.gyro = np.zeros(3)
        self.vel = np.zeros(3)
        self.contact = np.zeros(4)
        
        self.last_joint_vel = np.zeros(12)
        self.last_base_vel = np.zeros(3)
        self.last_time = time.time()
        
        self.feet_air_time = np.zeros(4)
        # Peak height of the current swing, per foot -- the foot-height terms are scored on this
        # at touchdown, mirroring feet_height_max in quadruped_env.py.
        self.feet_height_max = np.zeros(4)
        self.last_contact = np.zeros(4)
        
        # Reward names in order
        self.reward_names = [
            "track_lin_vel_xy",
            "track_ang_vel_z",
            "feet_air_time",
            "foot_height",
            "flat_orientation",
            "lin_vel_z",
            "ang_vel_xy",
            "dof_pos",
            "dof_torques",
            "dof_acc",
            "base_acc",
            "max_air_feet",
            "feet_air_penalty",
            "feet_air_penalty_static",
            "joint_vel_l2_static",
            "feet_grounded",
            "total_reward"
        ]

    def cmd_cb(self, msg: Twist):
        self.commands[0] = msg.linear.x
        self.commands[1] = msg.linear.y
        self.commands[2] = msg.angular.z

    def joint_cb(self, msg: JointState):
        if len(msg.position) >= 12:
            self.q = np.array(msg.position[:12])
            self.dq = np.array(msg.velocity[:12])

    def imu_cb(self, msg: Imu):
        self.gyro = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])

    def vel_cb(self, msg: Vector3):
        self.vel = np.array([msg.x, msg.y, msg.z])

    def contact_cb(self, msg: Float32MultiArray):
        if len(msg.data) >= 4:
            self.contact = np.array(msg.data[:4])

    def timer_cb(self):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0:
            dt = 0.02
        self.last_time = current_time
        
        # Compute derivatives
        dof_acc = (self.dq - self.last_joint_vel) / dt
        self.last_joint_vel = self.dq.copy()
        
        base_acc = (self.vel - self.last_base_vel) / dt
        self.last_base_vel = self.vel.copy()
        
        # Compute FK for foot heights
        foot_positions = self.kin.get_foot_positions(self.q)
        # Z-position relative to base + standing height approx
        foot_heights_z = foot_positions[:, 2] + 0.30
        
        # Update Air Time and calculate impulse reward
        target_air_time = self.rew_cfg.get('target_feet_air_time', 0.2)
        air_time_reward = 0.0
        
        # Foot height, matching the env's two forms: the reward is DENSE (per airborne foot per
        # step, instantaneous height), the penalty is charged once per landing on the swing apex.
        target_foot_height = self.rew_cfg.get('target_foot_height', 0.12)
        foot_height_sigma = self.rew_cfg.get('foot_height_sigma', 0.01)
        foot_height_match = 0.0
        foot_height_mismatch = 0.0

        for i in range(4):
            is_contact = self.contact[i] > 0.5
            was_contact = self.last_contact[i] > 0.5

            self.feet_height_max[i] = max(self.feet_height_max[i], foot_heights_z[i])

            # Dense lift reward: every step this foot is airborne, on its current height.
            if not is_contact:
                foot_height_match += np.exp(
                    -((foot_heights_z[i] - target_foot_height) ** 2) / foot_height_sigma
                )

            # First contact this step -- the apex penalty is charged here, once per swing.
            if is_contact and not was_contact:
                air_time_reward += max(0.0, self.feet_air_time[i] - target_air_time)
                err_sq = (self.feet_height_max[i] - target_foot_height) ** 2
                foot_height_mismatch += 1.0 - np.exp(-err_sq / foot_height_sigma)

            if not is_contact:
                self.feet_air_time[i] += dt
            else:
                self.feet_air_time[i] = 0.0
                self.feet_height_max[i] = 0.0

        self.last_contact = self.contact.copy()

        # --- REWARD CALCULATIONS ---
        rewards = []
        
        # 1. Track Lin Vel XY
        base_std = self.cmd_cfg.get('command_lin_vel_std', 0.5)
        command_speed_xy = np.linalg.norm(self.commands[:2])
        dynamic_lin_vel_denom = max(base_std * (command_speed_xy ** 2), 0.005)
        lin_vel_err = np.sum((self.vel[:2] - self.commands[:2])**2)
        rewards.append(self.rew_cfg.get('rew_scale_track_lin_vel_xy_exp', 1.5) * np.exp(-lin_vel_err / dynamic_lin_vel_denom))
        
        # 2. Track Ang Vel Z
        base_ang_std = self.cmd_cfg.get('command_ang_vel_std', 0.5)
        command_speed_w = abs(self.commands[2])
        dynamic_ang_vel_denom = max(base_ang_std * (command_speed_w ** 2), 0.005)
        ang_vel_err = (self.gyro[2] - self.commands[2])**2
        rewards.append(self.rew_cfg.get('rew_scale_track_ang_vel_z_exp', 1.5) * np.exp(-ang_vel_err / dynamic_ang_vel_denom))
        
        # 3. Feet Air Time
        is_moving = np.linalg.norm(self.commands[:3]) > self.cmd_cfg.get('static_velocity_threshold', 0.001)
        rewards.append(self.rew_cfg.get('rew_scale_feet_air_time', 1.0) * air_time_reward * float(is_moving))
        
        # 4. Foot Height, as in the env:
        #    rew_scale_foot_height_reward  (POSITIVE) pays the dense per-airborne-step match,
        #    rew_scale_foot_height_penalty (NEGATIVE) charges the per-landing apex mismatch.
        #    Published as one "foot_height" entry -- reward_names and PlotJuggler/reward_layout.xml
        #    index this list positionally, so the two directions are summed rather than split.
        rewards.append(
            (self.rew_cfg.get('rew_scale_foot_height_penalty', 0.0) * foot_height_mismatch
             + self.rew_cfg.get('rew_scale_foot_height_reward', 0.0) * foot_height_match)
            * float(is_moving)
        )
        
        # 5. Flat Orientation
        rewards.append(self.rew_cfg.get('rew_scale_flat_orientation_l2', -5.0) * np.sum(self.gyro[:2]**2))
        
        # 6. Lin Vel Z
        rewards.append(self.rew_cfg.get('rew_scale_lin_vel_z_l2', -2.0) * (self.vel[2]**2))
        
        # 7. Ang Vel XY
        rewards.append(self.rew_cfg.get('rew_scale_ang_vel_xy_l2', -0.05) * np.sum(self.gyro[:2]**2))
        
        # 8. DOF Pos
        rewards.append(self.rew_cfg.get('rew_scale_dof_pos_l2', -0.2) * np.sum(self.q**2))
        
        # 9. DOF Torques
        rewards.append(0.0) 
        
        # 10. DOF Acc
        rewards.append(self.rew_cfg.get('rew_scale_dof_acc_l2', -2.5e-7) * np.sum(dof_acc**2))
        
        # 11. Base Acc
        rewards.append(self.rew_cfg.get('rew_scale_base_acc_l2', -0.1) * np.sum(base_acc**2))
        
        # 12. Max Air Feet
        num_air_feet = 4.0 - np.sum(self.contact)
        max_allowed = self.rew_cfg.get('max_air_feet_allowed', 2.0)
        rewards.append(self.rew_cfg.get('rew_scale_max_air_feet', -0.5) * max(0.0, float(num_air_feet - max_allowed)))
        
        # 13. Feet Air Penalty
        rewards.append(self.rew_cfg.get('rew_scale_feet_air_penalty', -0.005) * float(num_air_feet))
        
        # 14. Feet Air Penalty Static
        static_mask = float(np.linalg.norm(self.commands[:3]) < self.cmd_cfg.get('static_velocity_threshold', 0.001))
        rewards.append(self.rew_cfg.get('rew_scale_feet_air_penalty_static', -5.0) * float(num_air_feet) * static_mask)
        
        # 15. Joint Vel L2 Static
        rewards.append(self.rew_cfg.get('rew_scale_joint_vel_l2_static', -1.0e-5) * np.sum(self.dq**2) * static_mask)

        # 15b. Grounded feet beyond feet_grounded_allowed, while a move command is active.
        # Appended AFTER joint_vel_l2_static so indices 0-14 keep their meaning -- reward_names
        # and PlotJuggler/reward_layout.xml address this list positionally.
        n_grounded = float(np.sum(self.contact > 0.5))
        grounded_excess = max(0.0, n_grounded - self.rew_cfg.get('feet_grounded_allowed', 2.0))
        rewards.append(
            self.rew_cfg.get('rew_scale_feet_grounded', 0.0) * grounded_excess * float(is_moving)
        )
        
        # 16. Total
        rewards.append(sum(rewards))
        
        # Publish
        msg_out = JointState()
        msg_out.header.stamp = self.get_clock().now().to_msg()
        msg_out.name = self.reward_names
        msg_out.position = [float(r) for r in rewards]
        self.pub_rewards.publish(msg_out)

def main(args=None):
    rclpy.init(args=args)
    node = RewardEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
