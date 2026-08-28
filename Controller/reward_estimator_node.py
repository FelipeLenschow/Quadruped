"""Live estimate of the training reward, computed from robot telemetry.

This node exists so the numbers you watch on a real robot (or in a sim2sim driver) mean the
same thing as the numbers TensorBoard shows during training. Every term below therefore
mirrors compute_rewards() / _compute_reward_terms() in quadruped_env.py, and reads its scale
from the same training_phases.yaml the training run used -- resolved through the same
`inherits` chain, not the raw `default` block.

Some training terms need signals the robot does not have (contact forces, absolute base
height, articulation-level torques). Those are published as exactly 0.0 and listed in
UNAVAILABLE_TERMS, so a missing term reads as missing rather than quietly shifting the total.
"""

import os
import sys
import copy
import collections.abc
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
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))
from kinematics import Kinematics
from Controller.robot_defaults import DEFAULT_STANCE_QPOS

# Terms that exist in training but have no robot-side signal. Published as 0.0.
UNAVAILABLE_TERMS = (
    "dof_torques_l2",       # needs applied_torque from the articulation
    "base_height_l2",       # needs absolute base height over the terrain patch
    "undesired_contacts",   # needs thigh/calf contact sensors
    "grf_balance",          # needs per-foot normal force
    "grf_target",           # needs per-foot normal force
    "max_contact_force",    # needs per-foot normal force
    "pos_deviation",        # needs the training-side leashed reference pose
    "yaw_deviation",
    "foot_landing_vel",     # see note below
)
# foot_landing_vel is the one entry here that is approximable rather than truly absent: the env
# reads world-frame foot Z velocity on the step BEFORE touchdown, and on the robot that would be
# d/dt of the body-frame foot Z from get_foot_positions() plus the base's own vertical velocity.
# The second half is a Kalman output, and base Z is the least trustworthy channel the estimator
# has, so the result would look like the training term while being driven by estimator error.
# Reported as 0.0 until there is a foot velocity worth trusting.


def _deep_update(base, override):
    """Same merge semantics as resolve_phase_launcher() in launcher.py."""
    if not isinstance(base, collections.abc.Mapping):
        base = {}
    if not isinstance(override, collections.abc.Mapping):
        return base
    for k, v in override.items():
        if v is None:
            continue
        if isinstance(v, collections.abc.Mapping):
            base[k] = _deep_update(base.get(k, {}), v)
        else:
            base[k] = v
    return base


def resolve_phase(full_config, phase_name):
    """Resolve one phase through its `inherits` chain.

    The previous implementation did `full_config.get(phase_name)`, but phases live under the
    top-level `phases:` key -- so that lookup always missed and every run silently scored
    itself against the `default` block instead of the phase it actually trained with.
    """
    phases = full_config.get("phases", {}) or {}
    node = phases.get(phase_name, {}) or {}
    parent = node.get("inherits", "default")
    if parent and parent != phase_name and parent != "default":
        parent_cfg = resolve_phase(full_config, parent)
    else:
        parent_cfg = full_config.get("default", {})
    return _deep_update(copy.deepcopy(parent_cfg), node)


def quat_to_projected_gravity(qx, qy, qz, qw):
    """World -Z expressed in the base frame, matching Isaac's projected_gravity_b."""
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-9:
        return np.array([0.0, 0.0, -1.0])
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    # R^T @ [0, 0, -1] is just the negated third ROW of R, matching the quat_to_rot_matrix
    # convention used by policy_runner (quaternion ordered w, x, y, z).
    return np.array(
        [
            -(2.0 * qx * qz - 2.0 * qw * qy),
            -(2.0 * qy * qz + 2.0 * qw * qx),
            -(1.0 - 2.0 * qx * qx - 2.0 * qy * qy),
        ]
    )


class RewardEstimatorNode(Node):
    def __init__(self):
        super().__init__('reward_estimator_node')

        self.declare_parameter('robot_type', 'go2')
        self.declare_parameter('config_path', '')
        self.declare_parameter('phase', 'phase1')

        self.robot_type = self.get_parameter('robot_type').value
        config_path = self.get_parameter('config_path').value
        self.phase = self.get_parameter('phase').value

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

        if self.phase not in (full_config.get("phases", {}) or {}):
            self.get_logger().warn(
                f"Phase '{self.phase}' is not defined in {os.path.basename(config_path)}; "
                f"falling back to the default block. Rewards will NOT match a phase run."
            )
        self.cfg = resolve_phase(full_config, self.phase)

        self.rew_cfg = self.cfg.get('rewards', {})
        self.cmd_cfg = self.cfg.get('commands', {})
        self.env_cfg = self.cfg.get('env', {})
        self.get_logger().info(
            f"[RewardEstimator] Resolved '{self.phase}': "
            f"sigma_exp={self._cmd('vel_tracking_sigma_exp')} "
            f"gait_sym={self._rew('rew_scale_gait_phase_sym')} "
            f"track_lin={self._rew('rew_scale_track_lin_vel_xy_exp')}"
        )
        self.get_logger().info(
            f"[RewardEstimator] No robot-side signal for: {', '.join(UNAVAILABLE_TERMS)} "
            f"-- these publish 0.0 and are excluded from the total."
        )

        self.kin = Kinematics(self.robot_type)
        self.action_scale = float(self.env_cfg.get('action_scale', 0.25))
        self.default_qpos = DEFAULT_STANCE_QPOS.astype(np.float64)

        # Per-joint weights for dof_pos_l2, mirroring dof_pos_l2_joint_weights in the env.
        hip_mult = float(self._rew('dof_pos_l2_hip_mult', 1.0))
        self.dof_pos_weights = np.array([hip_mult] * 4 + [1.0] * 8, dtype=np.float64)

        self.sub_cmd     = self.create_subscription(Twist, '/cmd_vel', self.cmd_cb, 10)
        self.sub_joints  = self.create_subscription(JointState, '/sensors/joint_states', self.joint_cb, 10)
        self.sub_imu     = self.create_subscription(Imu, '/sensors/imu', self.imu_cb, 10)
        self.sub_vel     = self.create_subscription(Vector3, '/estimator/base_lin_vel', self.vel_cb, 10)
        self.sub_contact = self.create_subscription(Float32MultiArray, '/estimator/feet_contact', self.contact_cb, 10)
        self.sub_targets = self.create_subscription(JointState, '/commands/joint_commands', self.target_cb, 10)

        self.pub_rewards = self.create_publisher(JointState, '/policy_rewards', 10)
        self.timer = self.create_timer(0.02, self.timer_cb)  # 50Hz evaluation

        self.commands = np.zeros(3)  # vx, vy, wz
        self.q = np.zeros(12)
        self.dq = np.zeros(12)
        self.gyro = np.zeros(3)
        self.proj_grav = np.array([0.0, 0.0, -1.0])
        self.vel = np.zeros(3)
        self.contact = np.zeros(4)
        self.actions = np.zeros(12)

        self.last_joint_vel = np.zeros(12)
        self.last_base_vel = np.zeros(3)
        self.last_actions = np.zeros(12)
        self.last_time = time.time()

        self.feet_air_time = np.zeros(4)
        # Peak height of the current swing, per foot -- the foot-height terms are scored on this
        # at touchdown, mirroring feet_height_max in quadruped_env.py.
        self.feet_height_max = np.zeros(4)
        self.last_contact = np.zeros(4)
        self.feet_height_max = np.zeros(4)

        # Gait phase bookkeeping (mirrors the strike-time tracking in _compute_reward_terms)
        self.t_now = 0.0
        self.last_strike_time = np.zeros(4)
        self.stride_duration = np.zeros(4)

        self.reward_names = [
            "alive",
            "track_lin_vel_xy_exp",
            "track_ang_vel_z_exp",
            "lin_vel_z_l2",
            "ang_vel_xy_l2",
            "flat_orientation_l2",
            "dof_pos_l2_walk",
            "dof_pos_l2_stance",
            "dof_acc_l2",
            "base_acc_l2",
            "action_rate_l2",
            "feet_air_time",
            "foot_height_penalty",
            "foot_height_reward",
            "feet_air_penalty",
            "feet_air_penalty_static",
            "joint_vel_l2_static",
            "gait_phase_sym",
        ] + list(UNAVAILABLE_TERMS) + ["total_reward"]

    # -- config helpers: no stale hardcoded fallbacks, a miss is reported once --
    def _rew(self, key, default=0.0):
        if key not in self.rew_cfg:
            self._warn_missing(f"rewards.{key}")
        return self.rew_cfg.get(key, default)

    def _cmd(self, key, default=0.0):
        if key not in self.cmd_cfg:
            self._warn_missing(f"commands.{key}")
        return self.cmd_cfg.get(key, default)

    def _warn_missing(self, key):
        if not hasattr(self, "_warned"):
            self._warned = set()
        if key not in self._warned:
            self._warned.add(key)
            self.get_logger().warn(f"[RewardEstimator] '{key}' missing from config; using 0.")

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
        o = msg.orientation
        self.proj_grav = quat_to_projected_gravity(o.x, o.y, o.z, o.w)

    def vel_cb(self, msg: Vector3):
        self.vel = np.array([msg.x, msg.y, msg.z])

    def contact_cb(self, msg: Float32MultiArray):
        if len(msg.data) >= 4:
            self.contact = np.array(msg.data[:4])

    def target_cb(self, msg: JointState):
        """Recover the policy action from the commanded targets: a = (target - default)/scale."""
        if len(msg.position) >= 12 and self.action_scale > 0:
            self.actions = (np.array(msg.position[:12]) - self.default_qpos) / self.action_scale

    def _gait_phase_sym(self, first_contact, dt, moving):
        """Cosine score over the six leg pairs -- see _compute_reward_terms in quadruped_env."""
        for foot in range(4):
            if first_contact[foot]:
                dur = self.t_now - self.last_strike_time[foot]
                if dur > 0.1:
                    self.stride_duration[foot] = dur
                self.last_strike_time[foot] = self.t_now

        pairs = [
            (0, 1, self._rew('gait_phase_offset_front', 0.5)),
            (2, 3, self._rew('gait_phase_offset_rear', 0.5)),
            (0, 2, self._rew('gait_phase_offset_left', 0.5)),
            (1, 3, self._rew('gait_phase_offset_right', 0.5)),
            (0, 3, self._rew('gait_phase_offset_diag1', 0.0)),
            (1, 2, self._rew('gait_phase_offset_diag2', 0.0)),
        ]
        total_score, total_weight = 0.0, 0.0
        for ref, other, target in pairs:
            dur = self.stride_duration[ref]
            active = (self.t_now - self.last_strike_time[ref]) < (dur * 1.5)
            if not (dur > 0.1 and moving and active):
                continue
            phase = ((self.last_strike_time[other] - self.last_strike_time[ref]) % dur) / dur
            total_score += 0.5 * (1.0 + np.cos((phase - target) * 2.0 * np.pi))
            total_weight += 1.0
        if total_weight == 0.0:
            return 0.0
        return total_score / total_weight

    def timer_cb(self):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0:
            dt = 0.02
        self.last_time = current_time
        self.t_now += dt

        dof_acc = (self.dq - self.last_joint_vel) / dt
        self.last_joint_vel = self.dq.copy()

        base_acc = (self.vel - self.last_base_vel) / dt
        self.last_base_vel = self.vel.copy()

        # Smooth static/moving ramp -- the same one every stepping reward in the env uses.
        cmd_norm = np.linalg.norm(self.commands[:3])
        ramp_lo = float(self._cmd('static_velocity_threshold', 0.001))
        ramp_hi = max(float(self._cmd('static_command_ramp', 0.1)), ramp_lo + 1e-6)
        moving_mask = float(np.clip((cmd_norm - ramp_lo) / (ramp_hi - ramp_lo), 0.0, 1.0))
        static_mask = 1.0 - moving_mask

        contact_b = self.contact > 0.5
        first_contact = contact_b & ~(self.last_contact > 0.5)
        airborne = ~contact_b

        # -- feet air time: potential-based shaping with a speed-dependent target --
        command_speed_xy = np.linalg.norm(self.commands[:2])
        speed_lo = float(self._rew('feet_air_time_speed_lo', 0.1))
        speed_hi = max(float(self._rew('feet_air_time_speed_hi', 0.35)), speed_lo + 1e-6)
        slow_frac = 1.0 - np.clip((command_speed_xy - speed_lo) / (speed_hi - speed_lo), 0.0, 1.0)
        target_fast = float(self._rew('target_feet_air_time', 0.5))
        target_slow = float(self._rew('target_feet_air_time_slow', 0.5))
        target_air = target_fast + (target_slow - target_fast) * slow_frac
        sigma_air = max(float(self._rew('feet_air_time_sigma', 0.06)), 1e-9)

        self.feet_air_time = np.where(contact_b, 0.0, self.feet_air_time + dt)
        air_err = self.feet_air_time - target_air
        phi_air = np.exp(-np.square(air_err) / sigma_air)
        dphi_dt = -2.0 * air_err / sigma_air * phi_air
        air_time_val = float(np.sum(dphi_dt * dt * airborne)) * moving_mask

        # -- foot height: swing apex, scored ONCE per landing, as the env does --
        # The env charges/pays for the apex on the step the foot touches down, not on every
        # airborne step, so the value depends on the apex alone: a slow high-clearance swing and
        # a quick one that reaches the same height score identically, and this term never bids
        # against feet_air_time over swing duration. feet_height_max is the running peak of the
        # current swing and is reset below, AFTER scoring, so it still holds this swing's apex on
        # the landing step. Split into the two scales the env uses -- penalty on the mismatch
        # (negative scale) and reward on the match (positive scale) -- rather than the single
        # rew_scale_foot_height_exp of the older reward set, which no longer exists.
        foot_positions = self.kin.get_foot_positions(self.q)
        foot_heights_z = foot_positions[:, 2] + 0.30
        target_fh = float(self._rew('target_foot_height', 0.1))
        sigma_fh = max(float(self._rew('foot_height_sigma', 0.01)), 1e-9)
        self.feet_height_max = np.maximum(self.feet_height_max, foot_heights_z)
        fh_match = np.exp(-np.square(self.feet_height_max - target_fh) / sigma_fh)
        landed_moving = first_contact.astype(np.float64) * moving_mask
        foot_height_penalty_val = float(np.sum((1.0 - fh_match) * landed_moving))
        foot_height_reward_val = float(np.sum(fh_match * landed_moving))
        self.feet_height_max = np.where(contact_b, 0.0, self.feet_height_max)

        gait_sym_val = self._gait_phase_sym(first_contact, dt, cmd_norm > ramp_lo) * moving_mask
        self.last_contact = self.contact.copy()

        # -- velocity tracking, with the configured dynamic sigma exponent --
        sigma_exp = float(self._cmd('vel_tracking_sigma_exp', 0.0))
        lin_std = float(self._cmd('command_lin_vel_std', 0.5))
        ang_std = float(self._cmd('command_ang_vel_std', 0.5))
        lin_denom = max(lin_std * (command_speed_xy ** sigma_exp), 0.005)
        ang_denom = max(ang_std * (abs(self.commands[2]) ** sigma_exp), 0.005)
        lin_err = float(np.sum((self.vel[:2] - self.commands[:2]) ** 2))
        ang_err = float((self.gyro[2] - self.commands[2]) ** 2)

        # -- posture, split static/moving with the hip weighting --
        dof_pos_err = float(np.sum(np.square(self.q - self.default_qpos) * self.dof_pos_weights))
        air_penalty_val = float(np.sum(self.feet_air_time * airborne))

        terms = {
            "alive": self._rew('rew_scale_alive') * 1.0,
            "track_lin_vel_xy_exp": self._rew('rew_scale_track_lin_vel_xy_exp') * np.exp(-lin_err / lin_denom),
            "track_ang_vel_z_exp": self._rew('rew_scale_track_ang_vel_z_exp') * np.exp(-ang_err / ang_denom),
            "lin_vel_z_l2": self._rew('rew_scale_lin_vel_z_l2') * (self.vel[2] ** 2),
            "ang_vel_xy_l2": self._rew('rew_scale_ang_vel_xy_l2') * float(np.sum(self.gyro[:2] ** 2)),
            "flat_orientation_l2": self._rew('rew_scale_flat_orientation_l2') * float(np.sum(self.proj_grav[:2] ** 2)),
            "dof_pos_l2_walk": self._rew('rew_scale_dof_pos_l2_walk') * dof_pos_err * moving_mask,
            "dof_pos_l2_stance": self._rew('rew_scale_dof_pos_l2_stance') * dof_pos_err * static_mask,
            "dof_acc_l2": self._rew('rew_scale_dof_acc_l2') * float(np.sum(dof_acc ** 2)),
            "base_acc_l2": self._rew('rew_scale_base_acc_l2') * float(np.sum(base_acc ** 2)),
            "action_rate_l2": self._rew('rew_scale_action_rate_l2') * float(np.sum((self.actions - self.last_actions) ** 2)),
            "feet_air_time": self._rew('rew_scale_feet_air_time') * air_time_val,
            "foot_height_penalty": self._rew('rew_scale_foot_height_penalty') * foot_height_penalty_val,
            "foot_height_reward": self._rew('rew_scale_foot_height_reward') * foot_height_reward_val,
            "feet_air_penalty": self._rew('rew_scale_feet_air_penalty') * air_penalty_val,
            "feet_air_penalty_static": self._rew('rew_scale_feet_air_penalty_static') * air_penalty_val * static_mask,
            "joint_vel_l2_static": self._rew('rew_scale_joint_vel_l2_static') * float(np.sum(self.dq ** 2)) * static_mask,
            "gait_phase_sym": self._rew('rew_scale_gait_phase_sym') * gait_sym_val,
        }
        self.last_actions = self.actions.copy()

        for name in UNAVAILABLE_TERMS:
            terms[name] = 0.0

        values = [float(terms[name]) for name in self.reward_names[:-1]]
        values.append(float(sum(values)))

        msg_out = JointState()
        msg_out.header.stamp = self.get_clock().now().to_msg()
        msg_out.name = self.reward_names
        msg_out.position = values
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
