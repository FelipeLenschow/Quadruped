import os
import sys
import time
import argparse
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32

# Ensure absolute path of the repository is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Configs.config_loader import load_config

# ---------------------------------------------------------------------------
# ANSI helpers for coloured terminal output
# ---------------------------------------------------------------------------
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


class SupervisorNode(Node):
    """
    Supervisor Node.
    Subscribes to robot telemetry and runs a safety check loop.
    Publishes max torque overrides to /safety/max_torque.
    """

    def __init__(self, robot_type: str = "go2", use_estimator: bool = False):
        super().__init__("supervisor_node")
        self.robot_type = robot_type

        # ------------------------------------------------------------------
        # 1. Load Configuration
        # ------------------------------------------------------------------
        self.config = load_config()
        self.safety_cfg = self.config.get("safety", {})
        self.freq = self.safety_cfg.get("supervisor_frequency", 10.0)

        # Fall-detection threshold — same semantic as training base_angle_termination_thresh.
        self.tilt_thresh: float = float(
            self.safety_cfg.get("base_angle_termination_thresh", 0.5)
        )
        self.forward_tilt_limit: float = float(
            self.safety_cfg.get("base_forward_tilt_limit_deg", 30.0)
        )
        self.rom_safety_margin: float = float(
            self.safety_cfg.get("joint_rom_safety_margin", 0.15)
        )

        # Dynamic max torque calculation based on percentage of motor limit
        self.motor_cfg = self.config.get("motor", {})
        self.motor_max_torque = float(self.motor_cfg.get("max_torque", 45.0))
        self.global_max_torque_percent = float(self.safety_cfg.get("global_max_torque_percent", 55.0))
        self.global_max_torque = (self.global_max_torque_percent / 100.0) * self.motor_max_torque

        # ------------------------------------------------------------------
        # Joint physical boundaries and 15% safety margins
        # ------------------------------------------------------------------
        limits = self.config.get("joint_limits", {})
        abd = limits.get("abduction", {"min": -1.047, "max": 1.047})
        hip_f = limits.get("hip_front", {"min": -1.571, "max": 3.491})
        hip_r = limits.get("hip_rear", {"min": -0.524, "max": 4.538})
        knee = limits.get("knee", {"min": -2.723, "max": -0.5})

        self.joint_limits_min = np.array(
            [abd["min"]] * 4 + [hip_f["min"]] * 2 + [hip_r["min"]] * 2 + [knee["min"]] * 4,
            dtype=np.float64
        )
        self.joint_limits_max = np.array(
            [abd["max"]] * 4 + [hip_f["max"]] * 2 + [hip_r["max"]] * 2 + [knee["max"]] * 4,
            dtype=np.float64
        )
        
        self.joint_rom = self.joint_limits_max - self.joint_limits_min
        self.joint_safe_min = self.joint_limits_min + self.rom_safety_margin * self.joint_rom
        self.joint_safe_max = self.joint_limits_max - self.rom_safety_margin * self.joint_rom
        
        self.joint_names_list = [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"
        ]

        # ------------------------------------------------------------------
        # 2. State Variables
        # ------------------------------------------------------------------
        # Projected gravity in body frame [x, y, z].
        # Default = [0, 0, -9.81] (upright robot).
        # Updated by /estimator/projected_gravity published by TelemetryManager.
        self.proj_gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)

        self.base_pos         = np.array([0.0, 0.0, 0.35], dtype=np.float64)
        self.joint_pos        = np.zeros(12, dtype=np.float64)
        self.base_lin_vel_body = np.zeros(3, dtype=np.float64)

        # Safety state machine
        self._robot_safe: bool   = True   # False once a dangerous condition is detected
        self._shutdown_logged: bool = False  # Avoid log spam
        self._has_received_joints: bool = False

        # ------------------------------------------------------------------
        # 3. ROS Subscriptions
        # ------------------------------------------------------------------
        # Derived state from TelemetryManager — no geometry math needed here
        self.create_subscription(Vector3,    "/estimator/projected_gravity",
                                 self.proj_gravity_cb, 10)
        # Raw sensors (kept for future safety extensions)
        self.create_subscription(JointState, "/sensors/joint_states", self.joint_cb,  10)
        
        odom_topic = "/odom/state_estimator" if use_estimator else "/odom/state_simulator"
        self.get_logger().info(f"Supervisor subscribing to: {odom_topic}")
        self.create_subscription(Odometry, odom_topic, self.odom_cb, 10)

        # ------------------------------------------------------------------
        # 4. ROS Publishers
        # ------------------------------------------------------------------
        self.max_torque_pub = self.create_publisher(Float32, "/safety/max_torque", 10)

        # ------------------------------------------------------------------
        # 5. Timer Loops
        # ------------------------------------------------------------------
        self.timer_period = 1.0 / self.freq
        self.create_timer(self.timer_period, self.safety_loop)

        self.get_logger().info(
            f"Supervisor Node initialised at {self.freq} Hz.  "
            f"Tilt threshold: cos(θ) < {self.tilt_thresh:.2f}, "
            f"Forward pitch threshold: {self.forward_tilt_limit:.1f}°, "
            f"Joint ROM safe boundary margin: {self.rom_safety_margin*100}%, "
            f"Global max torque limit: {self.global_max_torque:.2f} Nm ({self.global_max_torque_percent}%)"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def proj_gravity_cb(self, msg: Vector3):
        """Receives the pre-computed gravity vector in body frame from TelemetryManager."""
        self.proj_gravity = np.array([msg.x, msg.y, msg.z])

    def joint_cb(self, msg: JointState):
        if msg.position:
            self.joint_pos = np.array(msg.position[:12])
            self._has_received_joints = True

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self.base_pos = np.array([p.x, p.y, p.z])
        v = msg.twist.twist.linear
        self.base_lin_vel_body = np.array([v.x, v.y, v.z])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _check_joint_limits(self) -> tuple[bool, str]:
        """Check if any joint has violated its 15% safe boundary."""
        if not self._has_received_joints:
            return False, ""
            
        for i in range(12):
            val = self.joint_pos[i]
            if val < self.joint_safe_min[i]:
                msg = f"{self.joint_names_list[i]}: {val:+.3f} rad < safe min {self.joint_safe_min[i]:.3f}"
                return True, msg
            elif val > self.joint_safe_max[i]:
                msg = f"{self.joint_names_list[i]}: {val:+.3f} rad > safe max {self.joint_safe_max[i]:.3f}"
                return True, msg
        return False, ""

    def _get_dones(self) -> bool:
        """
        Evaluates active safety terminations.
        Returns True when any safety watchdog triggers a shutdown.
        """
        # 1. Base orientation tilt (fall)
        base_tilt_cos = -self.proj_gravity[2] / 9.81
        fall_danger = base_tilt_cos < self.tilt_thresh

        # 2. Forward tilt pitch detection
        pitch_rad = np.arctan2(self.proj_gravity[0], -self.proj_gravity[2])
        pitch_deg = np.degrees(pitch_rad)
        forward_danger = pitch_deg > self.forward_tilt_limit

        # 3. Joint range boundary detection
        joint_danger, _ = self._check_joint_limits()

        return bool(fall_danger or forward_danger or joint_danger)

    # ------------------------------------------------------------------
    # Main safety loop
    # ------------------------------------------------------------------
    def safety_loop(self):
        """
        Periodically evaluates robot safety.
        On violation, sets max torque to 0 Nm and logs detailed banner.
        """
        msg = Float32()

        # Check triggers
        base_tilt_cos = -self.proj_gravity[2] / 9.81
        tilt_deg = float(np.degrees(np.arccos(np.clip(base_tilt_cos, -1.0, 1.0))))
        
        pitch_rad = np.arctan2(self.proj_gravity[0], -self.proj_gravity[2])
        pitch_deg = float(np.degrees(pitch_rad))
        
        fall_danger = base_tilt_cos < self.tilt_thresh
        forward_danger = pitch_deg > self.forward_tilt_limit
        joint_danger, joint_msg = self._check_joint_limits()

        dangerous = fall_danger or forward_danger or joint_danger

        if dangerous and not self._shutdown_logged:
            reason = ""
            details_lines = []
            if fall_danger:
                reason = "Base orientation tilt (fall)"
                details_lines = [
                    f"cos(θ) = {base_tilt_cos:+.3f} (thresh: {self.tilt_thresh:.2f})",
                    f"Estimated tilt angle: {tilt_deg:5.1f}°"
                ]
            elif forward_danger:
                reason = "Dangerous forward tilt (pitch)"
                details_lines = [
                    f"Pitch angle: {pitch_deg:+.1f}° (thresh: {self.forward_tilt_limit:+.1f}°)"
                ]
            elif joint_danger:
                reason = "Joint range-of-motion boundary violation"
                details_lines = [
                    joint_msg
                ]

            print(f"\n{_BOLD}{_RED}╔══════════════════════════════════════════════════════╗")
            print(f"║  ⚠  SAFETY SUPERVISOR — EMERGENCY SHUTDOWN  ⚠       ║")
            print(f"╠══════════════════════════════════════════════════════╣")
            print(f"║  Reason: {reason:<43} ║")
            for line in details_lines:
                print(f"║  * {line:<49} ║")
            print(f"║                                                      ║")
            print(f"║  ➜  Max torque set to 0 Nm — robot is limp.         ║")
            print(f"║  ➜  Press [ENTER] here to RESET and re-enable.      ║")
            print(f"╚══════════════════════════════════════════════════════╝{_RESET}\n")

            self.get_logger().error(
                f"[SAFETY] EMERGENCY SHUTDOWN triggered! Reason: {reason}. Details: "
                f"{', '.join(details_lines)}. Torque zeroed. Waiting for reset..."
            )
            self._robot_safe      = False
            self._shutdown_logged = True
            
            # Start background reset listener
            import threading
            threading.Thread(target=self._reset_worker, daemon=True).start()

        if self._robot_safe:
            msg.data = float(self.global_max_torque)
        else:
            msg.data = 0.0

        self.max_torque_pub.publish(msg)

    def _reset_worker(self):
        """Background thread waiting for user to press Enter."""
        input()  # Wait for Enter
        print(f"{_YELLOW}{_BOLD}>>> Supervisor RESET. Re-enabling torque...{_RESET}")
        self._robot_safe = True
        self._shutdown_logged = False


def main():
    parser = argparse.ArgumentParser(description="Safety Supervisor for the Unitree Go2")
    parser.add_argument("--robot", type=str, default="go2", help="Robot model identifier")
    parser.add_argument("--use_estimator", action="store_true", help="Use estimated odometry instead of ground truth")
    args = parser.parse_args()

    rclpy.init()
    node = SupervisorNode(robot_type=args.robot, use_estimator=args.use_estimator)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
