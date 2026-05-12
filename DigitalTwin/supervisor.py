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

    Fall-detection logic mirrors the training environment's _get_dones():
      projected_gravity = R.T @ g_world   (gravity vector in body frame)
      base_tilt_cos     = -proj_gravity_z / 9.81
        → 1.0 when perfectly upright, 0.0 when tilted 90°
      DANGER when base_tilt_cos < base_angle_termination_thresh (≈ 45°)
    """

    def __init__(self, robot_type: str = "go2"):
        super().__init__("supervisor_node")
        self.robot_type = robot_type

        # ------------------------------------------------------------------
        # 1. Load Configuration
        # ------------------------------------------------------------------
        self.config = load_config()
        self.safety_cfg = self.config.get("safety", {})
        self.freq = self.safety_cfg.get("supervisor_frequency", 10.0)

        # Fall-detection threshold — same semantic as training base_angle_termination_thresh.
        # Default 0.5 (≈60°) is slightly more lenient than training (0.7/≈45°) so we get
        # an early warning without cutting power on minor stumbles.
        self.tilt_thresh: float = float(
            self.safety_cfg.get("base_angle_termination_thresh", 0.5)
        )

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
        self._robot_safe: bool   = True   # False once a dangerous tilt is detected
        self._shutdown_logged: bool = False  # Avoid log spam

        # ------------------------------------------------------------------
        # 3. ROS Subscriptions
        # ------------------------------------------------------------------
        # Derived state from TelemetryManager — no geometry math needed here
        self.create_subscription(Vector3,    "/estimator/projected_gravity",
                                 self.proj_gravity_cb, 10)
        # Raw sensors (kept for future safety extensions)
        self.create_subscription(JointState, "/sensors/joint_states", self.joint_cb,  10)
        self.create_subscription(Odometry,   "/odom",                 self.odom_cb,   10)

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
            f"Tilt shutdown threshold: cos(θ) < {self.tilt_thresh:.2f}"
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

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self.base_pos = np.array([p.x, p.y, p.z])
        v = msg.twist.twist.linear
        self.base_lin_vel_body = np.array([v.x, v.y, v.z])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_dones(self) -> bool:
        """
        Mirrors training's _get_dones() fall condition.

        Reads the pre-computed projected gravity from /estimator/projected_gravity
        (published by TelemetryManager) and checks:
            base_tilt_cos = -proj_gravity_z / 9.81
              ≈ 1.0  → upright
              ≈ 0.0  → lying on side
              < 0    → upside down

        Returns True when the robot is dangerously tilted.
        """
        base_tilt_cos = -self.proj_gravity[2] / 9.81
        return bool(base_tilt_cos < self.tilt_thresh)

    # ------------------------------------------------------------------
    # Main safety loop
    # ------------------------------------------------------------------
    def safety_loop(self):
        """
        Periodically evaluates robot safety using the same termination
        criteria as the training environment.

        On dangerous tilt:
          - Publishes 0 Nm to /safety/max_torque  → CommandProcessor zeros all joints
          - Prints a prominent terminal warning
          - Latches the shutdown state (robot must be manually restarted)
        """
        msg = Float32()

        dangerous = self._get_dones()

        if dangerous and not self._shutdown_logged:
            # ── DANGER ───────────────────────────────────────────────
            tilt_cos = -self.proj_gravity[2] / 9.81
            tilt_deg = float(np.degrees(np.arccos(np.clip(tilt_cos, -1.0, 1.0))))

            print(
                f"\n{_BOLD}{_RED}"
                f"╔══════════════════════════════════════════════════════╗\n"
                f"║  ⚠  SAFETY SUPERVISOR — EMERGENCY SHUTDOWN  ⚠       ║\n"
                f"╠══════════════════════════════════════════════════════╣\n"
                f"║  Robot tilt detected!                                ║\n"
                f"║  cos(θ) = {tilt_cos:+.3f}  (threshold: {self.tilt_thresh:.2f})          ║\n"
                f"║  Estimated tilt angle: {tilt_deg:5.1f}°                      ║\n"
                f"║  ➜  Max torque set to 0 Nm — robot is limp.         ║\n"
                f"║  ➜  Press [ENTER] here to RESET and re-enable.      ║\n"
                f"╚══════════════════════════════════════════════════════╝"
                f"{_RESET}\n"
            )
            self.get_logger().error(
                f"[SAFETY] DANGEROUS TILT — cos(θ)={tilt_cos:.3f} < {self.tilt_thresh:.2f} "
                f"(≈{tilt_deg:.1f}°). Torque zeroed. Waiting for reset..."
            )
            self._robot_safe      = False
            self._shutdown_logged = True
            
            # Start background reset listener
            import threading
            threading.Thread(target=self._reset_worker, daemon=True).start()

        if self._robot_safe:
            msg.data = float(self.safety_cfg.get("global_max_torque", 23.5))
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
    args = parser.parse_args()

    rclpy.init()
    node = SupervisorNode(robot_type=args.robot)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
