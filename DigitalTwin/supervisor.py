"""
Safety Supervisor — Heartbeat Broadcaster.

Reads safety parameters from config.yaml and broadcasts them on separate
ROS 2 Float32 topics at a configurable frequency.  The robot's internal
CommandSafetyProcessor subscribes to these topics and performs all actual
safety evaluation.

This node acts as a dead-man's switch: if it stops publishing, the robot's
internal watchdog will detect the loss and disable torque.
"""

import os
import sys
import time
import argparse

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

# Ensure absolute path of the repository is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Configs.config_loader import load_config

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------
_GREEN  = "\033[92m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


class SupervisorNode(Node):
    """
    Heartbeat Supervisor.

    Publishes safety configuration parameters on separate Float32 topics:
      /safety/heartbeat                   — alive signal (timestamp)
      /safety/max_torque_percent          — max torque as % of motor capacity
      /safety/base_tilt_limit_deg         — base tilt shutdown threshold
      /safety/base_forward_tilt_limit_deg — forward pitch shutdown threshold
      /safety/joint_rom_safety_margin     — joint ROM safe boundary fraction

    All values are read from config.yaml at startup and broadcast every cycle.
    The parameters are designed to be changeable on-the-fly in the future
    (e.g., via a GUI or ros2 topic pub override).
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

        # Safety parameters (read from config, broadcast to robot)
        self.motor_cfg = self.config.get("motor", {})
        self.motor_max_torque = float(self.motor_cfg.get("max_torque", 45.0))

        self.max_torque_percent = float(
            self.safety_cfg.get("global_max_torque_percent", 55.0))
        self.base_tilt_limit_deg = float(
            self.safety_cfg.get("base_tilt_limit_deg", 30.0))
        self.base_forward_tilt_limit_deg = float(
            self.safety_cfg.get("base_forward_tilt_limit_deg", 30.0))
        self.joint_rom_safety_margin = float(
            self.safety_cfg.get("joint_rom_safety_margin", 0.15))
        self.watchdog_timeout = float(
            self.safety_cfg.get("watchdog_timeout", 1.0))

        # ------------------------------------------------------------------
        # 2. ROS Publishers (separate Float32 topics)
        # ------------------------------------------------------------------
        self.heartbeat_pub = self.create_publisher(
            Float32, "/safety/heartbeat", 10)
        self.max_torque_percent_pub = self.create_publisher(
            Float32, "/safety/max_torque_percent", 10)
        self.base_tilt_pub = self.create_publisher(
            Float32, "/safety/base_tilt_limit_deg", 10)
        self.forward_tilt_pub = self.create_publisher(
            Float32, "/safety/base_forward_tilt_limit_deg", 10)
        self.rom_margin_pub = self.create_publisher(
            Float32, "/safety/joint_rom_safety_margin", 10)

        # ------------------------------------------------------------------
        # 3. Timer
        # ------------------------------------------------------------------
        self.timer_period = 1.0 / self.freq
        self.create_timer(self.timer_period, self.heartbeat_loop)
        self.heartbeat_count = 0

        max_torque_nm = (self.max_torque_percent / 100.0) * self.motor_max_torque
        self.get_logger().info(
            f"Supervisor Heartbeat initialized at {self.freq} Hz.\n"
            f"  Max torque: {self.max_torque_percent}% = {max_torque_nm:.1f} Nm\n"
            f"  Tilt limit: {self.base_tilt_limit_deg}°\n"
            f"  Forward tilt limit: {self.base_forward_tilt_limit_deg}°\n"
            f"  Joint ROM margin: {self.joint_rom_safety_margin * 100}%\n"
            f"  Watchdog timeout: {self.watchdog_timeout}s"
        )

    # ------------------------------------------------------------------
    # Heartbeat Loop
    # ------------------------------------------------------------------
    def heartbeat_loop(self):
        """Publish all safety parameters on their respective topics."""
        now = time.time()

        # Core heartbeat (alive signal)
        self.heartbeat_pub.publish(Float32(data=float(now)))

        # Safety parameters
        self.max_torque_percent_pub.publish(
            Float32(data=float(self.max_torque_percent)))
        self.base_tilt_pub.publish(
            Float32(data=float(self.base_tilt_limit_deg)))
        self.forward_tilt_pub.publish(
            Float32(data=float(self.base_forward_tilt_limit_deg)))
        self.rom_margin_pub.publish(
            Float32(data=float(self.joint_rom_safety_margin)))

        # Periodic console output (every 5 seconds)
        self.heartbeat_count += 1
        if self.heartbeat_count % int(self.freq * 5) == 0:
            max_nm = (self.max_torque_percent / 100.0) * self.motor_max_torque
            print(
                f"\r{_GREEN}[Supervisor]{_RESET} ♥ Heartbeat #{self.heartbeat_count} | "
                f"torque={self.max_torque_percent}% ({max_nm:.1f}Nm) | "
                f"tilt={self.base_tilt_limit_deg}° | "
                f"pitch={self.base_forward_tilt_limit_deg}° | "
                f"ROM margin={self.joint_rom_safety_margin*100:.0f}%   ",
                end="", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Safety Supervisor — Heartbeat Broadcaster")
    parser.add_argument("--robot", type=str, default="go2",
                        help="Robot model identifier")
    # Keep --use_estimator for CLI compatibility but it's unused now
    parser.add_argument("--use_estimator", action="store_true",
                        help="(Legacy, unused)")
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
