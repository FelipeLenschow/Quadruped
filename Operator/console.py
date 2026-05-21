"""
Console — Operator Command Center.

Reads safety parameters from config.yaml and broadcasts them on separate
ROS 2 Float32 topics at a configurable frequency.  The robot's internal
CommandSafetyProcessor subscribes to these topics and performs all actual
safety evaluation.

Additionally handles pipeline mode switching and pose commands for the
PoseGenerator.  The Console is the single operator interface for all
runtime parameters.

This node acts as a dead-man's switch: if it stops publishing, the robot's
internal watchdog will detect the loss and disable torque.
"""

import os
import sys
import time
import argparse

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String

# Ensure absolute path of the repository is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Configs.config_loader import load_config

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------
_GREEN  = "\033[92m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"
_CYAN   = "\033[96m"
_DIM    = "\033[2m"
_BG_GREEN = "\033[42m"
_BG_DEFAULT = "\033[49m"


class ConsoleNode(Node):
    """
    Operator Console.

    Publishes safety configuration parameters on separate Float32 topics:
      /safety/heartbeat                   — alive signal (timestamp)
      /safety/max_torque_percent          — max torque as % of motor capacity
      /safety/base_tilt_limit_deg         — base tilt shutdown threshold
      /safety/base_forward_tilt_limit_deg — forward pitch shutdown threshold
      /safety/joint_rom_safety_margin     — joint ROM safe boundary fraction

    Pipeline and Pose control:
      /pipeline/mode       — "pose" or "policy"
      /pose/command        — named pose (e.g. "stand", "pushup")
      /pose/interp_duration — interpolation time override

    All values are read from config.yaml at startup and broadcast every cycle.
    The parameters are designed to be changeable on-the-fly via the terminal.
    """

    def __init__(self, robot_type: str = "go2"):
        super().__init__("console_node")
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

        self.control_cfg = self.config.get("control", {})
        self.kp = float(self.control_cfg.get("kp", 0.0))
        self.kd = float(self.control_cfg.get("kd", 0.0))

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

        # Pipeline / Pose state (tracked locally since we send the commands)
        self.current_mode = "pose"
        self.pose_status_name = "none"
        self.pose_status_progress = 0.0
        self.pose_status_label = ""

        # Freeze Base state
        self.freeze_base = False

        # ------------------------------------------------------------------
        # 2. ROS Publishers — Safety Heartbeat
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
        self.kp_pub = self.create_publisher(
            Float32, "/control/kp", 10)
        self.kd_pub = self.create_publisher(
            Float32, "/control/kd", 10)

        # ------------------------------------------------------------------
        # 3. ROS Publishers — Pipeline & Pose Control
        # ------------------------------------------------------------------
        self.mode_pub = self.create_publisher(
            String, "/pipeline/mode", 10)
        self.pose_cmd_pub = self.create_publisher(
            String, "/pose/command", 10)
        self.pose_interp_pub = self.create_publisher(
            Float32, "/pose/interp_duration", 10)
        self.freeze_base_pub = self.create_publisher(
            Bool, "/base/freeze", 10)

        # ------------------------------------------------------------------
        # 4. ROS Subscriber — Pose Status Feedback
        # ------------------------------------------------------------------
        self.create_subscription(
            String, "/pose/status", self._pose_status_cb, 10)

        # ------------------------------------------------------------------
        # 5. Timer
        # ------------------------------------------------------------------
        self.timer_period = 1.0 / self.freq
        self.create_timer(self.timer_period, self.heartbeat_loop)
        self.heartbeat_count = 0
        self.input_submitted = False
        self.last_command = "None"

        # ------------------------------------------------------------------
        # 6. Command Input Thread
        # ------------------------------------------------------------------
        import threading
        self.cmd_thread = threading.Thread(target=self._command_listener, daemon=True)
        self.cmd_thread.start()

        # Publish initial gains once at startup
        self.kp_pub.publish(Float32(data=float(self.kp)))
        self.kd_pub.publish(Float32(data=float(self.kd)))

    # ------------------------------------------------------------------
    # Pose Status Feedback
    # ------------------------------------------------------------------
    def _pose_status_cb(self, msg: String):
        """Parse pose status: 'pose_name|progress|label'."""
        try:
            parts = msg.data.split("|")
            if len(parts) >= 3:
                self.pose_status_name = parts[0]
                self.pose_status_progress = float(parts[1])
                self.pose_status_label = parts[2]
        except (ValueError, IndexError):
            pass

    def _command_listener(self):
        """Listens for user commands from stdin to dynamically configure parameters."""
        import re
        import readline

        # Autocomplete options
        COMMANDS = [
            "Torque Limit = ",
            "Max Roll = ",
            "Max Pitch = ",
            "Joint ROM = ",
            "Watchdog = ",
            "Kp = ",
            "Kd = ",
            "Mode = pose",
            "Mode = policy",
            "Mode = ",
            "Pose = stand",
            "Pose = lie_flat",
            "Pose = sit",
            "Pose = pushup",
            "Pose = ",
            "Interp = ",
            "Freeze Base = on",
            "Freeze Base = off",
        ]

        def completer(text, state):
            # Case-insensitive prefix match
            options = [cmd for cmd in COMMANDS if cmd.lower().startswith(text.lower())]
            if state < len(options):
                return options[state]
            return None

        readline.set_completer(completer)
        # Treat the entire line as a single word to correctly autocomplete space-separated command names
        readline.set_completer_delims("")
        readline.parse_and_bind("tab: complete")

        while rclpy.ok():
            try:
                line = input()
                
                # Dynamic terminal scroll tracking
                self.input_submitted = True
                
                raw_line = line.strip()
                if raw_line:
                    self.last_command = raw_line
                    # Store submitted command in GNU readline history
                    readline.add_history(raw_line)

                # Enforce strict pattern: <Parameter Name> = <Value>
                match = re.match(r"^\s*([a-zA-Z\s]+)\s*=\s*(.+)\s*$", raw_line)
                if not match:
                    continue

                param = match.group(1).lower().strip()
                val_str = match.group(2).strip()

                # --- Numeric parameters ---
                if param in ("torque limit", "torque"):
                    self.max_torque_percent = float(val_str)
                elif param in ("max pitch", "pitch"):
                    self.base_forward_tilt_limit_deg = float(val_str)
                elif param in ("max roll", "roll"):
                    self.base_tilt_limit_deg = float(val_str)
                elif param in ("rom", "joint rom", "joint", "margin"):
                    self.joint_rom_safety_margin = float(val_str) / 100.0
                elif param in ("timeout", "watchdog"):
                    self.watchdog_timeout = float(val_str)
                elif param in ("kp", "p gain"):
                    self.kp = float(val_str)
                    self.kp_pub.publish(Float32(data=float(self.kp)))
                elif param in ("kd", "d gain"):
                    self.kd = float(val_str)
                    self.kd_pub.publish(Float32(data=float(self.kd)))

                # --- Pipeline mode ---
                elif param == "mode":
                    mode_val = val_str.lower().strip()
                    if mode_val in ("pose", "policy"):
                        self.current_mode = mode_val
                        msg = String()
                        msg.data = mode_val
                        self.mode_pub.publish(msg)
                    else:
                        pass  # silently ignore invalid modes

                # --- Pose commands ---
                elif param == "pose":
                    pose_name = val_str.lower().strip()
                    msg = String()
                    msg.data = pose_name
                    self.pose_cmd_pub.publish(msg)
                    # Auto-switch to pose mode if not already
                    if self.current_mode != "pose":
                        self.current_mode = "pose"
                        mode_msg = String()
                        mode_msg.data = "pose"
                        self.mode_pub.publish(mode_msg)

                # --- Interpolation duration ---
                elif param in ("interp", "interpolation", "duration"):
                    dur = float(val_str)
                    if dur > 0.0:
                        msg = Float32()
                        msg.data = dur
                        self.pose_interp_pub.publish(msg)

                # --- Freeze Base ---
                elif param in ("freeze base", "freeze"):
                    active = val_str.lower().strip() in ("on", "true", "1", "yes")
                    self.freeze_base = active
                    msg = Bool()
                    msg.data = active
                    self.freeze_base_pub.publish(msg)

            except (EOFError, KeyboardInterrupt):
                break
            except Exception:
                pass

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
        self.kp_pub.publish(
            Float32(data=float(self.kp)))
        self.kd_pub.publish(
            Float32(data=float(self.kd)))

        # Also re-publish mode on every heartbeat so late-joining nodes pick it up
        mode_msg = String()
        mode_msg.data = self.current_mode
        self.mode_pub.publish(mode_msg)

        # Console status report on every heartbeat (replaces the last multi-line block in-place)
        self.heartbeat_count += 1
        
        # Check if user just submitted input (which prints a terminal newline and scrolls)
        use_scroll_adjust = self.input_submitted

        # Total display lines (including blanks): 14
        DISPLAY_LINES = 14
        
        if self.heartbeat_count > 1:
            if use_scroll_adjust:
                # Move up to compensate for terminal scroll/newline
                print(f"\033[{DISPLAY_LINES + 1}A", end="")
                self.input_submitted = False
            else:
                # Save cursor position and move up
                print(f"\033[s\033[{DISPLAY_LINES}A", end="")

        max_nm = (self.max_torque_percent / 100.0) * self.motor_max_torque

        # --- Build the Mode line with progress bar background fill ---
        mode_label = self.current_mode.upper()
        if self.current_mode == "pose":
            progress = self.pose_status_progress
            status_name = self.pose_status_name or "idle"
            status_label = self.pose_status_label or ""

            if status_label:
                mode_text = f" ├─ Mode         : {mode_label} ({status_name} — {status_label})"
            else:
                mode_text = f" ├─ Mode         : {mode_label} ({status_name})"

            # Build progress bar via background-color fill
            # Fill proportion of the line width with green background
            line_width = max(len(mode_text), 56)
            fill_chars = int(progress * line_width)
            filled_part = mode_text[:fill_chars]
            unfilled_part = mode_text[fill_chars:]
            # Pad to full width for clean rendering
            filled_part = filled_part.ljust(fill_chars)
            unfilled_part = unfilled_part.ljust(line_width - fill_chars)
            mode_line = f"\r{_BG_GREEN}{filled_part}{_BG_DEFAULT}{unfilled_part}\033[K"
        else:
            mode_line = f"\r ├─ Mode        : {mode_label} (NN policy active)\033[K"

        print(f"\r\033[K")
        print(f"\r\033[K")
        print(f"\r  {_CYAN}[Last Command]{_RESET}: {self.last_command}\033[K")
        print(f"\r\033[K")
        print(f"\r{_GREEN}[Console]{_RESET} Heartbeat #{self.heartbeat_count}\033[K")
        print(mode_line)
        print(f"\r ├─ Torque Limit : {self.max_torque_percent}% ({max_nm:.1f} Nm)\033[K")
        print(f"\r ├─ Max Roll     : {self.base_tilt_limit_deg} deg\033[K")
        print(f"\r ├─ Max Pitch    : {self.base_forward_tilt_limit_deg} deg\033[K")
        print(f"\r ├─ Joint ROM    : {self.joint_rom_safety_margin*100:.0f}% margin\033[K")
        print(f"\r ├─ Active Kp    : {self.kp:.1f}\033[K")
        print(f"\r ├─ Active Kd    : {self.kd:.2f}\033[K")
        print(f"\r ├─ Watchdog     : {self.watchdog_timeout:.2f}s timeout\033[K")
        freeze_indicator = f"{_CYAN}\u2744 FROZEN @ 1.0m{_RESET}" if self.freeze_base else "\u25cb off"
        print(f"\r \u2514\u2500 Base Freeze  : {freeze_indicator}\033[K")

        if self.heartbeat_count > 1 and not use_scroll_adjust:
            # Restore saved cursor position for standard continuous typing
            print("\033[u", end="", flush=True)
        else:
            # Clear any typed text from this line and position cursor at the start of it
            print("\033[K\r", end="", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Console — Operator Command Center")
    parser.add_argument("--robot", type=str, default="go2",
                        help="Robot model identifier")
    # Keep --use_estimator for CLI compatibility but it's unused now
    parser.add_argument("--use_estimator", action="store_true",
                        help="(Legacy, unused)")
    args = parser.parse_args()

    rclpy.init()
    node = ConsoleNode(robot_type=args.robot)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
