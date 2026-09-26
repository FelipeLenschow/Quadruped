"""
Keyboard teleop - /cmd_vel with teleop_twist_keyboard's keys, plus the Walk These Ways gait commands.

Velocity (teleop_twist_keyboard layout, speeds scale with q/z, w/x, e/c):
    u i o      forward-left / forward / forward-right
    j k l      turn left / stop / turn right
    m , .      back-left / back / back-right
    U I O / J K L / M < >   the same with strafing instead of turning
    space      stop

Gait (published on /gait_command; a policy without a gait input ignores it):
    1 2 3 4 5  walk / trot / pace / bound / pronk
    [ ]        step frequency -/+ 0.25 Hz
    - =        duty -/+ 0.05
    r f        body pitch nose down / nose up, 0.05 rad
    y h        body height up / down, 0.01 m
    t g        swing height up / down, 0.01 m
    v b        stance width narrower / wider, 0.01 m
    0          gait back to nominal
    Ctrl-C     quit (sends a zero velocity)

Runs in place of teleop_twist_keyboard: two /cmd_vel sources fight. PolicyRunner clamps the gait
commands to the ranges the run trained on, so the values shown here can read wider than what the
policy gets.
"""

import argparse
import select
import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

GAITS = {
    "walk": (0.0, 0.75, 0.5),
    "trot": (0.5, 0.0, 0.0),
    "pace": (0.0, 0.0, 0.5),
    "bound": (0.0, 0.5, 0.0),
    "pronk": (0.0, 0.0, 0.0),
}
GAIT_KEYS = {"1": "walk", "2": "trot", "3": "pace", "4": "bound", "5": "pronk"}

# (x, y, yaw) direction per key, as teleop_twist_keyboard.
MOVE = {
    "i": (1, 0, 0), "o": (1, 0, -1), "j": (0, 0, 1), "l": (0, 0, -1),
    "u": (1, 0, 1), ",": (-1, 0, 0), ".": (-1, 0, 1), "m": (-1, 0, -1),
    "I": (1, 0, 0), "O": (1, -1, 0), "J": (0, 1, 0), "L": (0, -1, 0),
    "U": (1, 1, 0), "<": (-1, 0, 0), ">": (-1, -1, 0), "M": (-1, 1, 0),
    "k": (0, 0, 0), "K": (0, 0, 0), " ": (0, 0, 0),
}
# (linear factor, angular factor) per key.
SPEED = {"q": (1.1, 1.1), "z": (0.9, 0.9), "w": (1.1, 1.0), "x": (0.9, 1.0), "e": (1.0, 1.1), "c": (1.0, 0.9)}
# key: (field, step)
GAIT_STEPS = {
    "[": ("frequency", -0.25), "]": ("frequency", 0.25),
    "-": ("duty", -0.05), "=": ("duty", 0.05),
    "r": ("pitch", 0.05), "f": ("pitch", -0.05),
    "y": ("height", 0.01), "h": ("height", -0.01),
    "t": ("swing", 0.01), "g": ("swing", -0.01),
    "v": ("width", -0.01), "b": ("width", 0.01),
}
NOMINAL = {"height": 0.0, "frequency": 3.0, "swing": 0.08, "pitch": 0.0, "width": 0.34, "duty": 0.5}
LIMITS = {
    "height": (-0.08, 0.05), "frequency": (0.5, 4.0), "swing": (0.03, 0.20),
    "pitch": (-0.3, 0.3), "width": (0.25, 0.42), "duty": (0.35, 0.75),
}


class KeyboardTeleop(Node):
    def __init__(self, args):
        super().__init__("keyboard_teleop")
        self.speed, self.turn = args.speed, args.turn
        self.direction = (0, 0, 0)
        self.gait = "trot"
        self.fields = dict(NOMINAL)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.gait_pub = self.create_publisher(Float32MultiArray, "/gait_command", 10)
        self.create_timer(0.1, self._publish_velocity)
        self.create_timer(1.0, self._publish_gait)

    def on_key(self, key):
        if key in MOVE:
            self.direction = MOVE[key]
            self._publish_velocity()
        elif key in SPEED:
            lin, ang = SPEED[key]
            self.speed, self.turn = self.speed * lin, self.turn * ang
        elif key in GAIT_KEYS:
            self.gait = GAIT_KEYS[key]
            self._publish_gait()
        elif key in GAIT_STEPS:
            field, step = GAIT_STEPS[key]
            low, high = LIMITS[field]
            self.fields[field] = round(min(high, max(low, self.fields[field] + step)), 3)
            self._publish_gait()
        elif key == "0":
            self.gait, self.fields = "trot", dict(NOMINAL)
            self._publish_gait()
        else:
            return
        self._status()

    def stop(self):
        self.direction = (0, 0, 0)
        self._publish_velocity()

    def _publish_velocity(self):
        x, y, yaw = self.direction
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = x * self.speed, y * self.speed, yaw * self.turn
        self.cmd_pub.publish(msg)

    def _publish_gait(self):
        f = self.fields
        msg = Float32MultiArray()
        msg.data = [f["height"], f["frequency"], *GAITS[self.gait], f["swing"], f["pitch"], f["width"], f["duty"]]
        self.gait_pub.publish(msg)

    def _status(self):
        x, y, yaw = self.direction
        f = self.fields
        line = (
            f"vel ({x * self.speed:+.2f}, {y * self.speed:+.2f}, {yaw * self.turn:+.2f})  "
            f"max {self.speed:.2f} m/s {self.turn:.2f} rad/s | {self.gait} {f['frequency']:.2f} Hz "
            f"duty {f['duty']:.2f} pitch {f['pitch']:+.2f} height {f['height']:+.2f} "
            f"swing {f['swing']:.2f} width {f['width']:.2f}"
        )
        sys.stdout.write("\r" + line.ljust(150))
        sys.stdout.flush()


def read_key(timeout):
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    return sys.stdin.read(1) if ready else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--speed", type=float, default=0.5, help="starting linear speed, m/s")
    ap.add_argument("--turn", type=float, default=1.0, help="starting turn rate, rad/s")
    args, _ = ap.parse_known_args()

    rclpy.init()
    node = KeyboardTeleop(args)
    print(__doc__)
    settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        node._status()
        while rclpy.ok():
            key = read_key(0.05)
            if key == "\x03":
                break
            if key:
                node.on_key(key)
            rclpy.spin_once(node, timeout_sec=0.0)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.stop()
        print()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
