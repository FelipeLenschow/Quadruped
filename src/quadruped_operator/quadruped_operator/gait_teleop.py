"""
Gait teleop - Walk These Ways gait commands from the gamepad (F710 in X mode), beside teleop_twist_joy.

    Right stick up / down   body pitch, +-max_pitch (stick up = nose down)
    D-pad left / right      previous / next gait: walk, trot, pace, bound, pronk
    D-pad up / down         step frequency +- freq_step Hz
    LT / RT                 duty (stance share) -/+ duty_step

Publishes the 9 gait commands [height, frequency, phase, offset, bound, swing, pitch, width, duty] on
/gait_command, which LocomotionPipeline hands to a WTW policy (PolicyRunner clamps them to the ranges
the run trained on, snaps the gait to one it trained, and drops duty for a run without it). The right
stick's other axis is yaw in joy_f710.config.yaml; its up/down axis, the D-pad and the triggers are not
used by teleop_twist_joy or by the console's e-stop buttons.
"""

import argparse

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray

GAITS = {
    "trot": (0.5, 0.0, 0.0),
    "pace": (0.0, 0.0, 0.5),
    "bound": (0.0, 0.5, 0.0),
    "pronk": (0.0, 0.0, 0.0),
    "walk": (0.0, 0.75, 0.5),
}
GAIT_ORDER = ["walk", "trot", "pace", "bound", "pronk"]

# joy convention: left and up are positive, on the sticks and the D-pad alike. The triggers rest at
# +1 and read -1 fully pressed.
AXIS_LT, AXIS_RIGHT_Y, AXIS_RT, AXIS_DPAD_X, AXIS_DPAD_Y = 2, 4, 5, 6, 7


class GaitTeleop(Node):
    def __init__(self, args):
        super().__init__("gait_teleop")
        self.args = args
        self.gait_idx = GAIT_ORDER.index("trot")
        self.frequency = args.frequency
        self.duty = args.duty
        self.pitch = 0.0
        self.prev_axes = None
        self.pub = self.create_publisher(Float32MultiArray, "/gait_command", 10)
        self.create_subscription(Joy, "/joy", self._joy_cb, 10)
        self.create_timer(1.0, self._publish)
        self._publish()
        self._status()

    def _joy_cb(self, msg: Joy):
        axes = list(msg.axes)
        pitch = axes[AXIS_RIGHT_Y] * self.args.max_pitch if AXIS_RIGHT_Y < len(axes) else 0.0
        if abs(pitch) < self.args.deadzone * self.args.max_pitch:
            pitch = 0.0
        self.pitch = -pitch if self.args.invert_pitch else pitch
        dx = self._dpad_edge(axes, AXIS_DPAD_X)
        dy = self._dpad_edge(axes, AXIS_DPAD_Y)
        dd = self._trigger_edge(axes, AXIS_RT) - self._trigger_edge(axes, AXIS_LT)
        self.prev_axes = axes
        if dx:
            self.gait_idx = (self.gait_idx - dx) % len(GAIT_ORDER)
        if dy:
            low, high = self.args.freq_range
            self.frequency = min(high, max(low, self.frequency + dy * self.args.freq_step))
        if dd:
            low, high = self.args.duty_range
            self.duty = min(high, max(low, self.duty + dd * self.args.duty_step))
        self._publish()
        if dx or dy or dd:
            self._status()

    def _dpad_edge(self, axes, i):
        cur = axes[i] if i < len(axes) else 0.0
        old = self.prev_axes[i] if self.prev_axes is not None and i < len(self.prev_axes) else 0.0
        if abs(cur) > 0.5 and abs(old) <= 0.5:
            return 1 if cur > 0 else -1
        return 0

    def _trigger_edge(self, axes, i):
        cur = axes[i] if i < len(axes) else 1.0
        old = self.prev_axes[i] if self.prev_axes is not None and i < len(self.prev_axes) else 1.0
        return 1 if cur < -0.5 and old >= -0.5 else 0

    def _publish(self):
        a = self.args
        msg = Float32MultiArray()
        msg.data = [a.height, self.frequency, *GAITS[GAIT_ORDER[self.gait_idx]], a.swing, self.pitch, a.width,
                    self.duty]
        self.pub.publish(msg)

    def _status(self):
        self.get_logger().info(
            f"gait {GAIT_ORDER[self.gait_idx]}, {self.frequency:.2f} Hz, duty {self.duty:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--max_pitch", type=float, default=0.3, help="rad at full stick (trained: +-0.3)")
    ap.add_argument("--invert_pitch", action="store_true", help="stick up = nose up instead")
    ap.add_argument("--deadzone", type=float, default=0.1, help="fraction of the stick ignored around centre")
    ap.add_argument("--frequency", type=float, default=3.0, help="starting step frequency, Hz")
    ap.add_argument("--freq_step", type=float, default=0.25)
    ap.add_argument("--freq_range", type=float, nargs=2, default=(0.5, 4.0),
                    help="the policy runner clamps to the run's trained range as well")
    ap.add_argument("--duty", type=float, default=0.5, help="starting stance share")
    ap.add_argument("--duty_step", type=float, default=0.05)
    ap.add_argument("--duty_range", type=float, nargs=2, default=(0.35, 0.75))
    ap.add_argument("--height", type=float, default=0.0)
    ap.add_argument("--swing", type=float, default=0.08)
    ap.add_argument("--width", type=float, default=0.34)
    args, _ = ap.parse_known_args()

    rclpy.init()
    node = GaitTeleop(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
