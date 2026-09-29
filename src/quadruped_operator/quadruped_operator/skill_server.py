"""Skill server - named robot behaviours on the /skill service."""

import argparse
import math
import signal
import json
import threading
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from std_msgs.msg import Float32MultiArray, String
from std_srvs.srv import Empty

from quadruped_interfaces.srv import Skill

POSES = {"stand": "stand", "sit": "sit", "lie": "lie_flat"}
GAITS = {
    "trot": (0.5, 0.0, 0.0),
    "pace": (0.0, 0.0, 0.5),
    "bound": (0.0, 0.5, 0.0),
    "pronk": (0.0, 0.0, 0.0),
    "walk": (0.0, 0.75, 0.5),
}
GAIT_FIELDS = ["height", "frequency", "phase", "offset", "bound", "swing", "pitch", "width", "duty"]


class SkillServer(Node):
    def __init__(self, args):
        super().__init__("skill_server")
        self.args = args
        self.state = None
        self.state_time = 0.0
        self.busy = threading.Lock()
        group = ReentrantCallbackGroup()
        self.create_subscription(String, "/robot_state", self._state_cb, 10, callback_group=group)
        self.mode_pub = self.create_publisher(String, "/pipeline/mode", 10)
        self.pose_pub = self.create_publisher(String, "/pose/command", 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel/skill", 10)
        self.gait_pub = self.create_publisher(Float32MultiArray, "/gait_command", 10)
        self.override_pub = self.create_publisher(Float32MultiArray, "/gait_command/override", 10)
        self.amcl_pose = None
        self.amcl_time = 0.0
        self.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", self._amcl_cb, 10, callback_group=group)
        self.global_localization = self.create_client(Empty, "/reinitialize_global_localization",
                                                      callback_group=group)
        self.create_service(Skill, "/skill", self._skill_cb, callback_group=group)
        self.get_logger().info("ready: /skill")

    def _state_cb(self, msg):
        self.state = json.loads(msg.data)
        self.state_time = time.monotonic()

    def _amcl_cb(self, msg):
        self.amcl_pose = msg.pose
        self.amcl_time = time.monotonic()

    def _fresh(self):
        if self.state is None or time.monotonic() - self.state_time > 1.0:
            return None
        return self.state

    def _wait(self, cond, timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            s = self._fresh()
            if s is not None and cond(s):
                return s
            time.sleep(0.05)
        return None

    def _skill_cb(self, req, res):
        name = req.name.strip().lower()
        if not self.busy.acquire(blocking=False):
            res.success, res.message = False, "busy with another skill"
            return res
        try:
            res.success, res.message = self._run(name, float(req.duration))
        except Exception as e:
            res.success, res.message = False, f"error: {e}"
        finally:
            self.busy.release()
        self.get_logger().info(f"{name}: {'ok' if res.success else 'failed'} - {res.message}")
        return res

    def _run(self, name, duration):
        s = self._fresh()
        if s is None:
            return False, "no /robot_state: is the driver running?"
        if s["estop"]:
            return False, "e-stop is latched"
        if name == "stop":
            return self._stop(duration or 1.0)
        if name in POSES:
            return self._pose(POSES[name])
        if name == "walk":
            return self._walk()
        if name == "sniff":
            return self._sniff(duration or self.args.sniff_time)
        if name == "localize":
            return self._localize(duration or self.args.localize_time)
        if name.startswith("gait_") and name[5:] in GAITS:
            return self._gait(name[5:])
        return False, f"unknown skill '{name}'"

    def _localize(self, seconds):
        """Spread AMCL's particles over the whole map, then turn in place until they agree."""
        if self._fresh()["mode"] != "policy":
            return False, "localize needs policy mode; call walk first"
        if not self.global_localization.wait_for_service(timeout_sec=2.0):
            return False, "AMCL is not running: start navigation with a saved map"
        start = time.monotonic()
        future = self.global_localization.call_async(Empty.Request())
        while not future.done() and time.monotonic() - start < 3.0:
            time.sleep(0.05)
        turn = Twist()
        turn.angular.z = self.args.localize_turn_rate
        spread = None
        while time.monotonic() - start < seconds:
            self.cmd_pub.publish(turn)
            time.sleep(0.05)
            if self.amcl_pose is None or self.amcl_time < start + 1.0:
                continue
            cov = self.amcl_pose.covariance
            spread = (math.sqrt(max(cov[0], cov[7])), math.sqrt(cov[35]))
            if spread[0] < self.args.localize_xy and spread[1] < self.args.localize_yaw:
                self._hold_still(0.5)
                p = self.amcl_pose.pose
                yaw = 2.0 * math.atan2(p.orientation.z, p.orientation.w)
                return True, (f"localized at x={p.position.x:.2f} y={p.position.y:.2f} yaw={yaw:.2f} "
                              f"(spread {spread[0]:.2f} m, {spread[1]:.2f} rad)")
        self._hold_still(0.5)
        detail = f", spread {spread[0]:.2f} m {spread[1]:.2f} rad" if spread else ", no AMCL update"
        return False, f"not localized after {seconds:.0f} s{detail}"

    def _hold_still(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.cmd_pub.publish(Twist())
            time.sleep(0.05)

    def _stop(self, seconds):
        self._hold_still(seconds)
        return True, f"stopped, {self._fresh()['posture']}"

    def _pose(self, pose):
        s = self._fresh()
        if s["posture"] == "fallen" and pose != "lie_flat":
            return False, "robot has fallen; lie first"
        if s["mode"] == "policy":
            self._hold_still(self.args.settle_time)
            self.mode_pub.publish(String(data="pose"))
            if self._wait(lambda s: s["mode"] == "pose", 1.0) is None:
                return False, "pipeline did not switch to pose mode"
        if not self._fresh()["torque_on"]:
            return False, "torque is off (no console/supervisor heartbeat?)"
        for _ in range(2):
            self.pose_pub.publish(String(data=pose))
            if self._wait(lambda s: s["pose"] == pose and not s["pose_done"], 1.0) is not None:
                break
        else:
            return False, f"pose generator did not start '{pose}'"
        done = self._wait(lambda s: s["pose_done"] and not s["transition"], self.args.pose_timeout)
        if done is None:
            return False, f"'{pose}' not reached in {self.args.pose_timeout:.0f} s"
        return True, f"{pose} reached"

    def _walk(self):
        s = self._fresh()
        if s["mode"] == "policy":
            return True, "already in policy mode"
        if s["posture"] == "fallen":
            return False, "robot has fallen; lie, then stand"
        if not (s["pose"] == "stand" and s["pose_done"]):
            ok, msg = self._pose("stand")
            if not ok:
                return ok, msg
        self.mode_pub.publish(String(data="policy"))
        if self._wait(lambda s: s["mode"] == "policy" and not s["transition"], 6.0) is None:
            s = self._fresh()
            why = "e-stop" if s and s["estop"] else "refused or still blending"
            return False, f"policy mode not active ({why})"
        return True, "policy mode active, ready to move"

    def _gait_vector(self, gait, **fields):
        vec = [gait[f] for f in GAIT_FIELDS if f in gait]
        for key, value in fields.items():
            vec[GAIT_FIELDS.index(key)] = value
        return vec

    def _sniff(self, seconds):
        s = self._fresh()
        if s["mode"] != "policy":
            return False, "sniff needs policy mode; call walk first"
        if not s["gait"]:
            return False, "the running policy has no gait input (not a WTW policy)"
        msg = Float32MultiArray(data=self._gait_vector(s["gait"], height=self.args.sniff_height,
                                                       pitch=self.args.sniff_pitch))
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.cmd_pub.publish(Twist())
            self.override_pub.publish(msg)
            time.sleep(0.05)
            if self._fresh()["safety_blocked"]:
                return False, "safety tripped during the sniff"
        return True, f"sniffed for {seconds:.1f} s"

    def _gait(self, name):
        s = self._fresh()
        if not s["gait"]:
            return False, "the running policy has no gait input (not a WTW policy)"
        phase, offset, bound = GAITS[name]
        vec = self._gait_vector(s["gait"], phase=phase, offset=offset, bound=bound)
        self.gait_pub.publish(Float32MultiArray(data=vec))
        if self._wait(lambda s: s["gait"] and s["gait"]["name"] == name, 1.0) is None:
            return False, f"policy kept {self._fresh()['gait']['name']}; '{name}' was not trained"
        return True, f"gait {name}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--settle_time", type=float, default=1.0, help="s of zero velocity before leaving policy mode")
    ap.add_argument("--pose_timeout", type=float, default=10.0)
    ap.add_argument("--sniff_time", type=float, default=3.0)
    ap.add_argument("--localize_time", type=float, default=30.0, help="s of turning before giving up")
    ap.add_argument("--localize_turn_rate", type=float, default=0.5, help="rad/s")
    ap.add_argument("--localize_xy", type=float, default=0.15, help="m: particle spread that counts as found")
    ap.add_argument("--localize_yaw", type=float, default=0.15, help="rad")
    ap.add_argument("--sniff_height", type=float, default=-0.04,
                    help="WTW body height offset; -0.05 already trips the calf ROM margin in Gazebo")
    ap.add_argument("--sniff_pitch", type=float, default=0.2, help="WTW body pitch, + is nose down")
    args = ap.parse_args(remove_ros_args()[1:])

    rclpy.init()
    node = SkillServer(args)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
