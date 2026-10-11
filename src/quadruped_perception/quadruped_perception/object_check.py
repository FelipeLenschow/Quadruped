"""Gazebo check for object_mapper: /objects against the true poses of a world's models (default the nav world's
traffic cones). Prints, for each one, the nearest mapped object, its label and the distance, then any mapped
objects far from all of them. Gazebo's odom starts at the robot's spawn pose, so pass it with --start when the
robot does not spawn at the world's origin.
"""

import argparse
import json
import os
import re
import signal

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.utilities import remove_ros_args
from std_msgs.msg import String


def model_poses(world, uri):
    """(name, x, y) of each <include> of uri in the world file."""
    text = open(world).read()
    out = []
    for inc in re.findall(r"<include>(.*?)</include>", text, re.S):
        if f"model://{uri}" not in inc:
            continue
        name = re.search(r"<name>(.*?)</name>", inc)
        x, y = (float(v) for v in re.search(r"<pose>(.*?)</pose>", inc).group(1).split()[:2])
        out.append((name.group(1) if name else uri, x, y))
    return out


class ObjectCheck(Node):
    def __init__(self, truth, match_m):
        super().__init__("object_check")
        self.truth, self.match_m = truth, match_m
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/objects", self._cb, latched)

    def _cb(self, m):
        objects = json.loads(m.data)["objects"]
        pos = np.array([[o["x"], o["y"]] for o in objects]).reshape(-1, 2)
        used = set()
        lines, errors = [], []
        for name, x, y in self.truth:
            d = np.hypot(*(pos - (x, y)).T) if len(pos) else np.array([])
            if len(d) and d.min() < self.match_m:
                i = int(d.argmin())
                used.add(i)
                errors.append(d[i])
                o = objects[i]
                lines.append(f"  {name:10s} ({x:5.2f}, {y:5.2f}): {o['label']} #{o['id']} at ({o['x']:5.2f}, "
                             f"{o['y']:5.2f}), error {d[i] * 100:3.0f} cm, {o['sightings']} sightings")
            else:
                lines.append(f"  {name:10s} ({x:5.2f}, {y:5.2f}): not mapped")
        others = [o for i, o in enumerate(objects) if i not in used]
        mean = f"mean error {np.mean(errors) * 100:.0f} cm" if errors else "none mapped"
        print(f"{len(errors)}/{len(self.truth)} found, {mean}\n" + "\n".join(lines) +
              (f"\n  other objects: " + ", ".join(f"{o['label']} ({o['x']:.1f}, {o['y']:.1f})" for o in others)
               if others else ""), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", default="", help="world .sdf (default quadruped_description's gazebo/nav.sdf)")
    ap.add_argument("--model", default="traffic_cone", help="model uri to check (model://<this>)")
    ap.add_argument("--start", default="0,0,0", help="robot spawn x,y,yaw in the world (odom's origin)")
    ap.add_argument("--match", type=float, default=1.0, help="m; farther than this counts as not mapped")
    args = ap.parse_args(remove_ros_args()[1:])

    world = args.world or os.path.join(get_package_share_directory("quadruped_description"), "gazebo", "nav.sdf")
    sx, sy, syaw = (float(v) for v in args.start.split(","))
    c, s = np.cos(syaw), np.sin(syaw)
    truth = [(n, c * (x - sx) + s * (y - sy), -s * (x - sx) + c * (y - sy))
             for n, x, y in model_poses(world, args.model)]

    rclpy.init()
    node = ObjectCheck(truth, args.match)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
