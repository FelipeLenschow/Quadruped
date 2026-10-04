"""Robot computer load to ROS 2: CPU per core, RAM, GPU and temperatures as JSON on /system_stats.

Reads /proc and /sys directly, so inside the robot's (privileged) container it reports the
whole Jetson, not just the container.

    {"cpu": 45.2, "cores": [40.1, 52.3, ...], "ram_used_mb": 3100, "ram_total_mb": 7400,
     "gpu": 3.0, "temps": {"cpu-thermal": 52.1, ...}, "temp_max": 55.0, "load": [1.2, 1.0, 0.9]}

gpu is null when no GPU load file is found (pass --gpu_load to point at it).
"""

import argparse
import glob
import json
import os
import re
import sys

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from std_msgs.msg import String

# Jetson GPU load, in per-mille. ga10b is the Orin's GPU, gv11b the Xavier's, gp10b the TX2's.
GPU_LOAD_GLOBS = [
    "/sys/devices/platform/gpu.0/load",
    "/sys/devices/gpu.0/load",
    "/sys/devices/platform/*.ga10b/load",
    "/sys/devices/platform/*.gpu/load",
    "/sys/devices/platform/bus@0/*.gpu/load",
]
GPU_NAME = re.compile(r"gpu|ga10b|gv11b|gp10b", re.I)


def read(path):
    with open(path) as f:
        return f.read()


def parse_proc_stat(text):
    """[(busy, total)] jiffies: the 'cpu' line first, then cpu0, cpu1, ..."""
    out = []
    for line in text.splitlines():
        if not re.match(r"cpu\d*\s", line):
            continue
        v = [int(x) for x in line.split()[1:]]
        idle = v[3] + (v[4] if len(v) > 4 else 0)  # idle + iowait
        total = sum(v[:8])  # guest time is already inside user/nice
        out.append((total - idle, total))
    return out


def cpu_percent(prev, cur):
    return [
        round(100.0 * (b1 - b0) / (t1 - t0), 1) if t1 > t0 else 0.0
        for (b0, t0), (b1, t1) in zip(prev, cur)
    ]


def parse_meminfo(text):
    """(used_mb, total_mb), used meaning not available to new programs."""
    kb = {}
    for line in text.splitlines():
        k, _, rest = line.partition(":")
        kb[k] = int(rest.split()[0]) if rest.split() else 0
    total = kb.get("MemTotal", 0)
    avail = kb.get("MemAvailable", kb.get("MemFree", 0))
    return (total - avail) // 1024, total // 1024


def find_gpu_load():
    def valid(path):
        try:
            int(read(path).strip())
            return True
        except (OSError, ValueError):
            return False

    for pattern in GPU_LOAD_GLOBS:
        for path in sorted(glob.glob(pattern)):
            if valid(path):
                return path
    # Not where expected: walk /sys/devices a few levels down. No ** glob and no
    # following links, since sysfs symlinks loop.
    root_depth = "/sys/devices".count("/")
    for dirpath, dirnames, filenames in os.walk("/sys/devices", followlinks=False):
        if dirpath.count("/") - root_depth >= 4:
            dirnames[:] = []
        if "load" in filenames and GPU_NAME.search(os.path.basename(dirpath)):
            path = os.path.join(dirpath, "load")
            if valid(path):
                return path
    return None


def read_temps():
    temps = {}
    for zone in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        try:
            name = read(os.path.join(zone, "type")).strip()
            milli = int(read(os.path.join(zone, "temp")).strip())
        except (OSError, ValueError):
            continue
        # Disabled zones report nonsense (-40 C, or 100+ C on some boards' unused sensors).
        if -30000 < milli < 125000:
            temps[name] = round(milli / 1000.0, 1)
    return temps


class SystemMonitor(Node):
    def __init__(self, args):
        super().__init__("system_monitor")
        self.pub = self.create_publisher(String, "/system_stats", 10)
        self.gpu_path = args.gpu_load or find_gpu_load()
        self.get_logger().info(f"GPU load: {self.gpu_path or 'not found'}")
        self.prev = parse_proc_stat(read("/proc/stat"))
        self.create_timer(args.period, self.tick)

    def tick(self):
        cur = parse_proc_stat(read("/proc/stat"))
        pct = cpu_percent(self.prev, cur)
        self.prev = cur
        used, total = parse_meminfo(read("/proc/meminfo"))
        gpu = None
        if self.gpu_path:
            try:
                gpu = round(int(read(self.gpu_path).strip()) / 10.0, 1)
            except (OSError, ValueError):
                pass
        temps = read_temps()
        stats = {
            "cpu": pct[0] if pct else 0.0,
            "cores": pct[1:],
            "ram_used_mb": used,
            "ram_total_mb": total,
            "gpu": gpu,
            "temps": temps,
            "temp_max": max(temps.values()) if temps else None,
            "load": [round(x, 2) for x in os.getloadavg()],
        }
        self.pub.publish(String(data=json.dumps(stats)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--period", type=float, default=1.0)
    parser.add_argument("--gpu_load", default="", help="GPU load file (per-mille); found automatically if empty")
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])
    rclpy.init()
    node = SystemMonitor(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
