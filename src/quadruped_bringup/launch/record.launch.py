import os
from datetime import datetime

from launch import LaunchDescription
from launch.actions import ExecuteProcess, OpaqueFunction

from quadruped_bringup.launch_common import arg, get
from quadruped_core import paths

LIDAR_TOPICS = r"^/(lidar/points.*|lio/(cloud.*|map)|camera/image_raw)$"


def setup(context):
    path = get(context, "path") or paths.repo("Mcap", "Recordings", f"run_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    exclude = [] if get(context, "lidar").strip().lower() in ("1", "true", "yes", "on") else ["-x", LIDAR_TOPICS]
    return [ExecuteProcess(cmd=["ros2", "bag", "record", "-a", *exclude, "-s", "mcap", "-o", os.path.abspath(path)],
                           output="screen", sigterm_timeout="10")]


def generate_launch_description():
    return LaunchDescription([
        arg("path", "", "output folder"),
        arg("lidar", "true", "also record the lidar and LIO point clouds and the camera image"),
        OpaqueFunction(function=setup),
    ])
