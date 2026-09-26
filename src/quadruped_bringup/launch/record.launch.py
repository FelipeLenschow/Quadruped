import os
from datetime import datetime

from launch import LaunchDescription
from launch.actions import ExecuteProcess, OpaqueFunction

from quadruped_bringup.launch_common import arg, get
from quadruped_core import paths


def setup(context):
    path = get(context, "path") or paths.repo("Mcap", "Recordings", f"run_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return [ExecuteProcess(cmd=["ros2", "bag", "record", "-a", "-s", "mcap", "-o", os.path.abspath(path)],
                           output="screen", sigterm_timeout="10")]


def generate_launch_description():
    return LaunchDescription([
        arg("path", "", "output folder"),
        OpaqueFunction(function=setup),
    ])
