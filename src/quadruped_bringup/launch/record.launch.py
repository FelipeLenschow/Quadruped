import os
from datetime import datetime

from launch import LaunchDescription
from launch.actions import ExecuteProcess, OpaqueFunction

from quadruped_bringup.launch_common import arg, get
from quadruped_common import paths

# The large ones, left out unless lidar:=true: the raw lidar clouds, the LIOs' registered clouds and maps, images
LIDAR_TOPICS = (r"^/(lidar/points.*|helios/points|lio/(cloud.*|map)|helios_lio/(cloud.*|map)|camera/.*image_raw.*"
                r"|gazebo/.*image)$")
# lidar_merge's cloud, left out unless merged:=true: replay rebuilds it from /helios/points and /lidar/points
MERGED_TOPICS = r"^/lidars/points$"


def setup(context):
    path = get(context, "path") or paths.repo("Mcap", "Recordings", f"run_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    on = lambda name: get(context, name).strip().lower() in ("1", "true", "yes", "on")
    left_out = ([] if on("lidar") else [LIDAR_TOPICS]) + ([] if on("merged") else [MERGED_TOPICS])
    exclude = ["-x", "|".join(left_out)] if left_out else []
    return [ExecuteProcess(cmd=["ros2", "bag", "record", "-a", *exclude, "-s", "mcap", "-o", os.path.abspath(path)],
                           output="screen", sigterm_timeout="10")]


def generate_launch_description():
    return LaunchDescription([
        arg("path", "", "output folder"),
        arg("lidar", "true", "also record the lidar and LIO point clouds and the camera images"),
        arg("merged", "false", "also record lidar_merge's /lidars/points"),
        OpaqueFunction(function=setup),
    ])
