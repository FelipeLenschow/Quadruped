import os
from datetime import datetime

from launch import LaunchDescription
from launch.actions import ExecuteProcess, OpaqueFunction

from quadruped_bringup.launch_common import arg, get
from quadruped_common import paths

LIDAR_TOPICS = r"lidar/points.*|lio/(cloud.*|map)"
CAMERA_TOPICS = r"(go2_camera|camera)/.*/compressed"
# Never: the decoded frames of the compressed streams. camera_bridge makes them only for a subscriber, so `-a` had
# it decode every 1920 x 1080 JPEG and send 30 MB/s (6.2 MB a frame) to the recorder, which then lost 9 in 10
# /lidar/imu messages on the robot.
RAW_IMAGES = r".*/image_raw|gazebo/.*image"


def on(context, name):
    return get(context, name).strip().lower() in ("1", "true", "yes", "on")


def setup(context):
    path = get(context, "path") or paths.repo("Mcap", "Recordings", f"run_{datetime.now():%Y%m%d_%H%M%S}")
    # ros2 bag stops if the folder exists (the launcher's "repeat" gives the last run's name again): number it.
    base, n = path.rstrip("/"), 1
    while os.path.exists(path):
        n += 1
        path = f"{base}_{n}"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    skip = [RAW_IMAGES] + ([] if on(context, "lidar") else [LIDAR_TOPICS]) + ([] if on(context, "camera") else [CAMERA_TOPICS])
    exclude = ["-x", "^/(" + "|".join(skip) + ")$"]
    return [ExecuteProcess(cmd=["ros2", "bag", "record", "-a", *exclude, "-s", "mcap", "-o", os.path.abspath(path)],
                           output="screen", sigterm_timeout="10")]


def generate_launch_description():
    return LaunchDescription([
        arg("path", "", "output folder"),
        arg("lidar", "true", "also record the lidar and LIO point clouds"),
        arg("camera", "true", "also record the compressed camera streams (raw images never)"),
        OpaqueFunction(function=setup),
    ])
