"""Cameras: camera_bridge for config.yaml's cameras, and optionally the lidar mount check, object detection and
objects on the map (object_mapper, which needs nav.launch.py's map and filtered lidar).

Needs the robot running: real.launch.py (or, with sim:=true, Gazebo's nav world, whose gazebo_sensors
publishes the simulated image). The Go2's own camera (source: unitree) comes through quadruped_real's go2_camera,
started here.
"""

from launch import LaunchDescription
from launch.actions import OpaqueFunction

from quadruped_bringup.launch_common import arg, flag, get, node
from quadruped_common.config_loader import load_config


def setup(context):
    only = get(context, "only")
    net_interface = get(context, "interface")
    actions = [node(
        "quadruped_drivers", "camera_bridge",
        "--sim" if flag(context, "sim") else "",
        f"--only={only}" if only else "",
        main=True,
    )]
    cams = load_config().get("cameras") or {}
    wanted = [c for name, c in cams.items() if not only or name in only.split(",")]
    if not flag(context, "sim") and any(c.get("source", "unitree") == "unitree" for c in wanted):
        actions.append(node("quadruped_real", "go2_camera", f"--interface={net_interface}" if net_interface else ""))
    if flag(context, "check"):
        actions.append(node("quadruped_perception", "camera_lidar_check", f"--camera={get(context, 'check_camera')}"))
    if flag(context, "detect"):
        actions.append(node("quadruped_perception", "object_detector"))
    if flag(context, "map_objects"):
        actions.append(node("quadruped_perception", "object_mapper",
                            parameters=[{"use_sim_time": flag(context, "sim")}]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        arg("sim", "false", "Gazebo: read each camera's sim_topic"),
        arg("only", "", "comma-separated camera names (default all)"),
        arg("interface", "", "network interface for the Unitree SDK"),
        arg("check", "false", "also draw the lidar on a camera's image (camera_lidar_check)"),
        arg("check_camera", "front", "camera for the check"),
        arg("detect", "false", "also run object_detector (config.yaml's detector:)"),
        arg("map_objects", "false", "also run object_mapper (config.yaml's object_mapper:; needs nav.launch.py)"),
        OpaqueFunction(function=setup),
    ])
