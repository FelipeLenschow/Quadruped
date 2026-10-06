from launch import LaunchDescription
from launch.actions import OpaqueFunction

from quadruped_core import paths
from quadruped_bringup.launch_common import arg, common_args, flag, get, include, interface, interface_args, node, record, reward, reward_args

LIO_TOPICS = [("/aft_mapped_to_init", "/odom"), ("/cloud_registered", "/cloud"), ("/Laser_map", "/map"),
              ("/path", "/path"), ("/cloud_registered_body", "/cloud_body"), ("/cloud_effected", "/cloud_effected")]


def setup(context):
    checkpoint = get(context, "checkpoint")
    net_interface = get(context, "interface")
    actions = [node(
        "quadruped_drivers", "real_driver",
        f"--robot={get(context, 'robot')}",
        f"--obs_dim={get(context, 'obs_dim')}",
        f"--interface={net_interface}" if net_interface else "",
        f"--internal_policy={checkpoint}" if checkpoint else "",
        main=True,
    )]
    if flag(context, "sensors"):
        actions.append(node("quadruped_drivers", "real_sensors", f"--interface={net_interface}" if net_interface else ""))
    if flag(context, "lio"):
        actions.append(node("point_lio", "pointlio_mapping", output="log",
                            parameters=[paths.config("point_lio_go2.yaml"),
                                        {"pcd_save.pcd_save_en": flag(context, "lio_save")}],
                            remappings=[(t, "/lio" + n) for t, n in LIO_TOPICS]))
    if flag(context, "monitor"):
        actions.append(node("quadruped_drivers", "system_monitor"))
    actions += interface(context)
    if flag(context, "joy"):
        actions.append(include("joy.launch.py"))
    return actions + record(context, "real_deploy" if checkpoint else "real_telemetry") + reward(context)


def generate_launch_description():
    return LaunchDescription([
        *common_args(),
        *interface_args(),
        arg("checkpoint", "", "policy .pt; empty runs telemetry only"),
        arg("obs_dim", "45"),
        arg("interface", "", "network interface for the Unitree SDK"),
        arg("joy", "false", "also start the gamepad teleop"),
        arg("sensors", "true", "lidar cloud to /lidar/points and the sensor TF"),
        arg("lio", "false", "Point-LIO 3D odometry and map on /lio/* (frames lio_odom -> lio_imu)"),
        arg("lio_save", "false", "Point-LIO writes its 3D map to PCD/scans.pcd in the point_lio package on exit"),
        arg("monitor", "true", "CPU/GPU/RAM/temperatures to /system_stats"),
        *reward_args(),
        OpaqueFunction(function=setup),
    ])
