"""3D mapping on the Helios 16P: helios_driver (the lidar to /helios/points), FAST-LIO, and the app's voxel map.

Needs /lidar/imu (real_sensors: real.launch.py, or dls2.launch.py with sensors). FAST-LIO's outputs go to
/helios_lio/*: /helios_lio/odom (into DLS2's sensor_fusion as helios_lio, through ros2_translator),
/helios_lio/cloud, /helios_lio/path, in camera_init -> body (body: the base). The extrinsic is config.yaml's
helios: xyz, rpy, as for helios_driver's TF.

Replay an old bag (rslidar_sdk's /rslidar_points, the IMU on /imu/reliable):
    ros2 launch quadruped_bringup fast_lio.launch.py from_topic:=/rslidar_points imu_topic:=/imu/reliable
"""
import math

from launch import LaunchDescription
from launch.actions import OpaqueFunction

from quadruped_bringup.launch_common import arg, flag, get, node
from quadruped_common import paths
from quadruped_common.config_loader import load_config

LIO_TOPICS = [("/Odometry", "/odom"), ("/cloud_registered", "/cloud"), ("/cloud_registered_body", "/cloud_body"),
              ("/cloud_effected", "/cloud_effected"), ("/Laser_map", "/map"), ("/path", "/path")]


def rotation(rpy):
    """Rz(yaw) Ry(pitch) Rx(roll), row major, as quadruped_real's sensor_mounts.hpp."""
    cr, sr = math.cos(rpy[0]), math.sin(rpy[0])
    cp, sp = math.cos(rpy[1]), math.sin(rpy[1])
    cy, sy = math.cos(rpy[2]), math.sin(rpy[2])
    return [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
            sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
            -sp, cp * sr, cp * cr]


def setup(context):
    helios = load_config().get("helios", {})
    xyz = [float(v) for v in helios.get("xyz", [0.0, 0.0, 0.0])]
    rpy = [float(v) for v in helios.get("rpy", [0.0, 0.0, 0.0])]

    actions = []
    if flag(context, "driver"):
        from_topic = get(context, "from_topic")
        actions.append(node("quadruped_real", "helios_driver", f"--from_topic={from_topic}" if from_topic else ""))
    actions.append(node(
        "fast_lio", "fastlio_mapping", main=True,
        parameters=[paths.config("fast_lio_helios.yaml"), {
            "common.imu_topic": get(context, "imu_topic"),
            "mapping.extrinsic_T": xyz,
            "mapping.extrinsic_R": rotation(rpy),
            "pcd_save.pcd_save_en": flag(context, "save"),
        }],
        remappings=[(t, "/helios_lio" + n) for t, n in LIO_TOPICS]))
    if flag(context, "map_stream"):
        # The app's voxel map, from this LIO instead of Point-LIO (run one of the two)
        actions.append(node("quadruped_perception", "lio_map_stream", f"--voxel={get(context, 'voxel')}",
                            remappings=[("/lio/cloud", "/helios_lio/cloud"), ("/lio/odom", "/helios_lio/odom")]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        arg("driver", "true", "start helios_driver (false: /helios/points comes from elsewhere)"),
        arg("from_topic", "", "helios_driver reads this recorded rslidar_sdk cloud instead of the lidar"),
        arg("imu_topic", "/lidar/imu", "FAST-LIO's IMU (base axes and origin)"),
        arg("save", "false", "write the 3D map to PCD/scans.pcd in the fast_lio package on exit"),
        arg("map_stream", "true", "lio_map_stream: the app's voxel map from this LIO"),
        arg("voxel", "0.05", "the app's voxel size in m"),
        OpaqueFunction(function=setup),
    ])
