"""3D mapping: Point-LIO on the lidar and its IMU, and the voxel map for the app (lio_map_stream).

Needs the robot running (real.launch.py, whose real_sensors publishes /lidar/points and /lidar/imu).
Outputs /lio/odom, /lio/cloud in lio_odom -> lio_imu, and /lio/map_voxels(+/delta).
"""

from launch import LaunchDescription
from launch.actions import OpaqueFunction

from quadruped_bringup.launch_common import arg, flag, get, node
from quadruped_core import paths

LIO_TOPICS = [("/aft_mapped_to_init", "/odom"), ("/cloud_registered", "/cloud"), ("/Laser_map", "/map"),
              ("/path", "/path"), ("/cloud_registered_body", "/cloud_body"), ("/cloud_effected", "/cloud_effected")]


def setup(context):
    return [
        node("point_lio", "pointlio_mapping", main=True,
             parameters=[paths.config("point_lio_go2.yaml"), {"pcd_save.pcd_save_en": flag(context, "save")}],
             remappings=[(t, "/lio" + n) for t, n in LIO_TOPICS]),
        node("quadruped_perception", "lio_map_stream", f"--voxel={get(context, 'voxel')}"),
    ]


def generate_launch_description():
    return LaunchDescription([
        arg("save", "false", "write the 3D map to PCD/scans.pcd in the point_lio package on exit"),
        arg("voxel", "0.05", "the app's voxel size in m"),
        OpaqueFunction(function=setup),
    ])
