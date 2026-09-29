import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import OpaqueFunction
from launch_ros.actions import Node

from quadruped_bringup.launch_common import arg, flag
from quadruped_core import paths

NAV2_NODES = ["controller_server", "planner_server", "behavior_server", "bt_navigator"]


def setup(context):
    sim = {"use_sim_time": flag(context, "sim")}
    nav2 = [paths.config("nav2.yaml"), sim]
    to_mux = [("cmd_vel", "/cmd_vel/nav")]
    actions = [
        Node(package="quadruped_perception", executable="lidar_filter", output="screen", parameters=[sim]),
        Node(package="pointcloud_to_laserscan", executable="pointcloud_to_laserscan_node", output="log",
             remappings=[("cloud_in", "/lidar/points_filtered"), ("scan", "/scan")],
             parameters=[sim, {
                 "target_frame": "base_footprint", "transform_tolerance": 0.05,
                 "min_height": 0.10, "max_height": 0.60,
                 "angle_min": -3.14159, "angle_max": 3.14159, "angle_increment": 0.01745,
                 "scan_time": 0.1, "range_min": 0.2, "range_max": 15.0, "use_inf": True,
             }]),
        Node(package="nav2_controller", executable="controller_server", output="log", parameters=nav2,
             remappings=to_mux),
        Node(package="nav2_planner", executable="planner_server", output="log", parameters=nav2),
        Node(package="nav2_behaviors", executable="behavior_server", output="log", parameters=nav2,
             remappings=to_mux),
        Node(package="nav2_bt_navigator", executable="bt_navigator", output="log", parameters=nav2),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager", name="lifecycle_manager_navigation",
             output="screen", parameters=[sim, {"autostart": True, "node_names": NAV2_NODES}]),
    ]
    if flag(context, "slam"):
        actions.append(Node(package="slam_toolbox", executable="async_slam_toolbox_node", output="screen",
                            emulate_tty=True, parameters=[paths.config("slam.yaml"), sim]))
    if flag(context, "rviz"):
        view = os.path.join(get_package_share_directory("nav2_bringup"), "rviz", "nav2_default_view.rviz")
        actions.append(Node(package="rviz2", executable="rviz2", arguments=["-d", view], output="log",
                            parameters=[sim]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        arg("sim", "true", "use /clock (Gazebo); false on the real robot"),
        arg("slam", "true", "build the map with slam_toolbox"),
        arg("rviz", "true"),
        OpaqueFunction(function=setup),
    ])
