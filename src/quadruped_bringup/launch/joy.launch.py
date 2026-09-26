import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from quadruped_bringup.launch_common import arg
from quadruped_core import paths


def generate_launch_description():
    teleop = os.path.join(get_package_share_directory("teleop_twist_joy"), "launch", "teleop-launch.py")
    return LaunchDescription([
        arg("config", paths.config("joy_f710.config.yaml")),
        arg("gait", "true", "also run the Walk These Ways gait teleop"),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(teleop),
                                 launch_arguments={"config_filepath": LaunchConfiguration("config")}.items()),
        Node(package="quadruped_operator", executable="gait_teleop", output="screen", emulate_tty=True,
             condition=IfCondition(LaunchConfiguration("gait"))),
    ])
