from launch import LaunchDescription
from launch.actions import OpaqueFunction
from launch_ros.actions import Node

from quadruped_bringup.launch_common import arg, common_args, get, node, record
from quadruped_core import paths


def setup(context):
    # joy_node alone: teleop_twist_joy would be a second /cmd_vel source, and the sweep refuses to
    # run beside one.
    joy = Node(package="joy", executable="joy_node", output="log",
               parameters=[paths.config("joy_f710.config.yaml")])
    sweep = node(
        "quadruped_operator", "sweep_teleop",
        f"--robot={get(context, 'robot')}",
        f"--checkpoint={get(context, 'checkpoint')}",
        *[f"--{k}={get(context, k)}" for k in
          ("walk_s", "ramp_s", "axes", "max_lin_speed", "max_yaw_rate", "walk_m")],
        main=True,
    )
    return [joy, sweep] + record(context, "sweep")


def generate_launch_description():
    return LaunchDescription([
        *common_args(),
        arg("checkpoint", "", "the checkpoint the robot runs, only to file the report under"),
        arg("walk_s", "10"),
        arg("ramp_s", "0"),
        arg("axes", "x,y,yaw"),
        arg("max_lin_speed", "0.5"),
        arg("max_yaw_rate", "1.0"),
        arg("walk_m", "0"),
        OpaqueFunction(function=setup),
    ])
