from launch import LaunchDescription
from launch.actions import OpaqueFunction

from quadruped_bringup.launch_common import arg, common_args, get, node


def setup(context):
    params = {"robot_type": get(context, "robot")}
    for name in ("config_path", "phase"):
        if get(context, name):
            params[name] = get(context, name)
    return [node("quadruped_operator", "reward_estimator", parameters=[params], output="log")]


def generate_launch_description():
    return LaunchDescription([
        *common_args(),
        arg("config_path", "", "training_phases.yaml (default: Walk's)"),
        arg("phase", "", "phase the checkpoint finished under (default: phase1)"),
        OpaqueFunction(function=setup),
    ])
