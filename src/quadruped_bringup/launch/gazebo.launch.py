from launch import LaunchDescription
from launch.actions import OpaqueFunction

from quadruped_bringup.launch_common import arg, common_args, flag, get, include, node, record, reward, reward_args


def setup(context):
    checkpoint = get(context, "checkpoint")
    actions = [node(
        "quadruped_drivers", "gazebo_driver",
        f"--robot={get(context, 'robot')}",
        f"--world={get(context, 'world')}",
        f"--obs_dim={get(context, 'obs_dim')}",
        f"--internal_policy={checkpoint}" if checkpoint else "",
        "--use_estimator" if flag(context, "use_estimator") else "",
        main=True,
    )]
    if flag(context, "joy"):
        actions.append(include("joy.launch.py"))
    return actions + record(context, "gazebo") + reward(context)


def generate_launch_description():
    return LaunchDescription([
        *common_args(),
        arg("checkpoint"),
        arg("obs_dim", "49"),
        arg("world", "quadruped_world"),
        arg("use_estimator", "false"),
        arg("joy", "false", "also start the gamepad teleop"),
        *reward_args(),
        OpaqueFunction(function=setup),
    ])
