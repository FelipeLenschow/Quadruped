from launch import LaunchDescription
from launch.actions import OpaqueFunction

from quadruped_bringup.launch_common import arg, common_args, flag, get, include, node, record, reward, reward_args


def setup(context):
    checkpoint = get(context, "checkpoint")
    actions = [node(
        "quadruped_drivers", "mujoco_driver",
        f"--robot={get(context, 'robot')}",
        f"--obs_dim={get(context, 'obs_dim')}",
        f"--internal_policy={checkpoint}" if checkpoint else "",
        "--headless" if flag(context, "headless") else "",
        "--use_estimator" if flag(context, "use_estimator") else "",
        "--no_ground_truth" if flag(context, "no_ground_truth") else "",
        main=True,
    )]
    if flag(context, "joy"):
        actions.append(include("joy.launch.py"))
    return actions + record(context, "mujoco") + reward(context)


def generate_launch_description():
    return LaunchDescription([
        *common_args(),
        arg("checkpoint"),
        arg("obs_dim", "49"),
        arg("headless", "false"),
        arg("use_estimator", "false"),
        arg("no_ground_truth", "false", "withhold sim pos/vel, as on hardware"),
        arg("joy", "false", "also start the gamepad teleop"),
        *reward_args(),
        OpaqueFunction(function=setup),
    ])
