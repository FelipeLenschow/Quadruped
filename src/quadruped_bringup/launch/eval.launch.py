from launch import LaunchDescription
from launch.actions import OpaqueFunction

from quadruped_bringup.launch_common import arg, common_args, flag, get, node, record, reward, reward_args


def setup(context):
    return [node(
        "quadruped_drivers", "eval_mujoco",
        f"--robot={get(context, 'robot')}",
        f"--internal_policy={get(context, 'checkpoint')}",
        f"--obs_dim={get(context, 'obs_dim')}",
        "--headless" if flag(context, "headless") else "",
        "--use_estimator" if flag(context, "use_estimator") else "",
        main=True,
    )] + record(context, "eval") + reward(context)


def generate_launch_description():
    return LaunchDescription([
        *common_args(),
        arg("checkpoint"),
        arg("obs_dim", "49"),
        arg("headless", "false"),
        arg("use_estimator", "false"),
        *reward_args(),
        OpaqueFunction(function=setup),
    ])
