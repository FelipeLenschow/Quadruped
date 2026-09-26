from launch import LaunchDescription
from launch.actions import OpaqueFunction

from quadruped_bringup.launch_common import arg, common_args, flag, get, node, reward, reward_args


def setup(context):
    sim = get(context, "sim")
    if sim not in ("mujoco", "gazebo"):
        raise ValueError(f"sim must be mujoco or gazebo, got {sim!r}")
    return [node(
        "quadruped_operator", f"{sim}_twin",
        f"--robot={get(context, 'robot')}",
        "--use_estimator" if flag(context, "use_estimator") else "",
        "--no_ghost" if sim == "mujoco" and not flag(context, "ghost") else "",
        main=True,
    )] + reward(context)


def generate_launch_description():
    return LaunchDescription([
        *common_args(),
        arg("sim", "mujoco", "mujoco or gazebo"),
        arg("use_estimator", "false"),
        arg("ghost", "true", "MuJoCo only: draw the ghost robot"),
        *reward_args(),
        OpaqueFunction(function=setup),
    ])
