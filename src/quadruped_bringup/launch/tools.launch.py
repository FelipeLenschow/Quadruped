from launch import LaunchDescription
from launch.actions import ExecuteProcess, OpaqueFunction, Shutdown
from launch_ros.actions import Node

from quadruped_bringup.launch_common import arg, get
from quadruped_core import paths

TOOLS = {
    "plotjuggler": ("plotjuggler", "plotjuggler"),
    "rviz": ("rviz2", "rviz2"),
    "rqt_graph": ("rqt_graph", "rqt_graph"),
    "tf2_tree": ("rqt_tf_tree", "rqt_tf_tree"),
}


def setup(context):
    tool = get(context, "tool")
    if tool == "foxglove":
        return [ExecuteProcess(cmd=["ros2", "launch", "foxglove_bridge", "foxglove_bridge_launch.xml"],
                               output="screen", on_exit=[Shutdown()])]
    if tool not in TOOLS:
        raise ValueError(f"tool must be one of {sorted([*TOOLS, 'foxglove'])}, got {tool!r}")
    package, executable = TOOLS[tool]
    arguments = ["--layout", paths.config("plotjuggler_reward_layout.xml")] if tool == "plotjuggler" else []
    return [Node(package=package, executable=executable, arguments=arguments, output="screen", on_exit=[Shutdown()])]


def generate_launch_description():
    return LaunchDescription([
        arg("tool", "rviz", "plotjuggler, rviz, rqt_graph, tf2_tree or foxglove"),
        OpaqueFunction(function=setup),
    ])
