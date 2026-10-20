"""The ROS side of real.launch.py, one node at a time, while DLS2 (go2_hal + ros2_translator) has the robot
instead of real_driver. Every node is off until its argument turns it on."""
from launch import LaunchDescription
from launch.actions import OpaqueFunction

from quadruped_bringup.launch_common import arg, flag, get, interface, node


def setup(context):
    actions = []
    if flag(context, "mock"):
        # lo (default) keeps it off every network. Any other interface only on a cable with no Go2 on it (PC
        # straight to the Jetson): the Go2 uses the same DDS domain and topics. go2_hal uses the same interface.
        actions.append(node("quadruped_real", "mock_go2", f"--interface={get(context, 'mock_interface') or 'lo'}",
                            "" if flag(context, "mock_window") else "--headless"))
    if flag(context, "sensors"):
        net_interface = get(context, "interface")
        actions.append(node("quadruped_real", "real_sensors", f"--interface={net_interface}" if net_interface else ""))
    if flag(context, "monitor"):
        actions.append(node("quadruped_drivers", "system_monitor"))
    if flag(context, "gps"):
        actions.append(node("quadruped_drivers", "gps_driver"))
    return actions + interface(context)


def generate_launch_description():
    return LaunchDescription([
        arg("mock", "false", "mock_go2: a MuJoCo Go2 on the Unitree SDK topics"),
        arg("mock_interface", "lo", "mock_go2's network interface; not lo only on a cable with no Go2 on it"),
        arg("mock_window", "true", "show mock_go2's MuJoCo window"),
        arg("mux", "false", "twist_mux: /cmd_vel/{joy,keyboard,phone,skill,nav,llm} -> /cmd_vel"),
        arg("skills", "false", "skill_server: the /skill service"),
        arg("monitor", "false", "system_monitor: CPU/GPU/RAM/temperatures to /system_stats"),
        arg("sensors", "false", "real_sensors: lidar cloud to /lidar/points and the sensor TF (real Go2)"),
        arg("interface", "", "network interface for the Unitree SDK (real_sensors)"),
        arg("mcp", "false", "mcp_server: the MCP server for an LLM (config.yaml mcp:)"),
        arg("gps", "false", "gps_driver: the u-blox RTK GPS to /gps/fix (config.yaml gps:)"),
        OpaqueFunction(function=setup),
    ])
