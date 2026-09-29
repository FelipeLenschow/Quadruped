import math
import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import OpaqueFunction
from launch_ros.actions import Node

from quadruped_bringup.launch_common import arg, flag, get
from quadruped_core import paths

NAV2_NODES = ["controller_server", "planner_server", "behavior_server", "bt_navigator"]


def map_file(name):
    if name.endswith(".yaml") and os.path.exists(name):
        return os.path.abspath(name)
    return paths.repo("maps", f"{name}.yaml")


def merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            merge(base[k], v)
        else:
            base[k] = v


def nav2_params(sim, keepout, initial_pose):
    """nav2.yaml for this run: the real_robot overrides off sim, the keepout filter only when there is a mask,
    AMCL's start pose."""
    with open(paths.config("nav2.yaml")) as f:
        params = yaml.safe_load(f)
    real = params.pop("real_robot", {})
    if not sim:
        merge(params, real)
    for costmap in ("local_costmap", "global_costmap"):
        p = params[costmap][costmap]["ros__parameters"]
        if keepout:
            p["filters"] = ["keepout_filter"]
            p["keepout_filter"] = {"plugin": "nav2_costmap_2d::KeepoutFilter", "enabled": True,
                                   "filter_info_topic": "/costmap_filter_info"}
    amcl = params["amcl"]["ros__parameters"]
    amcl["set_initial_pose"] = initial_pose is not None
    if initial_pose is not None:
        x, y, yaw = initial_pose
        amcl["initial_pose"] = {"x": x, "y": y, "z": 0.0, "yaw": yaw}
    for node in params.values():
        node.setdefault("ros__parameters", {})["use_sim_time"] = sim
    out = tempfile.NamedTemporaryFile("w", prefix="nav2_", suffix=".yaml", delete=False)
    yaml.safe_dump(params, out)
    out.close()
    return out.name


def setup(context):
    sim = flag(context, "sim")
    use_sim = {"use_sim_time": sim}
    map_name = get(context, "map")
    pose = get(context, "initial_pose").strip().lower()
    initial_pose = None if pose in ("", "unknown") else tuple(float(v) for v in pose.split(","))
    map_yaml = map_file(map_name) if map_name else None
    keepout_yaml = map_yaml[:-len(".yaml")] + "_keepout.yaml" if map_yaml else None
    keepout = bool(keepout_yaml and os.path.exists(keepout_yaml))
    params = nav2_params(sim, keepout, initial_pose)
    to_mux = [("cmd_vel", "/cmd_vel/nav")]

    actions = [
        Node(package="quadruped_perception", executable="lidar_filter", output="screen", parameters=[use_sim]),
        Node(package="pointcloud_to_laserscan", executable="pointcloud_to_laserscan_node", output="log",
             remappings=[("cloud_in", "/lidar/points_filtered"), ("scan", "/scan")],
             parameters=[use_sim, {
                 "target_frame": "base_footprint", "transform_tolerance": 0.05,
                 "min_height": 0.10, "max_height": 0.60,
                 "angle_min": -math.pi, "angle_max": math.pi, "angle_increment": 0.01745,
                 "scan_time": 0.1, "range_min": 0.2, "range_max": 15.0, "use_inf": True,
             }]),
        Node(package="nav2_controller", executable="controller_server", output="log", parameters=[params],
             remappings=to_mux),
        Node(package="nav2_planner", executable="planner_server", output="log", parameters=[params]),
        Node(package="nav2_behaviors", executable="behavior_server", output="log", parameters=[params],
             remappings=to_mux),
        Node(package="nav2_bt_navigator", executable="bt_navigator", output="log", parameters=[params]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager", name="lifecycle_manager_navigation",
             output="screen", parameters=[use_sim, {"autostart": True, "node_names": NAV2_NODES}]),
    ]

    if map_yaml is None:
        actions.append(Node(package="slam_toolbox", executable="async_slam_toolbox_node", output="screen",
                            emulate_tty=True, parameters=[paths.config("slam.yaml"), use_sim]))
    else:
        if not os.path.exists(map_yaml):
            raise RuntimeError(f"map not found: {map_yaml}")
        localization = ["map_server", "amcl"]
        actions += [
            Node(package="nav2_map_server", executable="map_server", output="screen",
                 parameters=[use_sim, {"yaml_filename": map_yaml}]),
            Node(package="nav2_amcl", executable="amcl", output="screen", emulate_tty=True, parameters=[params]),
        ]
        if keepout:
            localization += ["filter_mask_server", "costmap_filter_info_server"]
            actions += [
                Node(package="nav2_map_server", executable="map_server", name="filter_mask_server", output="log",
                     parameters=[use_sim, {"yaml_filename": keepout_yaml, "topic_name": "/keepout_filter_mask",
                                           "frame_id": "map"}]),
                Node(package="nav2_map_server", executable="costmap_filter_info_server", output="log",
                     parameters=[use_sim, {"type": 0, "filter_info_topic": "/costmap_filter_info",
                                           "mask_topic": "/keepout_filter_mask", "base": 0.0, "multiplier": 1.0}]),
            ]
        actions.append(Node(package="nav2_lifecycle_manager", executable="lifecycle_manager",
                            name="lifecycle_manager_localization", output="screen",
                            parameters=[use_sim, {"autostart": True, "node_names": localization}]))
        if initial_pose is None and flag(context, "auto_localize"):
            actions.append(Node(package="quadruped_operator", executable="auto_localize", output="screen",
                                emulate_tty=True))

    if flag(context, "rviz"):
        view = os.path.join(get_package_share_directory("nav2_bringup"), "rviz", "nav2_default_view.rviz")
        actions.append(Node(package="rviz2", executable="rviz2", arguments=["-d", view], output="log",
                            parameters=[use_sim]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        arg("sim", "true", "use /clock (Gazebo); false on the real robot"),
        arg("map", "", "saved map name in maps/ (or a .yaml path): localize with AMCL; empty builds one with SLAM"),
        arg("initial_pose", "0,0,0", "x,y,yaw on the map for AMCL; 'unknown' to find it with the localize skill"),
        arg("auto_localize", "true", "with initial_pose:=unknown, run the localize skill once in policy mode"),
        arg("rviz", "true"),
        OpaqueFunction(function=setup),
    ])
