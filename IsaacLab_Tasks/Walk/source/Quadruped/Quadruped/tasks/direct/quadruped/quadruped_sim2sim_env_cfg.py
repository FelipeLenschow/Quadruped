# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import os
from isaaclab_assets.robots.unitree import (
    UNITREE_A1_CFG,
    UNITREE_GO1_CFG as UNITREE_QUADRUPED_CFG,
    UNITREE_GO2_CFG,
)
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from .quadruped_env_cfg import QuadrupedEnvCfg


@configclass
class QuadrupedSim2SimEnvCfg(QuadrupedEnvCfg):
    """Sim2Sim environment config.

    Three robots (A1, Quadruped, Go2) are spawned side-by-side (Y-axis) in every
    cloned environment. The agent sees ``num_envs * 3`` parallel instances, one
    per robot type, letting you compare policy transfer in a single session.
    """

    # terrain
    from .quadruped_env_cfg import TC_FLAT, TC_ALL, TC_ROUGH, _ter
    if _ter == "flat":
        _tc = TC_FLAT
    elif _ter == "all":
        _tc = TC_ALL
    else:
        _tc = TC_ROUGH

    # Smaller default + wider spacing to fit 3 robots per env
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=6, env_spacing=8.0, replicate_physics=True
    )
    scene.terrain = _tc

    # ── Robot articulation configs ──────────────────────────────────────────
    # Y-offsets applied in QuadrupedSim2SimEnv._setup_scene: A1=-1.8, Quadruped=0, Go2=+1.8
    robot_a1_cfg: ArticulationCfg = UNITREE_A1_CFG.replace(
        prim_path="/World/envs/env_.*/RobotA1"
    )
    robot_quadruped_cfg: ArticulationCfg = UNITREE_QUADRUPED_CFG.replace(
        prim_path="/World/envs/env_.*/RobotQuadruped"
    )
    robot_go2_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="/World/envs/env_.*/RobotGo2"
    )

    # ── Contact sensors (one per robot) ────────────────────────────────────
    contact_sensor_a1: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/RobotA1/.*",
        history_length=3,
        track_air_time=False,
    )
    contact_sensor_quadruped: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/RobotQuadruped/.*",
        history_length=3,
        track_air_time=False,
    )
    contact_sensor_go2: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/RobotGo2/.*",
        history_length=3,
        track_air_time=False,
    )
    # ── Events ─────────────────────────────────────────────────────────────
    @configclass
    class EventCfg(QuadrupedEnvCfg.EventCfg):
        """Configuration for events."""
        push_robot = None

    events: EventCfg = EventCfg()
