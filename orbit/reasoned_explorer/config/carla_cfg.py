# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import AssetBaseCfg
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.orbit.utils import configclass
from orbit.reasoned_explorer.utils import UnRealImporterCfg

##
# Pre-defined configs
##
# isort: off
from omni.isaac.orbit_assets.anymal import ANYMAL_C_CFG
from .base_cfg import ReasonedExplorerBaseCfg
from ..reasoned_explorer import DATA_DIR

##
# Scene definition
##


@configclass
class TerrainSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = UnRealImporterCfg(
        prim_path="/World/Carla",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        usd_path="../../../usd/carla.usd",
        groundplane=True,
        cw_config_file=os.path.join(DATA_DIR, "town01", "cw_multiply_cfg.yml"),
        sem_mesh_to_class_map=os.path.join(DATA_DIR, "town01", "keyword_mapping.yml"),
        people_config_file=os.path.join(DATA_DIR, "town01", "people_cfg.yml"),
        vehicle_config_file=os.path.join(DATA_DIR, "town01", "vehicle_cfg.yml"),
        axis_up="Z",
    )
    
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=1000.0,
        ),
    )

    # robots
    robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (8.0, -0.5, 0.6)
    robot.init_state.rot = (0.5253, 0.0, 0.0, 0.8509)

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/GroundPlane"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, debug_vis=False)


    # camera
    front_depth_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/depth_camera_front",
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5)),
        spawn=sim_utils.PinholeCameraCfg(),
        width=848,
        height=480,
        data_types=["distance_to_image_plane", "rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
        ),
    )
    left_depth_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/depth_camera_left",
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.18301270189221938, -0.6830127018922193, 0.6830127018922193, -0.18301270189221938)),
        spawn=sim_utils.PinholeCameraCfg(),
        width=848,
        height=480,
        data_types=["distance_to_image_plane", "rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
        ),
    )
    right_depth_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/depth_camera_right",
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.6830127018922193, -0.18301270189221938, 0.18301270189221938, -0.6830127018922193)),
        spawn=sim_utils.PinholeCameraCfg(),
        width=848,
        height=480,
        data_types=["distance_to_image_plane", "rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
        ),
    )

##
# Environment configuration
##


@configclass
class ReasonedExplorerCarlaCfg(ReasonedExplorerBaseCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: TerrainSceneCfg = TerrainSceneCfg(num_envs=1, env_spacing=1.0, replicate_physics=False)

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # adapt viewer
        self.viewer.eye = (105, -132, 6.5)
        self.viewer.lookat = (113.5, -132, 1.0)
        # change ANYmal position
        self.scene.robot.init_state.pos = (118.0, -126.0, 1.0)
