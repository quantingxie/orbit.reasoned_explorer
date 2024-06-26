# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains observation terms specific for reasoned_explorer.

The functions can be passed to the :class:`omni.isaac.orbit.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors.camera import CameraData

from .actions import NavigationAction

if TYPE_CHECKING:
    from omni.isaac.orbit.envs.base_env import BaseEnv


def matterport_raycast_camera_data(env: BaseEnv, sensor_cfg: SceneEntityCfg, data_type: str) -> torch.Tensor:
    """Images generated by the raycast camera."""
    # extract the used quantities (to enable type-hinting)
    sensor: CameraData = env.scene.sensors[sensor_cfg.name].data

    # return the data
    if data_type == "distance_to_image_plane":
        output = sensor.output[data_type].clone().unsqueeze(1)
        output[torch.isnan(output)] = 0.0
        output[torch.isinf(output)] = 0.0
        return output
    else:
        return sensor.output[data_type].clone().permute(0, 3, 1, 2)


def isaac_camera_data(env: BaseEnv, sensor_cfg: SceneEntityCfg, data_type: str) -> torch.Tensor:
    """Images generated by the usd camera."""
    # extract the used quantities (to enable type-hinting)
    sensor: CameraData = env.scene.sensors[sensor_cfg.name].data

    # return the data
    if data_type == "distance_to_image_plane":
        output = sensor.output[data_type].clone().unsqueeze(1)
        output[torch.isnan(output)] = 0.0
        output[torch.isinf(output)] = 0.0
        return output
    else:
        return sensor.output[data_type].clone()


def cam_position(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Position of the camera."""
    # extract the used quantities (to enable type-hinting)
    sensor: CameraData = env.scene.sensors[sensor_cfg.name].data

    return sensor.pos_w.clone()


def cam_orientation(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Orientation of the camera."""
    # extract the used quantities (to enable type-hinting)
    sensor: CameraData = env.scene.sensors[sensor_cfg.name].data

    return sensor.quat_w_world.clone()


def low_level_actions(env: BaseEnv) -> torch.Tensor:
    """Low-level actions."""
    # extract the used quantities (to enable type-hinting)
    action_term: NavigationAction = env.action_manager._terms[0]

    return action_term.low_level_actions.clone()
