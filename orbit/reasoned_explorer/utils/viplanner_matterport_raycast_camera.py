# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import yaml
from omni.isaac.matterport.domains import MatterportRayCasterCamera
from omni.isaac.orbit.sensors.ray_caster import RayCasterCameraCfg
from omni.isaac.orbit.utils.configclass import configclass
from orbit.reasoned_explorer.viplanner import DATA_DIR

from viplanner.config.viplanner_sem_meta import ReasonedExplorerSemMetaHandler


class ReasonedExplorerMatterportRayCasterCamera(MatterportRayCasterCamera):
    def __init__(self, cfg: object):
        super().__init__(cfg)

    def _color_mapping(self):
        viplanner_sem = ReasonedExplorerSemMetaHandler()
        with open(DATA_DIR + "/mpcat40_to_vip_sem.yml") as file:
            map_mpcat40_to_vip_sem = yaml.safe_load(file)
        color = viplanner_sem.get_colors_for_names(list(map_mpcat40_to_vip_sem.values()))
        self.color = torch.tensor(color, device=self._device, dtype=torch.uint8)


@configclass
class ReasonedExplorerMatterportRayCasterCameraCfg(RayCasterCameraCfg):
    class_type = ReasonedExplorerMatterportRayCasterCamera
