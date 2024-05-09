
import os

import time
import cv2
import logging
import networkx as nx
from openai import OpenAI

import carb
import omni.isaac.orbit.utils.math as math_utils
import torch
import torchvision.transforms as transforms
from omni.isaac.debug_draw import _debug_draw


"""
ReasonedExplorer Helpers
"""


class ReasonedExplorerAlgo:
    def __init__(self, type, model, branches, rounds, goal, openai_api_key):
        """Apply ReasonedExplorer Algorithm
        """
        super().__init__()

        # params
        self.goal = goal
        self.type = type
        self.openai_api_key = openai_api_key        
        #self.rrt = RRT(model, goal, branches, rounds, openai_api_key)
        #self.graph_manager = GraphManager()

        self.step_counter = 0
        self.iteration_count = 0

        self.client = OpenAI(api_key=openai_api_key)

        # get transforms for images
        self.transform = transforms.Resize(self.train_config.img_input_size, antialias=None)

        # setup waypoint display in Isaac
        self.draw = _debug_draw.acquire_debug_draw_interface()
        self.color_fear = [(1.0, 0.4, 0.1, 1.0)]  # red
        self.color_path = [(0.4, 1.0, 0.1, 1.0)]  # green
        self.size = [5.0]

    ###
    # Transformations
    ###

    def goal_transformer(self, goal: torch.Tensor, cam_pos: torch.Tensor, cam_quat: torch.Tensor) -> torch.Tensor:
        """transform goal into camera frame"""
        goal_cam_frame = goal - cam_pos
        goal_cam_frame[:, 2] = 0  # trained with z difference of 0
        goal_cam_frame = math_utils.quat_apply(math_utils.quat_inv(cam_quat), goal_cam_frame)
        return goal_cam_frame

    def path_transformer(
        self, path_cam_frame: torch.Tensor, cam_pos: torch.Tensor, cam_quat: torch.Tensor
    ) -> torch.Tensor:
        """transform path from camera frame to world frame"""
        return math_utils.quat_apply(
            cam_quat.unsqueeze(1).repeat(1, path_cam_frame.shape[1], 1), path_cam_frame
        ) + cam_pos.unsqueeze(1)

    def input_transformer(self, image: torch.Tensor) -> torch.Tensor:
        # transform images
        image = self.transform(image)
        image[image > self.max_depth] = 0.0
        image[~torch.isfinite(image)] = 0  # set all inf or nan values to 0
        return image

    ###
    # Debug Draw
    ###

    def debug_draw(self, paths: torch.Tensor, fear: torch.Tensor, goal: torch.Tensor):
        self.draw.clear_lines()
        self.draw.clear_points()

        def draw_single_traj(traj, color, size):
            traj[:, 2] = torch.mean(traj[:, 2])
            self.draw.draw_lines(traj[:-1].tolist(), traj[1:].tolist(), color * len(traj[1:]), size * len(traj[1:]))

        for idx, curr_path in enumerate(paths):
            if fear[idx] > self.fear_threshold:
                draw_single_traj(curr_path, self.color_fear, self.size)
                self.draw.draw_points(goal.tolist(), self.color_fear * len(goal), self.size * len(goal))
            else:
                draw_single_traj(curr_path, self.color_path, self.size)
                self.draw.draw_points(goal.tolist(), self.color_path * len(goal), self.size * len(goal))
