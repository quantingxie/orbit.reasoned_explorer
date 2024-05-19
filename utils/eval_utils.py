# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# python
import os
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


class BaseEvaluator:
    def __init__(
        self,
        distance_tolerance: float,
    ) -> None:
        # args
        self.distance_tolerance = distance_tolerance

        # parameters
        self._nbr_paths: int = 0

        return

    ##
    # Properties
    ##

    @property
    def nbr_paths(self) -> int:
        return self._nbr_paths

    def set_nbr_paths(self, nbr_paths: int) -> None:
        self._nbr_paths = nbr_paths
        return

    ##
    # Buffer
    ##

    def create_buffers(self) -> None:
        self.length_goal: np.ndarray = np.zeros(self._nbr_paths)
        self.length_path: np.ndarray = np.zeros(self._nbr_paths)
        self.path_extension: np.ndarray = np.zeros(self._nbr_paths)
        self.goal_distances: np.ndarray = np.zeros(self._nbr_paths)

    ##
    # Reset
    ##

    def reset(self) -> None:
        self.create_buffers()
        self.eval_stats = {}
        return

    ##
    # Eval Statistics
    ##

    def eval_statistics(self) -> None:
        # Evaluate results
        goal_reached = self.goal_distances < self.distance_tolerance
        goal_reached_rate = sum(goal_reached) / len(goal_reached)
        avg_distance_to_goal = sum(self.goal_distances) / len(self.goal_distances)
        avg_distance_to_goal_reached = sum(self.goal_distances[goal_reached]) / sum(goal_reached)

        print(
            "All path segments been passed. Results: \nReached goal rate"
            f" (thres: {self.distance_tolerance}):\t{goal_reached_rate} \nAvg"
            f" goal-distance (all):    \t{avg_distance_to_goal} \nAvg"
            f" goal-distance (reached):\t{avg_distance_to_goal_reached}"
        )

        self.eval_stats = {
            "goal_reached_rate": goal_reached_rate,
            "avg_distance_to_goal_all": avg_distance_to_goal,
            "avg_distance_to_goal_reached": avg_distance_to_goal_reached,
        }

        return

    def save_eval_results(self, save_name: str) -> None:
        # save eval results in yaml
        with open(save_name, "w") as file:
            yaml.dump(self.eval_stats, file)

    ##
    # Plotting
    ##

    def plt_single_model(self, eval_dir: str, show: bool = True) -> None:
        # check if directory exists
        os.makedirs(eval_dir, exist_ok=True)

        # get unique goal lengths and init buffers
        unique_goal_length = np.unique(np.round(self.length_goal, 1))
        mean_path_extension = []
        std_path_extension = []
        mean_goal_distance = []
        std_goal_distance = []
        goal_counts = []

        for x in unique_goal_length:
            # get subset of path predictions with goal length x
            subset_idx = np.round(self.length_goal, 1) == x

            mean_path_extension.append(np.mean(self.path_extension[subset_idx]))
            std_path_extension.append(np.std(self.path_extension[subset_idx]))

            mean_goal_distance.append(np.mean(self.goal_distances[subset_idx]))
            std_goal_distance.append(np.std(self.goal_distances[subset_idx]))
            goal_counts.append(len(self.goal_distances[subset_idx]))

        # plot with the distance to the goal depending on the length between goal and start
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.suptitle("Path Length Increase", fontsize=20)
        ax.plot(
            unique_goal_length,
            mean_path_extension,
            color="blue",
            label="Average path length",
        )
        ax.fill_between(
            unique_goal_length,
            np.array(mean_path_extension) - np.array(std_path_extension),
            np.array(mean_path_extension) + np.array(std_path_extension),
            color="blue",
            alpha=0.2,
            label="Uncertainty",
        )
        ax.set_xlabel("Start-Goal Distance", fontsize=16)
        ax.set_ylabel("Path Length", fontsize=16)
        ax.set_title(
            (
                "Avg increase of path length is"
                f" {round(np.mean(self.path_extension), 5)*100:.2f}% for"
                " successful paths with tolerance of"
                f" {self.distance_tolerance}"
            ),
            fontsize=16,
        )
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.legend()
        fig.savefig(os.path.join(eval_dir, "path_length.png"))
        if show:
            plt.show()
        else:
            plt.close()

        # plot to compare the increase in path length depending on the distance between goal and start
        goal_success_mean = np.sum(self.goal_distances < self.distance_tolerance) / len(self.goal_distances)

        # Create a figure and two axis objects, with the second one sharing the x-axis of the first
        fig, ax1 = plt.subplots(figsize=(12, 10))
        ax2 = ax1.twinx()
        fig.subplots_adjust(hspace=0.4)  # Add some vertical spacing between the two plots

        # Plot the goal distance data
        ax1.plot(
            unique_goal_length,
            mean_goal_distance,
            color="blue",
            label="Average goal distance length",
            zorder=2,
        )
        ax1.fill_between(
            unique_goal_length,
            np.array(mean_goal_distance) - np.array(std_goal_distance),
            np.array(mean_goal_distance) + np.array(std_goal_distance),
            color="blue",
            alpha=0.2,
            label="Uncertainty",
            zorder=1,
        )
        ax1.set_xlabel("Start-Goal Distance", fontsize=16)
        ax1.set_ylabel("Goal Distance", fontsize=16)
        ax1.set_title(
            (
                f"With a tolerance of {self.distance_tolerance} are"
                f" {round(goal_success_mean, 5)*100:.2f} % of goals reached"
            ),
            fontsize=16,
        )
        ax1.tick_params(axis="both", which="major", labelsize=14)

        # Plot the goal counts data on the second axis
        ax2.bar(
            unique_goal_length,
            goal_counts,
            color="red",
            alpha=0.5,
            width=0.05,
            label="Number of samples",
            zorder=0,
        )
        ax2.set_ylabel("Sample count", fontsize=16)
        ax2.tick_params(axis="both", which="major", labelsize=14)

        # Combine the legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        bars, bar_labels = ax2.get_legend_handles_labels()
        ax2.legend(lines + bars, labels + bar_labels, loc="upper center")

        plt.suptitle("Goal Distance", fontsize=20)
        fig.savefig(os.path.join(eval_dir, "goal_distance.png"))
        if show:
            plt.show()
        else:
            plt.close()

        return

    def plt_comparison(
        self,
        length_goal_list: List[np.ndarray],
        goal_distance_list: List[np.ndarray],
        path_extension_list: List[np.ndarray],
        model_dirs: List[str],
        save_dir: str,
        model_names: Optional[List[str]] = None,
    ) -> None:
        # path increase plot
        fig_path, axs_path = plt.subplots(figsize=(12, 10))
        fig_path.suptitle("Path Extension", fontsize=24)
        axs_path.set_xlabel("Start-Goal Distance [m]", fontsize=20)
        axs_path.set_ylabel("Path Extension [%]", fontsize=20)
        axs_path.tick_params(axis="both", which="major", labelsize=16)

        # goal distance plot
        fig_goal, axs_goal = plt.subplots(figsize=(12, 10))
        fig_goal.suptitle("Goal Distance", fontsize=24)
        axs_goal.set_xlabel("Start-Goal Distance [m]", fontsize=20)
        axs_goal.set_ylabel("Goal Distance [m]", fontsize=20)
        axs_goal.tick_params(axis="both", which="major", labelsize=16)

        bar_width = 0.8 / len(length_goal_list)

        for idx in range(len(length_goal_list)):
            if model_names is None:
                model_name = os.path.split(model_dirs[idx])[1]
            else:
                model_name = model_names[idx]

            goal_success_bool = goal_distance_list[idx] < self.distance_tolerance

            unique_goal_length = np.unique(np.round(length_goal_list[idx], 0))
            mean_path_extension = []
            std_path_extension = []
            mean_goal_distance = []
            std_goal_distance = []
            unqiue_goal_length_used = []

            for x in unique_goal_length:
                if x == 0:
                    continue

                # get subset of path predictions with goal length x
                subset_idx = np.round(length_goal_list[idx], 0) == x

                mean_path_extension.append(np.mean(path_extension_list[idx][subset_idx]))
                std_path_extension.append(np.std(path_extension_list[idx][subset_idx]))

                mean_goal_distance.append(np.mean(goal_distance_list[idx][subset_idx]))
                std_goal_distance.append(np.std(goal_distance_list[idx][subset_idx]))

                unqiue_goal_length_used.append(x)

            unique_goal_length = np.array(unqiue_goal_length_used)

            bar_pos = bar_width / 2 + idx * bar_width - 0.4
            # plot to compare the increase in path length depending in on the distance between goal and start for the successful paths
            avg_increase = np.mean(path_extension_list[idx])
            axs_path.bar(
                unique_goal_length + bar_pos,
                mean_path_extension,
                width=bar_width,
                label=(f"{model_name} (avg {round(avg_increase, 5)*100:.2f} %))"),
                alpha=0.8,
            )  # yerr=std_path_extension,
            # axs_path.plot(goal_length_path_exists, mean_path_extension, label=f'{model_name} ({round(avg_increase, 5)*100:.2f} %))')
            # axs_path.fill_between(goal_length_path_exists, np.array(mean_path_extension) - np.array(std_path_extension), np.array(mean_path_extension) + np.array(std_path_extension), alpha=0.2)

            # plot with the distance to the goal depending on the length between goal and start
            goal_success = np.sum(goal_success_bool) / len(goal_distance_list[idx])
            axs_goal.bar(
                unique_goal_length + bar_pos,
                mean_goal_distance,
                width=bar_width,
                label=(f"{model_name} (success rate" f" {round(goal_success, 5)*100:.2f} %)"),
                alpha=0.8,
            )  # yerr=std_goal_distance,
            # axs_goal.plot(unique_goal_length, mean_goal_distance, label=f'{model_name} ({round(goal_success, 5)*100:.2f} %)')
            # axs_goal.fill_between(unique_goal_length, np.array(mean_goal_distance) - np.array(std_goal_distance), np.array(mean_goal_distance) + np.array(std_goal_distance), alpha=0.2)

        # plot threshold for successful path
        axs_goal.axhline(
            y=self.distance_tolerance,
            color="red",
            linestyle="--",
            label="threshold",
        )

        axs_path.legend(fontsize=20)
        axs_goal.legend(fontsize=20)
        fig_path.savefig(os.path.join(save_dir, "path_length_comp.png"))
        fig_goal.savefig(os.path.join(save_dir, "goal_distance_comp.png"))

        plt.show()
        return


# EoF
