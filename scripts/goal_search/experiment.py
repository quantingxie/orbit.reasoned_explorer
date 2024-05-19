import argparse
import torch

# omni-isaac-orbit
from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to use the camera sensor.")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--conv_distance", default=0.2, type=float, help="Distance for a goal considered to be reached.")
parser.add_argument("--scene", default="carla", choices=["matterport", "carla", "warehouse"], type=str, help="Scene to load.")
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

from omni.isaac.orbit.envs import RLTaskEnv

from orbit.reasoned_explorer.config import (ReasonedExplorerCarlaCfg)

def main():
    # create environment cfg
    if args_cli.scene == "carla":
        env_cfg = ReasonedExplorerCarlaCfg()
        goal_pos = torch.tensor([111.0, -137.0, 1.0])
    else:
        raise NotImplementedError(f"Scene {args_cli.scene} not yet supported!")

    env = RLTaskEnv(env_cfg)
    obs, _ = env.reset()

    env.sim.pause()

    count = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 1000 == 0:
                obs, _ = env.reset()
                count = 0
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # infer action
            action = (0, 0, 0)
            # step env
            obs, _ = env.step(action)
            # update counter
            count += 1

            # print information from the sensors
            print("-------------------------------")
            print(obs["policy"]["base_lin_vel"])
            print(obs["policy"]["base_ang_vel"])
            print(obs["policy"]["projected_gravity"])
            print(obs["policy"]["velocity_commands"])
            print(obs["policy"]["joint_pos"])
            print(obs["policy"]["joint_vel"])
            print(obs["policy"]["actions"])
            print("-------------------------------")
            print(obs["policy"]["height_scan"])
            print("Depth Camera:")
            print(obs["policy"]["front_depth_measurement"])
            print("RGB Camera:")
            print(obs["policy"]["front_rgb_measurement"])

