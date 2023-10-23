import os
import sys
import time

import numpy as np
import torch
from PyQt5 import QtCore, QtWidgets
from tqdm import tqdm

import utils
from algorithms.factory import make_agent
from arguments import parse_args
from carla_wrapper import CarlaEnv
from env.wrappers import FrameStack_carla, VideoRecord_carla
from logger import Logger
from utils import (
    MainWindow_Reward,
    MainWindow_Tot_Reward,
    create_video_from_images,
    load_dataset_for_carla,
)


def main(args):
    # Set seed
    utils.set_seed_everywhere(args.seed)

    frame_skip = 1
    max_episode_steps = (args.episode_length + frame_skip - 1) // frame_skip

    load_dataset_for_carla()

    car = "citroen.c3"
    car_color = "255, 0, 0"

    # Create main environment
    env = CarlaEnv(
        True,
        2000,
        0,
        frame_skip,
        "sgqn_pixel",
        False,
        car,
        car_color,
        None,
        False,
        "Custom",  # "All",
        max_episode_steps,
        lower_limit_return_=args.lower_limit_return_,
    )
    env = FrameStack_carla(env, args.frame_stack)

    print("Observations:", env.observation_space.shape)

    shp = (env.observation_space[0].shape, env.observation_space[1].shape)
    print("Observations.shapenano:", shp)

    # Create the agent
    agent = make_agent(obs_shape=shp, action_shape=env.action_space.shape, args=args)

    folder = 10164

    # Load existing actor and critic
    episodes = [str(i) for i in range(100, 2600, 100)]
    for e in episodes:
        actor_state_dict = torch.load(
            f"/home/dcas/g.ferraro/gitRepos/SGQN-CARLA/logs/carla_drive/sac/{folder}/model/actor_{e}.pt"
        )
        critic_state_dict = torch.load(
            f"/home/dcas/g.ferraro/gitRepos/SGQN-CARLA/logs/carla_drive/sac/{folder}/model/critic_{e}.pt"
        )

        print(f"Evaluating actor and critic realted to episode {e}")

        agent.actor.load_state_dict(actor_state_dict)
        agent.critic.load_state_dict(critic_state_dict)

        # EVALUATE:

        episode_rewards = []

        for n_episode in range(2):
            obs = env.reset()
            window_tot_reward.reset_tot_reward()
            app2.processEvents()
            done = False
            episode_reward = 0
            episode_step = 0
            while not done:
                with torch.no_grad():
                    with utils.eval_mode(agent):
                        action = agent.select_action(obs)

                    cum_reward = 0
                    for _ in range(args.action_repeat):
                        obs, reward, done, _ = env.step(action)
                        episode_step += 1
                        cum_reward += reward

                        if done:
                            break

                    episode_reward += cum_reward

                    # Plot and update reward graph
                    window_reward.update_plot_data(episode_step, episode_reward)
                    app1.processEvents()

                    window_tot_reward.update_labels(n_episode, episode_reward, action)
                    app2.processEvents()

            episode_rewards.append(episode_reward)

        import matplotlib.pyplot as plt

        plt.plot(episode_rewards)
        # plt.title(f"Evaluating actor and critic realted to episode {e}")
        # plt.show()


if __name__ == "__main__":
    np.seterr("ignore")
    args = parse_args()

    app1 = QtWidgets.QApplication(sys.argv)
    window_reward = MainWindow_Reward()
    window_reward.show()

    app2 = QtWidgets.QApplication(sys.argv)
    window_tot_reward = MainWindow_Tot_Reward(args.action_repeat)
    window_tot_reward.show()

    main(args)

    # # create video from images
    # save_path = os.path.join("output", "video_records", "avi")
    # create_video_from_images(
    #     evaluated_episodes, args.algorithm, args.episode_length, save_path
    # )

    sys.exit(app1.exec_())
