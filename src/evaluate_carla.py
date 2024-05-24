import os
import sys
import time

import numpy as np
import torch
from PyQt5 import QtCore, QtWidgets
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
from algorithms.factory import make_agent
from arguments import parse_args
from carla_wrapper import CarlaEnv
from env.wrappers import FrameStack_carla, VideoRecord_carla
from logger import Logger
from utils import (MainWindow_Reward, MainWindow_Tot_Reward,
                   create_video_from_images, load_dataset_for_carla)


def main(args):
    # # Set seed
    # utils.set_seed_everywhere(args.seed)

    frame_skip = 1
    max_episode_steps = (args.episode_length + frame_skip - 1) // frame_skip

    # add tensorboard writer
    args.writer_tensorboard = SummaryWriter()
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
        distance_factor_between_WPs=10,
        size_target_point=args.size_target_point
    )
    env = FrameStack_carla(env, args.frame_stack)

    print("Observations:", env.observation_space.shape)

    shp = (env.observation_space[0].shape, env.observation_space[1].shape)
    print("Observations.shapenano:", shp)

    args.minimum_alpha = 0.3


    # Create the agent
    agent = make_agent(obs_shape=shp, action_shape=[2], env_action_spaces=env.action_space.spaces,args=args)

    folder = 10226

    k = 0
    # Load existing actor and critic
    episodes = [str(i) for i in range(50, 1000, 50)]
    for e in episodes:
        try:
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
            info = {"speed":0}

            start_time = time.time()
            for n_episode in range(3):
                obs = env.reset()
                window_tot_reward.reset_tot_reward()
                app2.processEvents()
                done = False
                episode_reward = 0
                episode_step = 0
                duration = 0
                distance = None
                while not done:
                    with torch.no_grad():
                        with utils.eval_mode(agent):
                            action = agent.sample_action(obs)

                        if abs(action[1]) < 0.1:
                            action[1]=0.0
                    
                        if info["speed"]>=20:
                            action[0]=0.0

                        cum_reward = 0
                        for _ in range(args.action_repeat):
                            obs, reward, done, info = env.step(action)
                            episode_step += 1
                            cum_reward += reward

                            if done:
                                break

                        episode_reward += cum_reward
                        distance = np.linalg.norm(np.array(obs[1][0:2]))

                        # Plot and update reward graph
                        window_reward.update_plot_data(episode_step, -distance)
                        app1.processEvents()

                        window_tot_reward.update_labels(
                            n_episode, episode_reward, action,info["#WP"]
                        )
                        app2.processEvents()

                duration = time.time() - start_time
                args.writer_tensorboard.add_scalar("Train/duration", duration, k)
                args.writer_tensorboard.add_scalar("Train/distance", distance, k)
                args.writer_tensorboard.add_scalar("Train/return", episode_reward, k)
                episode_rewards.append(episode_reward)
                k += 1
        except:
            print(f"model {e} not exist")
        # import matplotlib.pyplot as plt

        # plt.plot(episode_rewards)
        # plt.title(f"Evaluating actor and critic realted to episode {e}")
        # plt.show()


if __name__ == "__main__":
    np.seterr("ignore")
    np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
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
