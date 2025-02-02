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


def evaluate(
    env, agent, algorithm, num_episodes, L, step, test_env=False, eval_mode=None
):
    episode_rewards = []
    for i in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_step = 0
        torch_obs = []
        torch_action = []
        while not done:
            with torch.no_grad():
                with utils.eval_mode(agent):
                    action = agent.select_action(obs)

                obs, reward, done, _ = env.step(action)
                episode_reward += reward
                # log in tensorboard 15th step
                if algorithm == "sgsac":
                    if i == 0 and episode_step in [15, 16, 17, 18] and step > 0:
                        _obs = agent._obs_to_input(obs)
                        torch_obs.append(_obs)
                        torch_action.append(
                            torch.tensor(action).to(_obs.device).unsqueeze(0)
                        )
                        prefix = "eval" if eval_mode is None else eval_mode
                    if i == 0 and episode_step == 18 and step > 0:
                        agent.log_tensorboard(
                            torch.cat(torch_obs, 0),
                            torch.cat(torch_action, 0),
                            step,
                            prefix=prefix,
                        )

                episode_step += 1

        if L is not None:
            _test_env = f"_test_env_{eval_mode}" if test_env else ""
            L.log(f"eval/episode_reward{_test_env}", episode_reward, step)
        episode_rewards.append(episode_reward)


def main(args):
    # Set seed
    utils.set_seed_everywhere(args.seed)

    # Create working directory
    work_dir = os.path.join(
        "logs", args.domain_name + "_drive", args.algorithm, str(args.seed)
    )

    print("Working directory:", work_dir)
    assert not os.path.exists(
        os.path.join(work_dir, "train.log")
    ), "specified working directory already exists"

    utils.make_dir(work_dir)

    model_dir = utils.make_dir(os.path.join(work_dir, "model"))
    utils.write_info(args, os.path.join(work_dir, "info.log"))

    # Define logger
    L = Logger(work_dir)

    frame_skip = 1
    max_episode_steps = (args.episode_length + frame_skip - 1) // frame_skip

    load_dataset_for_carla()

    car = "citroen.c3"
    car_color = "255, 0, 0"

    # Create main environment
    env = CarlaEnv(
        False,
        2000,
        0,
        frame_skip,
        "pixel",
        False,
        car,
        car_color,
        None,
        False,
        "All",
        max_episode_steps,
    )
    env = FrameStack_carla(env, args.frame_stack)

    # Create test environments
    test_envs = []
    test_envs_mode = []
    for cond in ["color_easy"]:  # , "color_hard"]:
        if cond == "color_easy":
            # Easy scenario: no traffic, no dyanimc weather, no layers but roads and lighters
            test_env = CarlaEnv(
                True,
                2003,
                0,
                frame_skip,
                "pixel",
                False,
                car,
                car_color,
                None,
                False,
                "All",
                max_episode_steps,
            )
        else:  # Hard scenario
            test_env = CarlaEnv(
                True,
                2006,
                0.1,
                frame_skip,
                "pixel",
                True,
                car,
                car_color,
                None,
                False,
                None,
                max_episode_steps,
            )

        # test_env = #videoWrapper(env, cond, 1)
        test_env = VideoRecord_carla(test_env, args.algorithm)
        test_env = FrameStack_carla(test_env, args.frame_stack)

        test_envs.append(test_env)
        test_envs_mode.append(args.eval_mode)

    # Create replay buffer
    replay_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.train_steps,
        batch_size=args.batch_size,
    )

    # Define observation
    cropped_obs_shape = (
        3 * args.frame_stack,
        args.image_crop_size,
        args.image_crop_size,
    )
    print("Observations:", env.observation_space.shape)
    print("Cropped observations:", cropped_obs_shape)

    # Create the agent
    agent = make_agent(
        obs_shape=cropped_obs_shape, action_shape=env.action_space.shape, args=args
    )

    # Initialize variables
    n_episode, episode_reward, done = 0, 0, True

    # Start training
    start_time = time.time()
    for train_step in tqdm(range(0, args.train_steps + 1)):
        # EVALUATE:
        if done:
            if train_step > 0:
                L.log("train/duration", time.time() - start_time, train_step)
                L.dump(train_step)

                # Save agent periodically
                if train_step % args.save_freq == 0:
                    torch.save(
                        agent.actor.state_dict(),
                        os.path.join(model_dir, f"actor_{train_step}.pt"),
                    )
                    torch.save(
                        agent.critic.state_dict(),
                        os.path.join(model_dir, f"critic_{train_step}.pt"),
                    )
                    if args.algorithm == "sgsac":
                        torch.save(
                            agent.attribution_predictor.state_dict(),
                            os.path.join(
                                model_dir, f"attrib_predictor_{train_step}.pt"
                            ),
                        )

            # Evaluate agent periodically
            if train_step % args.eval_freq == 0:
                print("Evaluating:", work_dir)
                L.log("eval/n_episode", n_episode, train_step)
                evaluate(env, agent, args.algorithm, args.eval_episodes, L, train_step)
                if test_envs is not None:
                    for test_env, test_env_mode in zip(test_envs, test_envs_mode):
                        evaluate(
                            test_env,
                            agent,
                            args.algorithm,
                            args.eval_episodes,
                            L,
                            train_step,
                            test_env=True,
                            eval_mode=test_env_mode,
                        )
                L.dump(train_step)

            L.log("train/episode_reward", episode_reward, train_step)
            L.log("train/n_episode", n_episode, train_step)

            # Reset environment
            obs = env.reset()
            test_env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            window_tot_reward.reset_tot_reward()
            app2.processEvents()

            n_episode += 1
            start_time = time.time()

        # TRAIN:
        # Sample action for data collection
        if train_step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # Run training update
        if train_step >= args.init_steps:
            num_updates = args.init_steps if train_step == args.init_steps else 1
            for i in range(num_updates):
                agent.update(replay_buffer, L, train_step, i)

        # Take train_step
        cum_reward = 0
        for _ in range(args.action_repeat):
            next_obs, reward, done, _ = env.step(action)
            done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
            cum_reward += reward

        reward = cum_reward

        # Plot and update reward graph
        window_reward.update_plot_data(train_step, reward)
        app1.processEvents()

        window_tot_reward.update_labels(n_episode, reward, action)
        app2.processEvents()

        # Update replay buffer
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        episode_reward += reward
        obs = next_obs
        episode_step += 1

    print("Completed training for", work_dir)
    return n_episode


if __name__ == "__main__":
    app1 = QtWidgets.QApplication(sys.argv)
    window_reward = MainWindow_Reward()
    window_reward.show()

    app2 = QtWidgets.QApplication(sys.argv)
    window_tot_reward = MainWindow_Tot_Reward()
    window_tot_reward.show()

    args = parse_args()

    path = os.path.join(__file__[:-19], "logs", "carla_drive", "sgsac")
    if os.path.exists(path):
        args.seed = (
            max(
                map(
                    int,
                    os.listdir(path),
                )
            )
            + 1
        )

    n_episodes = main(args)

    # create video from images
    save_path = os.path.join("output", "video_records", "avi")
    create_video_from_images(n_episodes, args.episode_length, save_path)

    sys.exit(app1.exec_())
