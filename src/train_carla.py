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
    env, agent, algorithm, n_episodes, L, step, test_env=False, eval_mode=None
):
    episode_rewards = []

    for n_episode in range(n_episodes):
        obs = env.reset()
        window_tot_reward.reset_tot_reward()
        app2.processEvents()
        done = False
        episode_reward = 0
        episode_step = 0
        torch_obs = []
        torch_action = []
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
                distance = np.linalg.norm(np.array(obs[1][0:2]))

                # Plot and update reward graph
                # window_reward.update_plot_data(episode_step, episode_reward)
                window_reward.update_plot_data(episode_step, -distance)
                app1.processEvents()

                window_tot_reward.update_labels(env.episode, episode_reward, action)
                app2.processEvents()
                # log in tensorboard 15th step
                if algorithm == "sgsac":
                    if n_episode == 0 and episode_step in [15, 16, 17, 18] and step > 0:
                        _obs = agent._obs_to_input(obs)
                        torch_obs.append(_obs)
                        torch_action.append(
                            torch.tensor(action).to(_obs.device).unsqueeze(0)
                        )
                        prefix = "eval" if eval_mode is None else eval_mode
                    if n_episode == 0 and episode_step == 18 and step > 0:
                        agent.log_tensorboard(
                            torch.cat(torch_obs, 0),
                            torch.cat(torch_action, 0),
                            step,
                            prefix=prefix,
                        )

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
        "sgqn_pixel",
        False,
        car,
        car_color,
        None,
        False,
        "Custom",  # "All",
        max_episode_steps,
        lower_limit_return_=args.lower_limit_return_,
        # visualize_target=True
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
                "sgqn_pixel",
                False,
                car,
                car_color,
                None,
                False,
                "Custom",  # "All",
                max_episode_steps,
                lower_limit_return_=args.lower_limit_return_,
                # visualize_target=True
            )
        else:
            # Hard scenario
            test_env = CarlaEnv(
                True,
                2006,
                0.1,
                frame_skip,
                "sgqn_pixel",
                True,
                car,
                car_color,
                None,
                False,
                None,
                max_episode_steps,
                lower_limit_return_=args.lower_limit_return_,
            )

        # test_env = #videoWrapper(env, cond, 1)
        test_env = VideoRecord_carla(test_env, args.algorithm, args.seed)
        test_env = FrameStack_carla(test_env, args.frame_stack)

        test_envs.append(test_env)
        test_envs_mode.append(args.eval_mode)

    # Create replay buffer
    replay_buffer = utils.Replay_Buffer_carla(
        capacity=args.capacity, batch_size=args.batch_size, device="cuda"
    )

    print("Observations:", env.observation_space.shape)

    shp = (env.observation_space[0].shape, env.observation_space[1].shape)
    print("Observations.shape:", shp)

    # Create the agent
    agent = make_agent(obs_shape=shp, action_shape=env.action_space.shape, args=args)

    # Initialize variables
    n_episode, episode_reward, done = 0, 0, True
    evaluated_episodes = []

    # initialize replay buffer

    # Start training
    start_time = time.time()
    train_step = 0
    for train_step in range(0, args.train_steps + 1):
        # while n_episode < args.n_episodes + 1:
        # EVALUATE:
        if done:
            # if train_step > 0:
            if n_episode > 0:
                L.log("train/episode", n_episode, train_step - 1)
                L.log("train/duration", time.time() - start_time, train_step - 1)
                L.dump(train_step - 1)

                # Save agent periodically
                if n_episode % args.save_freq == 0 and n_episode > 1:
                    torch.save(
                        agent.actor.state_dict(),
                        os.path.join(model_dir, f"actor_{n_episode}.pt"),
                    )
                    torch.save(
                        agent.critic.state_dict(),
                        os.path.join(model_dir, f"critic_{n_episode}.pt"),
                    )
                    if args.algorithm == "sgsac":
                        torch.save(
                            agent.attribution_predictor.state_dict(),
                            os.path.join(model_dir, f"attrib_predictor_{n_episode}.pt"),
                        )

            # Evaluate agent periodically
            if n_episode % args.eval_freq == 0 and n_episode > 0:
                print("Evaluating:", work_dir)
                L.log("eval/episode", n_episode, train_step - 1)
                # evaluate(env, agent, args.algorithm, args.eval_episodes, L, train_step)
                test_env.env.episode = n_episode
                if test_envs is not None:
                    for test_env, test_env_mode in zip(test_envs, test_envs_mode):
                        evaluate(
                            test_env,
                            agent,
                            args.algorithm,
                            args.eval_episodes,
                            L,
                            train_step - 1,
                            test_env=True,
                            eval_mode=test_env_mode,
                        )
                L.dump(train_step - 1)

                for i in range(args.eval_episodes):
                    evaluated_episodes.append(n_episode + i)

            L.log("train/episode_reward", episode_reward, train_step - 1)
            L.log("train/episode", n_episode, train_step - 1)

            # Reset environment
            obs = env.reset()
            # test_env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            window_tot_reward.reset_tot_reward()
            app2.processEvents()

            # free up memory
            torch.cuda.empty_cache()

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
            num_updates = args.init_steps if train_step == args.init_steps else 1
            for i in range(num_updates):
                agent.update(replay_buffer, L, train_step)

        # Take train_step
        cum_reward = 0
        for _ in range(args.action_repeat):
            next_obs, reward, done, _ = env.step(action)
            episode_step += 1
            done_bool = 0
            if episode_step + 1 != env._max_episode_steps:
                done_bool = float(done)

            cum_reward += reward
            if done:
                break

        reward = cum_reward
        distance = np.linalg.norm(np.array(next_obs[1][:2]))

        # Plot and update reward graph
        # window_reward.update_plot_data(train_step, reward)
        window_reward.update_plot_data(train_step, -distance)
        app1.processEvents()

        window_tot_reward.update_labels(n_episode, episode_reward, action)
        app2.processEvents()

        # Update replay buffer
        observation = (obs, action, reward, next_obs, done_bool)
        replay_buffer.add(observation)

        episode_reward += reward
        obs = next_obs
        train_step += 1

    print("Completed training for", work_dir)
    return evaluated_episodes


if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"
    np.seterr("ignore")
    np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    args = parse_args()

    app1 = QtWidgets.QApplication(sys.argv)
    window_reward = MainWindow_Reward()
    window_reward.show()

    app2 = QtWidgets.QApplication(sys.argv)
    window_tot_reward = MainWindow_Tot_Reward(args.action_repeat)
    window_tot_reward.show()

    path = os.path.join(__file__[:-19], "logs", "carla_drive", "sac")
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

    # try:
    evaluated_episodes = main(args)

    # create video from images
    save_path = os.path.join("output", str(args.seed), "video_records", "avi")
    images_path = os.path.join("output", str(args.seed), "video_records", "display")

    create_video_from_images(
        evaluated_episodes,
        args.algorithm,
        args.episode_length,
        images_path,
        save_path,
    )

    # except Exception as e:
    #     print(e)
    #     from IPython import embed

    #     embed()

    # finally:
    #     sys.exit(app1.exec_())
