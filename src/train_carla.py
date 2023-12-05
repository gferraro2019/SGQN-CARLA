import os
import os.path as op
import sys
import time

import numpy as np
import torch
from PyQt5 import QtCore, QtWidgets
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import simple_sac
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
    env,
    agent,
    algorithm,
    n_episodes,
    L,
    step,
    args,
    test_env=False,
    eval_mode=None,
):
    episode_returns = []
    distance = None
    for n_episode in range(n_episodes):
        obs = env.reset()
        window_tot_reward.reset_tot_reward()
        app2.processEvents()
        done = False
        episode_return = 0
        episode_step = 0
        torch_obs = []
        torch_action = []
        steps = 0
        while not done:
            with torch.no_grad():
                with utils.eval_mode(agent):
                    action = agent.sample_action(obs)
                #         else:
                # with utils.eval_mode(agent):
                # action = agent.sample_action(obs)

                # #simple sac
                # agent.set_eval_mode()
                # action = agent.select_action(obs)

                cum_reward = 0
                # repeat action k times
                for _ in range(args.action_repeat):
                    steps += 1
                    obs, reward, done, info = env.step(action)
                    episode_step += 1
                    cum_reward += reward
                    distance = info["distance"]

                    if done:
                        break

                episode_return += cum_reward

                # Plot and update reward graph
                # window_reward.update_plot_data(episode_step, episode_return)
                window_reward.update_plot_data(episode_step, -distance)
                app1.processEvents()

                window_tot_reward.update_labels(env.episode, episode_return, action)
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
            # _test_env = f"_test_env_{eval_mode}" if test_env else ""
            L.log(f"eval/episode", n_episode, step)
            L.log(f"eval/episode_return", episode_return, step)
            L.log("eval/distance", distance, step)
            L.dump(step)
        episode_returns.append(episode_return)

        args.writer_tensorboard.add_scalar("Eval/distance", distance, step)
        args.writer_tensorboard.add_scalar("Eval/return", episode_return, step)


def main(
    args,
    load_model=None,
):
    # Set seed
    # utils.set_seed_everywhere(args.seed)

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

    # add tensorboard writer
    args.writer_tensorboard = SummaryWriter()

    # Define logger
    L = Logger(work_dir)

    # load datasent to to blend actual image from camera with a radom one from the database
    load_dataset_for_carla()

    # set parameters from carla env
    frame_skip = 1
    max_episode_steps = (args.episode_length + frame_skip - 1) // frame_skip
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

    # wrap env
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

        # wrap test envs
        test_env = VideoRecord_carla(test_env, args.algorithm, args.seed)
        test_env = FrameStack_carla(test_env, args.frame_stack)

        test_envs.append(test_env)
        test_envs_mode.append(args.eval_mode)

    # Create replay buffer
    replay_buffer = utils.Replay_Buffer_carla(
        capacity=args.capacity, batch_size=args.batch_size, device=args.device
    )

    print("Observations:", env.observation_space.shape)

    shp_observation = (env.observation_space[0].shape, env.observation_space[1].shape)
    print("Observations.shape:", shp_observation)

    shp_action = [2]
    print("actions.shape:", shp_action)

    # Create the agent
    agent = make_agent(obs_shape=shp_observation, action_shape=shp_action, args=args)
    # agent = simple_sac.SACAgent(state_dim=shp, action_dim=2)

    # load existing model to keep training it
    if load_model is not None:
        path, episode = load_model
        actor_state_dict = torch.load(
            op.join(path, "model", f"actor_{str(episode)}.pt")
        )
        critic_state_dict = torch.load(
            op.join(path, "model", f"critic_{str(episode)}.pt")
        )

        agent.actor.load_state_dict(actor_state_dict)
        agent.critic.load_state_dict(critic_state_dict)
        print(f"Loaded actor and critic from episode {episode}")

    # Initialize variables
    n_episode, episode_return, done = -1, 0, True
    evaluated_episodes = []
    distance = 5

    # Start training
    steps_per_episode = 0
    for train_step in range(0, args.train_steps + 1):
        # while n_episode < args.n_episodes + 1:
        # EVALUATE:
        if done:
            # if train_step > 0:
            if n_episode > 0:
                L.log("train/episode", n_episode, train_step - 1)
                L.log(
                    "train/steps_per_episode",
                    steps_per_episode,
                    train_step - 1,
                )
                L.log("train/distance", distance, train_step - 1)
                L.log("train/return", episode_return, train_step - 1)

                L.dump(train_step - 1)

                args.writer_tensorboard.add_scalar(
                    "Train/steps_per_episode",
                    steps_per_episode,
                    train_step - 1,
                )
                args.writer_tensorboard.add_scalar(
                    "Train/distance", distance, train_step - 1
                )
                args.writer_tensorboard.add_scalar(
                    "Train/return", episode_return, train_step - 1
                )

                # Save agent periodically
                if n_episode % args.save_freq == 0:
                    torch.save(
                        agent.actor.state_dict(),
                        os.path.join(model_dir, f"actor_{n_episode}.pt"),
                    )
                    torch.save(
                        agent.critic.state_dict(),
                        # agent.critic1.state_dict(),
                        os.path.join(model_dir, f"critic_{n_episode}.pt"),
                    )
                    if args.algorithm == "sgsac":
                        torch.save(
                            agent.attribution_predictor.state_dict(),
                            os.path.join(model_dir, f"attrib_predictor_{n_episode}.pt"),
                        )

                # Evaluate agent periodically
                if n_episode % args.eval_freq == 0:
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
                                args,
                                test_env=True,
                                eval_mode=test_env_mode,
                            )
                    L.dump(train_step - 1)

                    # update list evaluated episode  in order to crete the video after the training
                    for i in range(args.eval_episodes):
                        evaluated_episodes.append(n_episode + i)

            # Reset environment
            obs = env.reset()
            done = False
            episode_return = 0
            episode_step = 0
            steps_per_episode = 0  # for trackingsteps_per_episode
            window_tot_reward.reset_tot_reward()
            app2.processEvents()

            # free up memory
            torch.cuda.empty_cache()

            n_episode += 1

        # TRAIN:
        # Sample action for data collection
        if train_step < args.init_steps:
            action = env.action_space.sample()
            # a = np.zeros(2)
            # a[0] = np.clip(action[0], 0, 1)
            # a[1] = np.clip(action[1], -0.3, 0.3)
            action = np.concatenate((action[0], action[1]))
        else:
            # sgqn
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)
                # a = np.zeros(2)
                # a[0] = np.clip(action[0], 0, 1)
                # a[1] = np.clip(action[1], -0.3, 0.3)
                # action = a  # np.concatenate((a[0], a[1]))
            #     action[0] = np.clip(action[0], 0, 1)
            #     action[1] = np.clip(action[1], -0.3, 0.3)

            # simple sac
            # agent.set_train_mode()
            # action = agent.select_action(obs)

            # Run training update
            num_updates = 1  # args.init_steps if train_step == args.init_steps else 1

            for i in range(num_updates):
                agent.update(replay_buffer, L, train_step)

                # simple sac
                # agent.update(
                #     replay_buffer, batch_size=256, logger=L, trainstep=train_step
                # )

        # Take train_step
        cum_reward = 0
        for _ in range(args.action_repeat):
            steps_per_episode += 1
            next_obs, reward, done, info = env.step(action)

            episode_step += 1
            done_bool = 0

            if episode_step + 1 != env._max_episode_steps:
                done_bool = float(done)

            cum_reward += reward
            distance = info["distance"]

            # if done:
            #     if info["goal"] is True:
            #         cum_reward = 0
            #     break

        reward = cum_reward

        # Update replay buffer
        #        if not looped:
        observation = (obs, action, reward, next_obs, done_bool)
        replay_buffer.add(observation)

        episode_return += reward

        # Plot and update reward graph
        window_reward.update_plot_data(train_step, -distance)
        app1.processEvents()

        window_tot_reward.update_labels(n_episode, episode_return, action)
        app2.processEvents()

        obs = next_obs

    print("Completed training for", work_dir)
    return evaluated_episodes


if __name__ == "__main__":
    from tensorboard import program

    tracking_address = "runs"
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", tracking_address, "--port", str(7008)])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")

    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"
    np.seterr("ignore")
    # np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    args = parse_args()
    args.device = "cpu"

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

    folder = 10379
    episode = 4000
    load_model = (
        f"/home/dcas/g.ferraro/gitRepos/SGQN-CARLA/logs/carla_drive/sac/{folder}",
        episode,
    )
    load_model = None
    # try:
    evaluated_episodes = main(args, load_model)

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
