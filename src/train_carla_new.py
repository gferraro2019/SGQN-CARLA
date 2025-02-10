import os
import sys
import time

import numpy as np
import torch
from PyQt5 import QtCore, QtWidgets
from zmq import device

# import utils
from algorithms_new.sac import SAC
from arguments import parse_args
from carla_wrapper import CarlaEnv
from env.wrappers import FrameStack_carla
from utils import (
    MainWindow_Reward,
    MainWindow_Tot_Reward,
    ReplayBuffer_carla,
    load_dataset_for_carla,
)
from utils_new import ReplayBuffer_Carla

os.system("pkill -f 'terminal'")
time.sleep(2)
os.system(
    'gnome-terminal -- bash -c "cd /home/dcas/g.ferraro/Desktop/CARLA/CARLA_0.9.14 && sh ./CarlaUE4.sh  -carla-port=2000 ; exec bash"'
)
time.sleep(3)
# np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
args = parse_args()
args.device = "cuda"
args.init_steps = 2000

app1 = QtWidgets.QApplication(sys.argv)
window_reward = MainWindow_Reward()
window_reward.show()

app2 = QtWidgets.QApplication(sys.argv)
window_tot_reward = MainWindow_Tot_Reward(args.action_repeat)
window_tot_reward.show()

# create video from images
save_path = os.path.join("output", str(args.seed), "video_records", "avi")
images_path = os.path.join("output", str(args.seed), "video_records", "display")


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
    distance_factor_between_WPs=10,
    image_size=28,
    # visualize_target=True
)

# wrap env
env = FrameStack_carla(env, args.frame_stack)

# # Create replay buffer
# replay_buffer = utils.Replay_Buffer_carla(
#     capacity=args.capacity,
#     batch_size=args.batch_size,
#     device=args.device,
#     state_shape=env.observation_space.spaces,
# )


print("Observations:", env.observation_space.shape)
shp_observation = (env.observation_space[0].shape, env.observation_space[1].shape)
print("Observations.shape:", shp_observation)

shp_action = 2
print("actions.shape:", shp_action)

replay_buffer = ReplayBuffer_Carla(
    args.capacity,
    args.batch_size,
    shp_observation,
    [shp_action],
    device=args.device,
)

# Create the agent
# agent = make_agent(shp_observation, shp_action,env.action_space.spaces, args)
agent = SAC(
    "carla",
    state_dim=shp_observation,
    action_dim=shp_action,
    device=args.device,
    replay_buffer=replay_buffer,
)
model_dir = "model"

# Initialize variables
n_episode, episode_return, done = -1, 0, True
evaluated_episodes = []
distance = 5

# Start training
steps_per_episode = 0
info = {"speed": 0}
for train_step in range(0, args.train_steps + 1):
    if done:
        if n_episode >= 0:
            # Save agent periodically
            if n_episode % args.save_freq == 0:
                agent.save(model_dir, "carla", n_episode)

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
    if train_step < args.init_steps:
        action = np.random.uniform(low=-1, high=1, size=2)
        if abs(action[1]) < 0.1:
            action[0] = 0.0
            action[1] = 0.0
    else:
        action, entropy = agent.select_action(
            (
                torch.tensor(obs[0], dtype=torch.float32).unsqueeze(0).to(args.device),
                torch.tensor(obs[1], dtype=torch.float32).unsqueeze(0).to(args.device),
            )
        )
        # clipping when close to 0
        idx = abs(action) < 0.01
        action[idx] = 0.0
        action = action[0]

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
        if done:
            break
    reward = cum_reward

    # train
    entropy = agent.train(train_step, args.device)

    # Update replay buffer
    # observation = (obs, action, reward, next_obs, done_bool)
    # replay_buffer.add(observation)

    replay_buffer.add(obs, next_obs, action, reward, done_bool)

    episode_return += reward

    # Plot and update reward graph
    window_reward.update_plot_data(train_step, -distance)
    app1.processEvents()

    window_tot_reward.update_labels(n_episode, episode_return, action, info["#WP"])
    app2.processEvents()

    del obs, action, reward, done_bool
    obs = next_obs
    del next_obs
