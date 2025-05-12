import os
import pickle
import random

import gym
import numpy as np
import torch
from zmq import device


def load_replay_buffer(filename="replay_buffer"):
    print("loading replay buffer...")
    file = open(filename, "rb")
    replay_buffer = pickle.load(file)
    file.close()
    print("Done")
    print(len(replay_buffer))
    return replay_buffer


def seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/self.py


class ReplayBuffer_Carla:

    @torch.no_grad()
    def __init__(
        self,
        capacity=10_000,
        batch_size=32,
        state_shape=(4, 120, 160),
        action_shape=(1, 2),
        device="cpu",
        normalize_rewards=False,
        min_number_samples=30_000,
    ):
        self.device = device
        # self.content = []
        self.state_shape = state_shape

        self.capacity = capacity
        self.min_number_samples = min_number_samples
        self.idx = 0
        self.filled = False
        self.batch_size = batch_size
        self.indices = np.zeros(batch_size)
        self.normalize_rewards = normalize_rewards
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.states_img = (
            torch.empty((capacity, *state_shape[0]), dtype=torch.float32)
            .detach()
            .to(self.device)
        )
        self.states = (
            torch.empty((capacity, *state_shape[1]), dtype=torch.float32)
            .detach()
            .to(self.device)
        )
        self.actions = (
            torch.empty((capacity, *action_shape), dtype=torch.float32)
            .detach()
            .to(self.device)
        )
        self.rewards = (
            torch.empty(capacity, dtype=torch.float32).detach().to(self.device)
        )
        self.next_states_img = (
            torch.empty((capacity, *state_shape[0]), dtype=torch.float32)
            .detach()
            .to(self.device)
        )
        self.next_states = (
            torch.empty((capacity, *state_shape[1]), dtype=torch.float32)
            .detach()
            .to(self.device)
        )
        self.dones = torch.empty(capacity, dtype=torch.bool).detach().to(self.device)

    def change_bacth_size(self, batch_size):
        self.batch_size = batch_size
        del self.indices
        self.indices = np.zeros(batch_size)

    def save(self, filename="replay_buffer"):
        print("saving replay buffer...")
        file = open(filename, "wb")
        pickle.dump(self, file, 4)
        file.close()
        print("Done")

    @torch.no_grad()
    def add(self, obs, next_obs, actions, rewards, dones):
        temp_tensor = torch.tensor(obs[0], dtype=torch.float32).detach().to(self.device)

        self.states_img[self.idx] = temp_tensor
        del temp_tensor

        temp_tensor = torch.tensor(obs[1], dtype=torch.float32).detach().to(self.device)
        self.states[self.idx] = temp_tensor
        del temp_tensor

        temp_tensor = (
            torch.tensor(actions, dtype=torch.float32).detach().to(self.device)
        )
        self.actions[self.idx] = temp_tensor
        del temp_tensor

        temp_tensor = (
            torch.tensor(rewards, dtype=torch.float32).detach().to(self.device)
        )
        self.rewards[self.idx] = temp_tensor
        del temp_tensor

        temp_tensor = (
            torch.tensor(next_obs[0], dtype=torch.float32).detach().to(self.device)
        )
        self.next_states_img[self.idx] = temp_tensor
        del temp_tensor
        temp_tensor = (
            torch.tensor(next_obs[1], dtype=torch.float32).detach().to(self.device)
        )
        self.next_states[self.idx] = temp_tensor

        del temp_tensor

        temp_tensor = torch.tensor(dones, dtype=torch.float32).detach().to(self.device)
        self.dones[self.idx] = temp_tensor
        del temp_tensor

        self.idx += 1

        if self.idx == self.capacity:
            self.filled = True

        self.idx = self.idx % self.capacity

    def can_sample(self):
        res = False
        # if len(self) >= self.capacity:
        if len(self) >= self.min_number_samples:
            res = True
        # print(f"{len(self)} collected")
        return res

    @torch.no_grad()
    def sample(self, sample_capacity=None, device="cpu"):
        if self.can_sample():
            if sample_capacity:
                idx = np.random.randint(0, len(self), sample_capacity)

            else:
                idx = np.random.randint(0, len(self), self.batch_size)

            self.indices[:] = idx

            rewards = self.rewards[self.indices].to(device)

            if self.normalize_rewards is True:
                rewards = (rewards - self.rewards.min()) / (
                    self.rewards.max() - self.rewards.min()
                )
            return (
                (
                    self.states_img[self.indices].to(device),
                    self.states[self.indices].to(device),
                ),
                self.actions[self.indices].to(device),
                rewards,
                (
                    self.next_states_img[self.indices].to(device),
                    self.next_states[self.indices].to(device),
                ),
                self.dones[self.indices].to(device),
            )
        else:
            assert "Can't sample: not enough elements!"

    def __len__(self):
        size = 0
        if self.filled:
            size = self.dones.shape[0]
        else:
            size = self.idx
        return size

    @torch.no_grad()
    def increase_capacity(self, new_capacity):
        assert new_capacity > self.capacity
        increment = new_capacity - self.capacity
        self.states_img = torch.cat(
            (
                self.states_img,
                torch.empty((increment, *self.state_shape), dtype=torch.float32)
                .detach()
                .to(self.device),
            )
        )
        self.actions = torch.cat(
            (
                self.actions,
                torch.empty((increment, *self.action_shape), dtype=torch.float32)
                .detach()
                .to(self.device),
            )
        ).to(self.device)

        self.rewards = torch.cat(
            (
                self.rewards,
                torch.empty(increment, dtype=torch.float32).detach().to(self.device),
            )
        ).to(self.device)

        self.next_states_img = torch.cat(
            (
                self.next_states_img,
                torch.empty((increment, *self.state_shape), dtype=torch.float32)
                .detach()
                .to(self.device),
            )
        ).to(self.device)

        self.dones = torch.cat(
            (
                self.dones,
                torch.empty(increment, dtype=torch.bool).detach().to(self.device),
            )
        ).to(self.device)
        self.filled = False
        self.idx = self.capacity
        self.capacity = new_capacity


class ReplayBuffer:

    @torch.no_grad()
    def __init__(
        self,
        capacity=10_000,
        batch_size=32,
        state_shape=(4, 120, 160),
        action_shape=(1, 2),
        device="cpu",
        normalize_rewards=False,
    ):
        self.device = device
        # self.content = []
        self.state_shape = state_shape

        self.capacity = capacity
        self.idx = 0
        self.filled = False
        self.batch_size = batch_size
        self.indices = np.zeros(batch_size)
        self.normalize_rewards = normalize_rewards
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.states_img = (
            torch.empty((capacity, *state_shape), dtype=torch.float32)
            .detach()
            .to(self.device)
        )
        self.actions = (
            torch.empty((capacity, *action_shape), dtype=torch.float32)
            .detach()
            .to(self.device)
        )
        self.rewards = (
            torch.empty(capacity, dtype=torch.float32).detach().to(self.device)
        )
        self.next_states_img = (
            torch.empty((capacity, *state_shape), dtype=torch.float32)
            .detach()
            .to(self.device)
        )
        self.dones = torch.empty(capacity, dtype=torch.bool).detach().to(self.device)

    def change_bacth_size(self, batch_size):
        self.batch_size = batch_size
        del self.indices
        self.indices = np.zeros(batch_size)

    def save(self, filename="replay_buffer"):
        print("saving replay buffer...")
        file = open(filename, "wb")
        pickle.dump(self, file, 4)
        file.close()
        print("Done")

    @torch.no_grad()
    def add(self, obs, next_obs, actions, rewards, dones):
        temp_tensor = torch.tensor(obs, dtype=torch.float32).detach().to(self.device)
        size_sample = temp_tensor.shape[0]
        interval = [self.idx, 0]
        if self.idx + size_sample < self.capacity:
            interval[1] = self.idx + size_sample
        else:
            interval[1] = None  # size_sample + interval[0]

        self.states_img[self.idx] = temp_tensor
        del temp_tensor

        temp_tensor = (
            torch.tensor(actions, dtype=torch.float32).detach().to(self.device)
        )
        self.actions[self.idx] = temp_tensor
        del temp_tensor

        temp_tensor = (
            torch.tensor(rewards, dtype=torch.float32).detach().to(self.device)
        )
        self.rewards[self.idx] = temp_tensor
        del temp_tensor

        temp_tensor = (
            torch.tensor(next_obs, dtype=torch.float32).detach().to(self.device)
        )
        self.next_states_img[self.idx] = temp_tensor
        del temp_tensor

        temp_tensor = torch.tensor(dones, dtype=torch.float32).detach().to(self.device)
        self.dones[self.idx] = temp_tensor
        del temp_tensor

        if self.idx + size_sample == self.capacity:
            self.filled = True

        self.idx = (self.idx + size_sample) % self.capacity

    def can_sample(self):
        res = False
        # if len(self) >= self.capacity:
        if len(self) >= self.batch_size * 2:
            res = True
        # print(f"{len(self)} collected")
        return res

    @torch.no_grad()
    def sample(self, sample_capacity=None, device="cpu"):
        if self.can_sample():
            if sample_capacity:
                idx = np.random.randint(0, len(self), sample_capacity)

            else:
                idx = np.random.randint(0, len(self), self.batch_size)

            self.indices[:] = idx

            rewards = self.rewards[self.indices].to(device)

            if self.normalize_rewards is True:
                rewards = (rewards - self.rewards.min()) / (
                    self.rewards.max() - self.rewards.min()
                )
            return (
                self.states_img[self.indices].to(device),
                self.actions[self.indices].to(device),
                rewards,
                self.next_states_img[self.indices].to(device),
                self.dones[self.indices].to(device),
            )
        else:
            assert "Can't sample: not enough elements!"

    def __len__(self):
        size = 0
        if self.filled:
            size = self.dones.shape[0]
        else:
            size = self.idx
        return size

    @torch.no_grad()
    def increase_capacity(self, new_capacity):
        assert new_capacity > self.capacity
        increment = new_capacity - self.capacity
        self.states_img = torch.cat(
            (
                self.states_img,
                torch.empty((increment, *self.state_shape), dtype=torch.float32)
                .detach()
                .to(self.device),
            )
        )
        self.actions = torch.cat(
            (
                self.actions,
                torch.empty((increment, *self.action_shape), dtype=torch.float32)
                .detach()
                .to(self.device),
            )
        ).to(self.device)

        self.rewards = torch.cat(
            (
                self.rewards,
                torch.empty(increment, dtype=torch.float32).detach().to(self.device),
            )
        ).to(self.device)

        self.next_states_img = torch.cat(
            (
                self.next_states_img,
                torch.empty((increment, *self.state_shape), dtype=torch.float32)
                .detach()
                .to(self.device),
            )
        ).to(self.device)

        self.dones = torch.cat(
            (
                self.dones,
                torch.empty(increment, dtype=torch.bool).detach().to(self.device),
            )
        ).to(self.device)
        self.filled = False
        self.idx = self.capacity
        self.capacity = new_capacity


@torch.no_grad()
def saturate_replay_buffer(replay_buffer):
    print("Saturating replay buffer...")

    cursor = replay_buffer.idx
    while cursor < replay_buffer.capacity:

        if cursor * 2 < replay_buffer.capacity:
            x = cursor
        else:
            x = replay_buffer.capacity - cursor

        replay_buffer.states_img[cursor : cursor + x] = replay_buffer.states_img[:x]
        replay_buffer.actions[cursor : cursor + x] = replay_buffer.actions[:x]
        replay_buffer.rewards[cursor : cursor + x] = replay_buffer.rewards[:x]
        replay_buffer.next_states_img[cursor : cursor + x] = (
            replay_buffer.next_states_img[:x]
        )
        replay_buffer.dones[cursor : cursor + x] = replay_buffer.dones[:x]

        cursor += x
    replay_buffer.idx = cursor
    print("Done")


import configparser
import time


def read_file_if_modified(args, file_path, last_mod_time):
    # Get the current modification time of the file
    current_mod_time = os.path.getmtime(file_path)

    # If the file has been modified since the last read
    if current_mod_time != last_mod_time:
        with open(file_path, "r") as file:
            args = parse_arguments_from_ini(file_path)

        print("modifications in config.ini ")
        # Update the last modification time
        return current_mod_time, args, True

    return last_mod_time, args, False


def parse_arguments_from_ini(file_path):
    config = configparser.ConfigParser()
    config.read_file(open(file_path))

    arguments = {}

    for section, cfg in config.items():
        for key, value in config[section].items():
            print("Parsing the key: ", key)
            if value.lower() in ["none", "null"]:
                value = None
            elif value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            # elif "/" in value:
            #     pass
            elif "." in value:
                value = float(value)
            elif "'" in value or '"' in value:
                value = value[1:-1]
            else:
                try:
                    value = int(value)
                except:
                    # is a string
                    pass

            arguments[key] = value

    return arguments


def evaluate_policy(env, policy, eval_episodes=10, max_timesteps=500):
    avg_reward = 0.0
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        step = 0
        while not done and step < max_timesteps:
            action = policy.predict(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
            step += 1

    avg_reward /= eval_episodes

    return avg_reward
