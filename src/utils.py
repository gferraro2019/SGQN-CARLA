import glob
import json
import os
import random
import subprocess
from datetime import datetime

import numpy as np
import pygame
import torch

import augmentations


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def cat(x, y, axis=0):
    return torch.cat([x, y], axis=0)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def write_info(args, fp):
    data = {
        "timestamp": str(datetime.now()),
        "git": subprocess.check_output(["git", "describe", "--always"])
        .strip()
        .decode(),
        "args": vars(args),
    }
    with open(fp, "w") as f:
        json.dump(data, f, indent=4, separators=(",", ": "))


def load_config(key=None):
    path = os.path.join("setup", "config.cfg")
    with open(path) as f:
        data = json.load(f)
    if key is not None:
        return data[key]
    return data


def make_dir(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def listdir(dir_path, filetype="jpg", sort=True):
    fpath = os.path.join(dir_path, f"*.{filetype}")
    fpaths = glob.glob(fpath, recursive=True)
    if sort:
        return sorted(fpaths)
    return fpaths


def prefill_memory(obses, capacity, obs_shape):
    """Reserves memory for replay buffer"""
    c, h, w = obs_shape
    for _ in range(capacity):
        frame = np.ones((3, h, w), dtype=np.uint8)
        obses.append(frame)
    return obses


class ReplayBuffer(object):
    """Buffer to store environment transitions"""

    def __init__(self, obs_shape, action_shape, capacity, batch_size, prefill=True):
        self.capacity = capacity
        self.batch_size = batch_size

        self._obses = []
        if prefill:
            self._obses = prefill_memory(self._obses, capacity, obs_shape)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        obses = (obs, next_obs)
        if self.idx >= len(self._obses):
            self._obses.append(obses)
        else:
            self._obses[self.idx] = obses
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def _get_idxs(self, n=None):
        if n is None:
            n = self.batch_size
        return np.random.randint(0, self.capacity if self.full else self.idx, size=n)

    def _encode_obses(self, idxs):
        obses, next_obses = [], []
        for i in idxs:
            obs, next_obs = self._obses[i]
            obses.append(np.array(obs, copy=False))
            next_obses.append(np.array(next_obs, copy=False))
        return np.array(obses), np.array(next_obses)

    def sample_soda(self, n=None):
        idxs = self._get_idxs(n)
        obs, _ = self._encode_obses(idxs)
        return torch.as_tensor(obs).cuda().float()

    def sample_curl(self, n=None):
        idxs = self._get_idxs(n)

        obs, next_obs = self._encode_obses(idxs)
        obs = torch.as_tensor(obs).cuda().float()
        next_obs = torch.as_tensor(next_obs).cuda().float()
        actions = torch.as_tensor(self.actions[idxs]).cuda()
        rewards = torch.as_tensor(self.rewards[idxs]).cuda()
        not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

        pos = augmentations.random_crop(obs.clone())
        obs = augmentations.random_crop(obs)
        next_obs = augmentations.random_crop(next_obs)

        return obs, actions, rewards, next_obs, not_dones, pos

    def sample_drq(self, n=None, pad=4):
        idxs = self._get_idxs(n)

        obs, next_obs = self._encode_obses(idxs)
        obs = torch.as_tensor(obs).cuda().float()
        next_obs = torch.as_tensor(next_obs).cuda().float()
        actions = torch.as_tensor(self.actions[idxs]).cuda()
        rewards = torch.as_tensor(self.rewards[idxs]).cuda()
        not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

        obs = augmentations.random_shift(obs, pad)
        next_obs = augmentations.random_shift(next_obs, pad)

        return obs, actions, rewards, next_obs, not_dones

    def sample_sacai(self, n=None, pad=4):
        idxs = self._get_idxs(n)

        obs, next_obs = self._encode_obses(idxs)
        obs = torch.as_tensor(obs).cuda().float()
        next_obs = torch.as_tensor(next_obs).cuda().float()
        actions = torch.as_tensor(self.actions[idxs]).cuda()
        rewards = torch.as_tensor(self.rewards[idxs]).cuda()
        not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

        return obs, actions, rewards, next_obs, not_dones

    def sample(self, n=None):
        idxs = self._get_idxs(n)

        obs, next_obs = self._encode_obses(idxs)
        obs = torch.as_tensor(obs).cuda().float()
        next_obs = torch.as_tensor(next_obs).cuda().float()
        actions = torch.as_tensor(self.actions[idxs]).cuda()
        rewards = torch.as_tensor(self.rewards[idxs]).cuda()
        not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

        obs = augmentations.random_crop(obs)
        next_obs = augmentations.random_crop(next_obs)

        return obs, actions, rewards, next_obs, not_dones


class LazyFrames(object):
    def __init__(self, frames, extremely_lazy=True):
        self._frames = frames
        self._extremely_lazy = extremely_lazy
        self._out = None

    @property
    def frames(self):
        return self._frames

    def _force(self):
        if self._extremely_lazy:
            return np.concatenate(self._frames, axis=0)
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        if self._extremely_lazy:
            return len(self._frames)
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        if self.extremely_lazy:
            return len(self._frames)
        frames = self._force()
        return frames.shape[0] // 3

    def frame(self, i):
        return self._force()[i * 3 : (i + 1) * 3]


def count_parameters(net, as_int=False):
    """Returns total number of params in a network"""
    count = sum(p.numel() for p in net.parameters())
    if as_int:
        return count
    return f"{count:,}"


## CARLA
def vector_to_scalar(vector):
    scalar = np.around(np.sqrt(vector.x**2 + vector.y**2 + vector.z**2), 2)
    return scalar


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = "ubuntumono"
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def get_actor_name(actor, truncate=250):
    name = " ".join(actor.type_id.replace("_", ".").title().split(".")[1:])
    return (name[: truncate - 1] + "\u2026") if len(name) > truncate else name


def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))


def load_dataset_for_carla(max_episode_steps=20000, n_frames=10000):
    from carla_wrapper import CarlaEnv

    path_to_dataset = os.path.join(__file__[:-13], "datasets", "carla")
    count_frame = len(os.listdir(os.path.join(path_to_dataset)))
    if count_frame < n_frames:
        print("collecting frame for dataset...")
        # collect 1000 pictures from high realistic Carla env with autopilot
        env = CarlaEnv(
            True,
            2000,
            0,
            1,
            "pixel",
            True,
            "tesla.cybertruck",
            None,
            True,
            None,
            max_episode_steps,
        )

        while count_frame < n_frames:
            env.reset()
            done = False
            iteration = 0
            while not done:
                next_obs, reward, done, info = env.step([1, 0])
                iteration += 1
                if iteration % 10 == 0:
                    np.save(
                        os.path.join(path_to_dataset, f"frame_{count_frame}"), next_obs
                    )
                    count_frame += 1
                if count_frame >= n_frames:
                    break

        env.close()

        print(f"... {count_frame} collected.")
    else:
        print(f"Noisy Dataset Carla alredy full.")


import os
import sys  # We need sys so that we can pass argv to QApplication
from random import randint

import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
from pyqtgraph import PlotWidget, plot


class MainWindow_Reward(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow_Reward, self).__init__(*args, **kwargs)

        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)

        self.x = np.zeros(100)
        self.y = np.zeros(100)
        self.idx = 0

        self.graphWidget.setBackground("b")

        pen = pg.mkPen(color=(255, 255, 255))
        self.data_line = self.graphWidget.plot(self.x, self.y, pen=pen)

        # self.timer = QtCore.QTimer()
        # self.timer.setInterval(1000)
        # self.timer.timeout.connect(self.update_plot_data)
        # self.timer.start()

    def update_plot_data(self, step, reward):
        if self.idx < 100:
            self.x[self.idx :] = step
            self.y[self.idx] = reward
            self.idx += 1
        else:
            self.x = np.roll(self.x, -1)
            self.x[-1] = step
            self.y = np.roll(self.y, -1)
            self.y[-1] = reward
        self.data_line.setData(self.x, self.y)  # Update the data.


import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget


class MainWindow_Tot_Reward(QMainWindow):
    def __init__(self):
        super(MainWindow_Tot_Reward, self).__init__()

        self.episode = 0
        self.tot_reward = 0
        self.action = [0, 0]
        self.frame = 0

        self.setWindowTitle("My App")
        widget = QWidget()
        layout = QVBoxLayout()

        label1 = QLabel("Current Total Reward Episode:")
        font = label1.font()
        font.setPointSize(20)
        label1.setFont(font)
        label1.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self.label2 = QLabel(str(self.tot_reward))
        font = self.label2.font()
        font.setPointSize(30)
        self.label2.setFont(font)
        self.label2.setAlignment(Qt.AlignTop | Qt.AlignRight)

        label3 = QLabel("#Episode:")
        font = label3.font()
        font.setPointSize(20)
        label3.setFont(font)
        label3.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self.label4 = QLabel(str(self.episode))
        font = self.label4.font()
        font.setPointSize(30)
        self.label4.setFont(font)
        self.label4.setAlignment(Qt.AlignTop | Qt.AlignRight)

        label5 = QLabel("(+)Throttle/(-)Brake:")
        font = label5.font()
        font.setPointSize(20)
        label5.setFont(font)
        label5.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self.label6 = QLabel(str(self.action[0]))
        font = self.label6.font()
        font.setPointSize(30)
        self.label6.setFont(font)
        self.label6.setAlignment(Qt.AlignTop | Qt.AlignRight)

        label7 = QLabel("#Steer:")
        font = label7.font()
        font.setPointSize(20)
        label7.setFont(font)
        label7.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self.label8 = QLabel(str(self.action[1]))
        font = self.label8.font()
        font.setPointSize(30)
        self.label8.setFont(font)
        self.label8.setAlignment(Qt.AlignTop | Qt.AlignRight)

        label9 = QLabel("#Frame:")
        font = label9.font()
        font.setPointSize(20)
        label9.setFont(font)
        label9.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self.label10 = QLabel(str(self.frame))
        font = self.label10.font()
        font.setPointSize(30)
        self.label10.setFont(font)
        self.label10.setAlignment(Qt.AlignTop | Qt.AlignRight)

        layout.addWidget(label3)  # episode
        layout.addWidget(self.label4)
        layout.addWidget(label9)  # frame
        layout.addWidget(self.label10)
        layout.addWidget(label1)  # tot reward
        layout.addWidget(self.label2)
        layout.addWidget(label5)  # throttle
        layout.addWidget(self.label6)
        layout.addWidget(label7)  # steer
        layout.addWidget(self.label8)

        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def update_labels(self, n_episode, reward, action):
        self.tot_reward += reward
        self.action = action
        self.frame += 1
        self.label2.setText(str(self.tot_reward))
        self.label4.setText(str(n_episode))
        self.label6.setText(str(self.action[0]))
        self.label8.setText(str(self.action[1]))
        self.label10.setText(str(self.frame))

    def reset_tot_reward(self):
        self.tot_reward = 0
        self.action = [0, 0]
        self.frame = 0


import os

import cv2
import numpy as np


def images_to_video(episode, path_images, save_path, fps=20, width=800, height=600):
    # Crea un oggetto VideoWriter per scrivere il video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(
        os.path.join(save_path, f"episode_{str(episode)}.mp4"),
        fourcc,
        fps,
        (width, height),
    )

    for path_img in path_images:
        #     print(path_img)
        image = cv2.imread(path_img)
        video.write(image)

    # Rilascia le risorse
    video.release()


def create_video_from_images(n_episodes, lenght_episode, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for episode in range(n_episodes):
        path_images = []
        for frame in range(lenght_episode):
            path_images.append(
                os.path.join(
                    "output",
                    "video_records",
                    "sgsac_" + str(episode) + "_" + str(frame) + ".png",
                )
            )

        images_to_video(episode, path_images, save_path)
