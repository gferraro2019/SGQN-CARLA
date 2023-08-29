import os
import random
from collections import deque

import dmc2gym
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from dmc2gym.wrappers import DMCWrapper
from numpy.random import randint
from PIL import Image

import utils


def make_env(
    domain_name,
    task_name,
    seed=0,
    episode_length=1000,
    frame_stack=3,
    action_repeat=4,
    image_size=100,
    mode="train",
    intensity=0.0,
):
    """Make environment for experiments"""
    assert mode in {
        "train",
        "color_easy",
        "color_hard",
        "video_easy",
        "video_hard",
        "distracting_cs",
        "all",
    }, f'specified mode "{mode}" is not supported'

    paths = []
    is_distracting_cs = mode == "distracting_cs"
    if is_distracting_cs:
        import env.distracting_control.suite as dc_suite

        loaded_paths = [
            os.path.join(dir_path, "DAVIS/JPEGImages/480p")
            for dir_path in utils.load_config("datasets")
        ]
        for path in loaded_paths:
            if os.path.exists(path):
                paths.append(path)
    env = dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
        seed=seed,
        visualize_reward=False,
        from_pixels=True,
        height=image_size,
        width=image_size,
        episode_length=episode_length,
        frame_skip=action_repeat,
        is_distracting_cs=is_distracting_cs,
        distracting_cs_intensity=intensity,
        background_dataset_paths=paths,
    )
    if not is_distracting_cs:
        env = VideoWrapper(env, mode, seed)
    env = FrameStack(env, frame_stack)
    if not is_distracting_cs:
        env = ColorWrapper(env, mode, seed)

    return env


class ColorWrapper(gym.Wrapper):
    """Wrapper for the color experiments"""

    def __init__(self, env, mode, seed=None):
        # assert isinstance(env, FrameStack), "wrapped env must be a framestack"
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps
        self._mode = mode
        self._random_state = np.random.RandomState(seed)
        self.time_step = 0
        if "color" in self._mode:
            self._load_colors()

    def reset(self):
        self.time_step = 0
        if "color" in self._mode:
            self.randomize()
        elif "video" in self._mode:
            # apply greenscreen
            setting_kwargs = {
                "skybox_rgb": [0.2, 0.8, 0.2],
                "skybox_rgb2": [0.2, 0.8, 0.2],
                "skybox_markrgb": [0.2, 0.8, 0.2],
            }
            if self._mode == "video_hard":
                setting_kwargs["grid_rgb1"] = [0.2, 0.8, 0.2]
                setting_kwargs["grid_rgb2"] = [0.2, 0.8, 0.2]
                setting_kwargs["grid_markrgb"] = [0.2, 0.8, 0.2]
            self.reload_physics(setting_kwargs)
        return self.env.reset()

    def step(self, action):
        self.time_step += 1
        return self.env.step(action)

    def randomize(self):
        assert (
            "color" in self._mode
        ), f"can only randomize in color mode, received {self._mode}"
        self.reload_physics(self.get_random_color())

    def _load_colors(self):
        assert self._mode in {"color_easy", "color_hard"}
        self._colors = torch.load(f"src/env/data/{self._mode}.pt")

    def get_random_color(self):
        assert len(self._colors) >= 100, "env must include at least 100 colors"
        return self._colors[self._random_state.randint(len(self._colors))]

    def reload_physics(self, setting_kwargs=None, state=None):
        from dm_control.suite import common

        domain_name = self._get_dmc_wrapper()._domain_name
        if setting_kwargs is None:
            setting_kwargs = {}
        if state is None:
            state = self._get_state()
        self._reload_physics(
            *common.settings.get_model_and_assets_from_setting_kwargs(
                domain_name + ".xml", setting_kwargs
            )
        )
        self._set_state(state)

    def get_state(self):
        return self._get_state()

    def set_state(self, state):
        self._set_state(state)

    def _get_dmc_wrapper(self):
        _env = self.env
        while not isinstance(_env, DMCWrapper) and hasattr(_env, "env"):
            _env = _env.env
        assert isinstance(_env, DMCWrapper), "environment is not dmc2gym-wrapped"

        return _env

    def _reload_physics(self, xml_string, assets=None):
        _env = self.env
        while not hasattr(_env, "_physics") and hasattr(_env, "env"):
            _env = _env.env
        assert hasattr(_env, "_physics"), "environment does not have physics attribute"
        _env.physics.reload_from_xml_string(xml_string, assets=assets)

    def _get_physics(self):
        _env = self.env
        while not hasattr(_env, "_physics") and hasattr(_env, "env"):
            _env = _env.env
        assert hasattr(_env, "_physics"), "environment does not have physics attribute"

        return _env._physics

    def _get_state(self):
        return self._get_physics().get_state()

    def _set_state(self, state):
        self._get_physics().set_state(state)


# from numpy import asarray


class ColorWrapper_carla(gym.Wrapper):
    """Wrapper for the color experiments"""

    def __init__(
        self, env, mode, alpha=0.5, n_frames=9, path_to_dataset=None, seed=None
    ):
        # assert isinstance(env, FrameStack), "wrapped env must be a framestack"
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps
        self._mode = mode
        self._random_state = np.random.RandomState(seed)
        self.alpha = alpha
        self.n_frames = n_frames
        if path_to_dataset is None:
            self.path_to_dataset = os.path.join(
                __file__[:-19], "datasets", "noisy_dataset_carla"
            )

    # def reset(self):
    #     if "color" in self._mode:
    #         self.randomize()
    #     elif "video" in self._mode:
    #         # apply greenscreen
    #         setting_kwargs = {
    #             "skybox_rgb": [0.2, 0.8, 0.2],
    #             "skybox_rgb2": [0.2, 0.8, 0.2],
    #             "skybox_markrgb": [0.2, 0.8, 0.2],
    #         }
    #         if self._mode == "video_hard":
    #             setting_kwargs["grid_rgb1"] = [0.2, 0.8, 0.2]
    #             setting_kwargs["grid_rgb2"] = [0.2, 0.8, 0.2]
    #             setting_kwargs["grid_markrgb"] = [0.2, 0.8, 0.2]

    #     return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        frames_from_dataset = self.sample_frames_from_dataset()
        for i, frame in enumerate(obs.frames):
            obs.frames[i] = (
                self.alpha * frame + (1 - self.alpha) * frames_from_dataset[i]
            )
            obs.frames[i] = obs.frames[i].astype(np.int16)
            # plt.imshow(obs.frames[i].reshape((84, 84, 3)))
            # plt.show()
        return obs, reward, done, info

    def sample_frames_from_dataset(self):
        files = os.listdir(os.path.join(self.path_to_dataset))

        # permute list
        files = random.sample(files, self.n_frames)

        datasets = []
        for f in files:
            frame = np.load(os.path.join(self.path_to_dataset, f))
            datasets.append(frame)

        return datasets


class FrameStack(gym.Wrapper):
    """Stack frames as observation"""

    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype,
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return utils.LazyFrames(list(self._frames))


from gym import spaces


class FrameStack_carla(gym.Wrapper):
    """Stack frames as observation"""

    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)

        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space[0].shape

        self.observation_space = spaces.Tuple(
            (
                spaces.Box(0, 255, shape=((shp[0] * k,) + shp[1:]), dtype=np.uint8),
                spaces.Box(-np.inf, np.inf, shape=(2,)),
            )
        )

        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        # obs = obs.reshape((3, 84, 84))
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # obs = obs.reshape((3, 84, 84))
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return utils.LazyFrames(list(self._frames))


import os
import os.path as op

import pygame
from PIL import Image


class VideoRecord_carla(gym.Wrapper):
    """Stack frames as observation"""

    def __init__(self, env, name_algorithm, n_episodes=100):
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.count_frame = 0
        self.episode = -1
        self.paths = [
            op.join("output", "video_records", "camera"),
            op.join("output", "video_records", "display"),
        ]
        self.name_algorithm = name_algorithm
        for path in self.paths:
            if not op.exists(path):
                os.makedirs(path)

        self.filename = ""
        self._max_episode_steps = env._max_episode_steps
        self.n_episodes = n_episodes

    def reset(self):
        self.episode += 1
        self.count_frame = 0
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # if self.episode % self.n_episodes == 0:
        self.update_filename(self.paths[0])
        self.save_image(self.filename, obs[0].reshape(84, 84, 3))
        if self.render_display is not False:
            self.update_filename(self.paths[1])
            pygame.image.save(self.render_display, self.filename)
        self.count_frame += 1

        return obs, reward, done, info

    def update_filename(self, path):
        self.filename = op.join(
            path,
            self.name_algorithm
            + "_"
            + str(self.episode)
            + "_"
            + str(self.count_frame)
            + ".png",
        )

    def save_image(self, filename, array, width=800, heigth=600):
        im = Image.fromarray(array, "RGB")
        im = im.resize((width, heigth))
        im.save(filename)


def rgb_to_hsv(r, g, b):
    """Convert RGB color to HSV color"""
    maxc = max(r, g, b)
    minc = min(r, g, b)
    v = maxc
    if minc == maxc:
        return 0.0, 0.0, v
    s = (maxc - minc) / maxc
    rc = (maxc - r) / (maxc - minc)
    gc = (maxc - g) / (maxc - minc)
    bc = (maxc - b) / (maxc - minc)
    if r == maxc:
        h = bc - gc
    elif g == maxc:
        h = 2.0 + rc - bc
    else:
        h = 4.0 + gc - rc
    h = (h / 6.0) % 1.0
    return h, s, v


def do_green_screen(x, bg):
    """Removes green background from observation and replaces with bg; not optimized for speed"""
    assert isinstance(x, np.ndarray) and isinstance(
        bg, np.ndarray
    ), "inputs must be numpy arrays"
    assert x.dtype == np.uint8 and bg.dtype == np.uint8, "inputs must be uint8 arrays"

    # Get image sizes
    x_h, x_w = x.shape[1:]

    # Convert to RGBA images
    im = TF.to_pil_image(torch.ByteTensor(x))
    im = im.convert("RGBA")
    pix = im.load()
    bg = TF.to_pil_image(torch.ByteTensor(bg))
    bg = bg.convert("RGBA")
    bg = bg.load()

    # Replace pixels
    for x in range(x_w):
        for y in range(x_h):
            r, g, b, a = pix[x, y]
            h_ratio, s_ratio, v_ratio = rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            h, s, v = (h_ratio * 360, s_ratio * 255, v_ratio * 255)

            min_h, min_s, min_v = (100, 80, 70)
            max_h, max_s, max_v = (185, 255, 255)
            if min_h <= h <= max_h and min_s <= s <= max_s and min_v <= v <= max_v:
                pix[x, y] = bg[x, y]

    return np.moveaxis(np.array(im).astype(np.uint8), -1, 0)[:3]


class VideoWrapper(gym.Wrapper):
    """Green screen for video experiments"""

    def __init__(self, env, mode, seed):
        gym.Wrapper.__init__(self, env)
        self._mode = mode
        self._seed = seed
        self._random_state = np.random.RandomState(seed)
        self._index = 0
        self._video_paths = []
        if "video" in mode:
            self._get_video_paths()
        self._num_videos = len(self._video_paths)
        self._max_episode_steps = env._max_episode_steps

    def _get_video_paths(self):
        video_dir = os.path.join("src/env/data", self._mode)
        if "video_easy" in self._mode:
            self._video_paths = [
                os.path.join(video_dir, f"video{i}.mp4") for i in range(10)
            ]
        elif "video_hard" in self._mode:
            self._video_paths = [
                os.path.join(video_dir, f"video{i}.mp4") for i in range(100)
            ]
        else:
            raise ValueError(f'received unknown mode "{self._mode}"')

    def _load_video(self, video):
        """Load video from provided filepath and return as numpy array"""
        import cv2

        cap = cv2.VideoCapture(video)
        assert (
            cap.get(cv2.CAP_PROP_FRAME_WIDTH) >= 100
        ), "width must be at least 100 pixels"
        assert (
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT) >= 100
        ), "height must be at least 100 pixels"
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        buf = np.empty(
            (
                n,
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                3,
            ),
            np.dtype("uint8"),
        )
        i, ret = 0, True
        while i < n and ret:
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buf[i] = frame
            i += 1
        cap.release()
        return np.moveaxis(buf, -1, 1)

    def _reset_video(self):
        self._index = (self._index + 1) % self._num_videos
        self._data = self._load_video(self._video_paths[self._index])

    def reset(self):
        if "video" in self._mode:
            self._reset_video()
        self._current_frame = 0
        return self._greenscreen(self.env.reset())

    def step(self, action):
        self._current_frame += 1
        obs, reward, done, info = self.env.step(action)
        return self._greenscreen(obs), reward, done, info

    def _interpolate_bg(self, bg, size: tuple):
        """Interpolate background to size of observation"""
        bg = torch.from_numpy(bg).float().unsqueeze(0) / 255.0
        bg = F.interpolate(bg, size=size, mode="bilinear", align_corners=False)
        return (bg * 255.0).byte().squeeze(0).numpy()

    def _greenscreen(self, obs):
        """Applies greenscreen if video is selected, otherwise does nothing"""
        if "video" in self._mode:
            bg = self._data[self._current_frame % len(self._data)]  # select frame
            bg = self._interpolate_bg(bg, obs.shape[1:])  # scale bg to observation size
            return do_green_screen(obs, bg)  # apply greenscreen
        return obs

    def apply_to(self, obs):
        """Applies greenscreen mode of object to observation"""
        obs = obs.copy()
        channels_last = obs.shape[-1] == 3
        if channels_last:
            obs = torch.from_numpy(obs).permute(2, 0, 1).numpy()
        obs = self._greenscreen(obs)
        if channels_last:
            obs = torch.from_numpy(obs).permute(1, 2, 0).numpy()
        return obs
