#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import os
import random
import sys
import time

import gym
import pygame
import numpy as np

from gym import spaces

# from agents.navigation.roaming_agent import RoamingAgent

try:
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass

import carla


# In[2]:


class CarlaEnv(gym.Env):
    def __init__(
        self,
        render,
        carla_port,
        changing_weather_speed,
        frame_skip,
        observations_type,
        traffic,
        vehicle_name,
        map_name,
        autopilot,
        unload_map_layer=None,
        max_episode_steps=1000,
    ):
        super(CarlaEnv, self).__init__()
        self.render_display = render
        self.changing_weather_speed = float(changing_weather_speed)
        self.frame_skip = frame_skip
        self.observations_type = observations_type
        self.traffic = traffic
        self.vehicle_name = vehicle_name
        self.map_name = map_name
        self.autopilot = autopilot
        self.actor_list = []
        self.count = 0
        print(max_episode_steps)
        self._max_episode_steps = int(max_episode_steps)
        self.time_step = 0

        # initialize renderingAttributeError: module 'tensorflow' has no attribute 'contrib'
        if self.render_display:
            pygame.init()
            self.render_display = pygame.display.set_mode(
                (800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF
            )
            self.font = get_font()
            self.clock = pygame.time.Clock()

        # initialize client with timeout
        self.client = carla.Client("localhost", carla_port)
        self.client.set_timeout(4.0)

        # initialize world and map
        if self.map_name is not None:
            self.world = self.client.load_world_world(self.map_name)
        else:
            self.world = self.client.get_world()

        self.map = self.world.get_map()

        # unload map layers
        if unload_map_layer is not None:
            if unload_map_layer == "All":
                self.world.unload_map_layer(carla.MapLayer.All)

        # remove old vehicles and sensors (in case they survived)
        self.world.tick()
        actor_list = self.world.get_actors()
        for vehicle in actor_list.filter("*vehicle*"):
            print("Warning: removing old vehicle")
            vehicle.destroy()
        for sensor in actor_list.filter("*sensor*"):
            print("Warning: removing old sensor")
            sensor.destroy()

        # create vehicle
        self.vehicle = None
        self.vehicles_list = []
        self._reset_vehicle()
        self.actor_list.append(self.vehicle)

        # initialize blueprint library
        blueprint_library = self.world.get_blueprint_library()

        # spawn camera for rendering
        if self.render_display:
            self.camera_display = self.world.spawn_actor(
                blueprint_library.find("sensor.camera.rgb"),
                carla.Transform(
                    carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)
                ),
                attach_to=self.vehicle,
            )
            self.actor_list.append(self.camera_display)

        # spawn camera for pixel observations
        if self.observations_type == "pixel":
            bp = blueprint_library.find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(84))
            bp.set_attribute("image_size_y", str(84))
            bp.set_attribute("fov", str(84))
            location = carla.Location(x=1.6, z=1.7)
            self.camera_vision = self.world.spawn_actor(
                bp,
                carla.Transform(location, carla.Rotation(yaw=0.0)),
                attach_to=self.vehicle,
            )
            self.actor_list.append(self.camera_vision)

        # context manager initialization
        if self.render_display and self.observations_type == "pixel":
            self.sync_mode = CarlaSyncMode(
                self.world, self.camera_display, self.camera_vision, fps=20
            )
        elif self.render_display and self.observations_type == "state":
            self.sync_mode = CarlaSyncMode(self.world, self.camera_display, fps=20)
        elif not self.render_display and self.observations_type == "pixel":
            self.sync_mode = CarlaSyncMode(self.world, self.camera_vision, fps=20)
        elif not self.render_display and self.observations_type == "state":
            self.sync_mode = CarlaSyncMode(self.world, fps=20)
        else:
            raise ValueError("Unknown observation_type. Choose between: state, pixel")

        # weather
        self.weather = Weather(self.world, self.changing_weather_speed)

        # collision detection
        self.collision = False
        sensor_blueprint = self.world.get_blueprint_library().find(
            "sensor.other.collision"
        )
        self.collision_sensor = self.world.spawn_actor(
            sensor_blueprint, carla.Transform(), attach_to=self.vehicle
        )
        self.collision_sensor.listen(lambda event: self._on_collision(event))

        #         # initialize autopilot
        #         self.agent = RoamingAgent(self.vehicle)

        # get initial observation
        if self.observations_type == "state":
            obs = self._get_state_obs()
        else:
            obs = np.zeros((3, 84, 84))

        # gym environment specific variables
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype="float32")
        self.obs_dim = obs.shape
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=self.obs_dim, dtype="float32"
        )

    def reset(self):
        self._reset_vehicle()
        self.world.tick()
        self._reset_other_vehicles()
        self.world.tick()
        self.count = 0
        self.collision = False
        obs, _, _, _ = self.step([0, 0])
        self.time_step = 0
        return obs

    def _reset_vehicle(self):
        # choose random spawn point
        init_transforms = self.world.get_map().get_spawn_points()
        vehicle_init_transform = random.choice(init_transforms)

        # create the vehicle
        if self.vehicle is None:
            blueprint_library = self.world.get_blueprint_library()
            vehicle_blueprint = blueprint_library.find("vehicle." + self.vehicle_name)
            if vehicle_blueprint.has_attribute("color"):
                color = random.choice(
                    vehicle_blueprint.get_attribute("color").recommended_values
                )
                vehicle_blueprint.set_attribute("color", color)
            self.vehicle = self.world.spawn_actor(
                vehicle_blueprint, vehicle_init_transform
            )
        else:
            self.vehicle.set_transform(vehicle_init_transform)

    def _reset_other_vehicles(self):
        if not self.traffic:
            return

        # clear out old vehicles
        self.client.apply_batch(
            [carla.command.DestroyActor(x) for x in self.vehicles_list]
        )
        self.world.tick()
        self.vehicles_list = []

        # initialize traffic manager
        traffic_manager = self.client.get_trafficmanager()
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.global_percentage_speed_difference(30.0)
        blueprints = self.world.get_blueprint_library().filter("vehicle.*")
        blueprints = [
            x for x in blueprints if int(x.get_attribute("number_of_wheels")) == 4
        ]

        # choose random spawn points
        num_vehicles = 20
        init_transforms = self.world.get_map().get_spawn_points()
        init_transforms = np.random.choice(init_transforms, num_vehicles)

        # spawn vehicles
        batch = []
        for transform in init_transforms:
            transform.location.z += (
                0.1  # otherwise can collide with the road it starts on
            )
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute("color"):
                color = random.choice(
                    blueprint.get_attribute("color").recommended_values
                )
                blueprint.set_attribute("color", color)
            if blueprint.has_attribute("driver_id"):
                driver_id = random.choice(
                    blueprint.get_attribute("driver_id").recommended_values
                )
                blueprint.set_attribute("driver_id", driver_id)
            blueprint.set_attribute("role_name", "autopilot")
            batch.append(
                carla.command.SpawnActor(blueprint, transform).then(
                    carla.command.SetAutopilot(carla.command.FutureActor, True)
                )
            )

        for response in self.client.apply_batch_sync(batch, False):
            self.vehicles_list.append(response.actor_id)

        for response in self.client.apply_batch_sync(batch):
            if response.error:
                pass
            else:
                self.vehicles_list.append(response.actor_id)

    def _compute_action(self):
        return self.agent.run_step()

    def step(self, action):
        rewards = []
        next_obs, done, info = np.array([]), False, {}
        for _ in range(self.frame_skip):
            if self.autopilot:
                self.vehicle.set_autopilot(True)
                vehicle_control = self.vehicle.get_control()
                steer = float(vehicle_control.steer)
                if vehicle_control.throttle > 0.0 and vehicle_control.brake == 0.0:
                    throttle_brake = vehicle_control.throttle
                elif vehicle_control.brake > 0.0 and vehicle_control.throttle == 0.0:
                    throttle_brake = (
                        -vehicle_control.brake
                    )  # should be - vehicle_control.brake
                else:
                    throttle_brake = 0.0
                action = [throttle_brake, steer]
            next_obs, reward, done, info = self._simulator_step(action)
            rewards.append(reward)

            if done:
                break
        return next_obs.reshape(3, 84, 84), np.mean(rewards), done, info

    def _simulator_step(self, action):
        if self.render_display:
            if should_quit():
                return
            self.clock.tick()

        # calculate actions
        throttle_brake = float(action[0])
        steer = float(action[1])
        if throttle_brake >= 0.0:
            throttle = throttle_brake
            brake = 0.0
        else:
            throttle = 0.0
            brake = -throttle_brake

        # apply control to simulation
        vehicle_control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake),
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
        )

        self.vehicle.apply_control(vehicle_control)

        # advance the simulation and wait for the data
        if self.render_display and self.observations_type == "pixel":
            snapshot, display_image, vision_image = self.sync_mode.tick(timeout=2.0)
        elif self.render_display and self.observations_type == "state":
            snapshot, display_image = self.sync_mode.tick(timeout=2.0)
        elif not self.render_display and self.observations_type == "pixel":
            snapshot, vision_image = self.sync_mode.tick(timeout=2.0)
        elif not self.render_display and self.observations_type == "state":
            self.sync_mode.tick(timeout=2.0)
        else:
            raise ValueError("Unknown observation_type. Choose between: state, pixel")

        # Weather evolves
        self.weather.tick()

        # draw the display
        if self.render_display:
            draw_image(self.render_display, display_image)
            self.render_display.blit(
                self.font.render("Frame: %d" % self.count, True, (255, 255, 255)),
                (8, 10),
            )
            self.render_display.blit(
                self.font.render("Thottle: %f" % throttle, True, (255, 255, 255)),
                (8, 28),
            )
            self.render_display.blit(
                self.font.render("Steer: %f" % steer, True, (255, 255, 255)), (8, 46)
            )
            self.render_display.blit(
                self.font.render("Brake: %f" % brake, True, (255, 255, 255)), (8, 64)
            )
            self.render_display.blit(
                self.font.render(str(self.weather), True, (255, 255, 255)), (8, 82)
            )
            pygame.display.flip()

        # get reward and next observation
        reward, done, info = self._get_reward()
        if self.observations_type == "state":
            next_obs = self._get_state_obs()
        else:
            next_obs = self._get_pixel_obs(vision_image)

        # increase frame counter
        self.count += 1
        self.time_step += 1

        if self.time_step >= self._max_episode_steps:
            done = True

        return next_obs, reward, done, info

    def _get_pixel_obs(self, vision_image):
        bgra = np.array(vision_image.raw_data).reshape(84, 84, 4)
        bgr = bgra[:, :, :3]
        rgb = np.flip(bgr, axis=2)
        return rgb

    def _get_state_obs(self):
        transform = self.vehicle.get_transform()
        location = transform.location
        rotation = transform.rotation
        x_pos = location.x
        y_pos = location.y
        z_pos = location.z
        pitch = rotation.pitch
        yaw = rotation.yaw
        roll = rotation.roll
        acceleration = vector_to_scalar(self.vehicle.get_acceleration())
        angular_velocity = vector_to_scalar(self.vehicle.get_angular_velocity())
        velocity = vector_to_scalar(self.vehicle.get_velocity())
        return np.array(
            [
                x_pos,
                y_pos,
                z_pos,
                pitch,
                yaw,
                roll,
                acceleration,
                angular_velocity,
                velocity,
            ],
            dtype=np.float64,
        )

    def _get_reward(self):
        vehicle_location = self.vehicle.get_location()
        follow_waypoint_reward = self._get_follow_waypoint_reward(vehicle_location)
        done, collision_reward = self._get_collision_reward()
        cost = self._get_cost()
        total_reward = 100 * follow_waypoint_reward + 100 * collision_reward

        info_dict = dict()
        info_dict["follow_waypoint_reward"] = follow_waypoint_reward
        info_dict["collision_reward"] = collision_reward
        info_dict["cost"] = cost

        return total_reward, done, info_dict

    def _get_follow_waypoint_reward(self, location):
        nearest_wp = self.map.get_waypoint(location, project_to_road=True)
        distance = np.sqrt(
            (location.x - nearest_wp.transform.location.x) ** 2
            + (location.y - nearest_wp.transform.location.y) ** 2
        )
        return -distance

    def _get_collision_reward(self):
        if not self.collision:
            return False, 0
        else:
            return True, -1

    def _get_cost(self):
        # TODO: define cost function
        return 0

    def _on_collision(self, event):
        other_actor = get_actor_name(event.other_actor)
        self.collision = True
        self._reset_vehicle()

    def close(self):
        for actor in self.actor_list:
            actor.destroy()
        print("\ndestroying %d vehicles" % len(self.vehicles_list))
        self.client.apply_batch(
            [carla.command.DestroyActor(x) for x in self.vehicles_list]
        )
        time.sleep(0.5)
        pygame.quit()

    def render(self, mode):
        pass


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


# In[3]:


import glob
import os
import sys
import queue

try:
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass
import carla


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context
        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)
    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get("fps", 20)
        self._queues = []
        self._settings = None

        self.start()

    def start(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(
            carla.WorldSettings(
                no_rendering_mode=False,
                synchronous_mode=True,
                fixed_delta_seconds=self.delta_seconds,
            )
        )

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


# In[4]:


import glob
import os
import sys
import math

try:
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass
import carla


class Weather(object):
    def __init__(self, world, changing_weather_speed):
        self.world = world
        self.reset()
        self.weather = world.get_weather()
        self.changing_weather_speed = changing_weather_speed
        self._sun = Sun(self.weather.sun_azimuth_angle, self.weather.sun_altitude_angle)
        self._storm = Storm(self.weather.precipitation)

    def reset(self):
        weather_params = carla.WeatherParameters(sun_altitude_angle=90.0)
        self.world.set_weather(weather_params)

    def tick(self):
        self._sun.tick(self.changing_weather_speed)
        self._storm.tick(self.changing_weather_speed)
        self.weather.cloudiness = self._storm.clouds
        self.weather.precipitation = self._storm.rain
        self.weather.precipitation_deposits = self._storm.puddles
        self.weather.wind_intensity = self._storm.wind
        self.weather.fog_density = self._storm.fog
        self.weather.wetness = self._storm.wetness
        self.weather.sun_azimuth_angle = self._sun.azimuth
        self.weather.sun_altitude_angle = self._sun.altitude
        self.world.set_weather(self.weather)

    def __str__(self):
        return "%s %s" % (self._sun, self._storm)


class Sun(object):
    def __init__(self, azimuth, altitude):
        self.azimuth = azimuth
        self.altitude = altitude
        self._t = 0.0

    def tick(self, delta_seconds):
        self._t += 0.008 * delta_seconds
        self._t %= 2.0 * math.pi
        self.azimuth += 0.25 * delta_seconds
        self.azimuth %= 360.0
        min_alt, max_alt = [20, 90]
        self.altitude = 0.5 * (max_alt + min_alt) + 0.5 * (
            max_alt - min_alt
        ) * math.cos(self._t)

    def __str__(self):
        return "Sun(alt: %.2f, azm: %.2f)" % (self.altitude, self.azimuth)


class Storm(object):
    def __init__(self, precipitation):
        self._t = precipitation if precipitation > 0.0 else -50.0
        self._increasing = True
        self.clouds = 0.0
        self.rain = 0.0
        self.wetness = 0.0
        self.puddles = 0.0
        self.wind = 0.0
        self.fog = 0.0

    def tick(self, delta_seconds):
        delta = (1.3 if self._increasing else -1.3) * delta_seconds
        self._t = clamp(delta + self._t, -250.0, 100.0)
        self.clouds = clamp(self._t + 40.0, 0.0, 90.0)
        self.clouds = clamp(self._t + 40.0, 0.0, 60.0)
        self.rain = clamp(self._t, 0.0, 80.0)
        delay = -10.0 if self._increasing else 90.0
        self.puddles = clamp(self._t + delay, 0.0, 85.0)
        self.wetness = clamp(self._t * 5, 0.0, 100.0)
        self.wind = 5.0 if self.clouds <= 20 else 90 if self.clouds >= 70 else 40
        self.fog = clamp(self._t - 10, 0.0, 30.0)
        if self._t == -250.0:
            self._increasing = True
        if self._t == 100.0:
            self._increasing = False

    def __str__(self):
        return "Storm(clouds=%d%%, rain=%d%%, wind=%d%%)" % (
            self.clouds,
            self.rain,
            self.wind,
        )


def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))


# In[5]:


# if __name__ == '__main__':
#     # Easy scenario: no traffic, no dyanimc weather, no layers but roads and lighters
#     env = CarlaEnv(True, 2000, 0, 1, 'pixel', False, 'tesla.cybertruck', None, False, "All")
#     #     # Hard scenario
#     #     env = CarlaEnv(True,2000,0.1,1,'pixel',True,'tesla.cybertruck',None,True)
#     env.reset()
#     done = False
#     while not done:
#         next_obs, reward, done, info = env.step([1, 0])
#         print(next_obs, reward, done, info)
#     env.close()

# In[6]:


import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument("--domain_name", default="carla")
    parser.add_argument("--task_name", default="drive")
    parser.add_argument("--frame_stack", default=3, type=int)
    parser.add_argument("--action_repeat", default=4, type=int)
    parser.add_argument("--episode_length", default=500, type=int)
    parser.add_argument("--eval_mode", default="color_hard", type=str)

    # agent
    parser.add_argument("--algorithm", default="sgsac", type=str)
    parser.add_argument("--train_steps", default="500k", type=str)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--init_steps", default=500, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--hidden_dim", default=1024, type=int)

    # actor
    parser.add_argument("--actor_lr", default=1e-3, type=float)
    parser.add_argument("--actor_beta", default=0.9, type=float)
    parser.add_argument("--actor_log_std_min", default=-10, type=float)
    parser.add_argument("--actor_log_std_max", default=2, type=float)
    parser.add_argument("--actor_update_freq", default=2, type=int)

    # critic
    parser.add_argument("--critic_lr", default=1e-3, type=float)
    parser.add_argument("--critic_beta", default=0.9, type=float)
    parser.add_argument("--critic_tau", default=0.01, type=float)
    parser.add_argument("--critic_target_update_freq", default=2, type=int)
    parser.add_argument("--critic_weight_decay", default=0, type=float)

    # architecture
    parser.add_argument("--num_shared_layers", default=11, type=int)
    parser.add_argument("--num_head_layers", default=0, type=int)
    parser.add_argument("--num_filters", default=32, type=int)
    parser.add_argument("--projection_dim", default=100, type=int)
    parser.add_argument("--encoder_tau", default=0.05, type=float)

    # entropy maximization
    parser.add_argument("--init_temperature", default=0.1, type=float)
    parser.add_argument("--alpha_lr", default=1e-4, type=float)
    parser.add_argument("--alpha_beta", default=0.5, type=float)

    # auxiliary tasks
    parser.add_argument("--aux_lr", default=3e-4, type=float)
    parser.add_argument("--aux_beta", default=0.9, type=float)
    parser.add_argument("--aux_update_freq", default=2, type=int)

    # soda
    parser.add_argument("--soda_batch_size", default=256, type=int)
    parser.add_argument("--soda_tau", default=0.005, type=float)

    # svea
    parser.add_argument("--svea_alpha", default=0.5, type=float)
    parser.add_argument("--svea_beta", default=0.5, type=float)
    parser.add_argument("--sgqn_quantile", default=0.90, type=float)
    parser.add_argument("--svea_contrastive_coeff", default=0.1, type=float)
    parser.add_argument("--svea_norm_coeff", default=0.1, type=float)
    parser.add_argument("--attrib_coeff", default=0.25, type=float)
    parser.add_argument("--consistency", default=1, type=int)

    # eval
    parser.add_argument("--save_freq", default="10k", type=str)
    parser.add_argument("--eval_freq", default="1k", type=str)
    parser.add_argument("--eval_episodes", default=10, type=int)
    parser.add_argument("--distracting_cs_intensity", default=0.0, type=float)

    # misc
    parser.add_argument("--seed", default=10082, type=int)
    parser.add_argument("--log_dir", default="logs", type=str)
    parser.add_argument("--save_video", default=False, action="store_true")
    args = parser.parse_args()

    assert args.algorithm in {
        "sac",
        "rad",
        "curl",
        "pad",
        "soda",
        "drq",
        "svea",
        "saca",
        "sacfa",
        "sgsac",
    }, f'specified algorithm "{args.algorithm}" is not supported'

    assert args.eval_mode in {
        "train",
        "color_easy",
        "color_hard",
        "video_easy",
        "video_hard",
        "distracting_cs",
        "all",
        "none",
    }, f'specified mode "{args.eval_mode}" is not supported'
    assert args.seed is not None, "must provide seed for experiment"
    assert args.log_dir is not None, "must provide a log directory for experiment"

    intensities = {0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5}
    assert (
        args.distracting_cs_intensity in intensities
    ), f"distracting_cs has only been implemented for intensities: {intensities}"

    args.train_steps = int(args.train_steps.replace("k", "000"))
    args.save_freq = int(args.save_freq.replace("k", "000"))
    args.eval_freq = int(args.eval_freq.replace("k", "000"))

    if args.eval_mode == "none":
        args.eval_mode = None

    if args.algorithm in {"rad", "curl", "pad", "soda"}:
        args.image_size = 100
        args.image_crop_size = 84
    else:
        args.image_size = 84
        args.image_crop_size = 84

    return args


# In[7]:


import torch
import os
import numpy as np
import gym
from algorithms.rl_utils import make_obs_grad_grid
import utils
import time

# from arguments import parse_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from logger import Logger
from video import VideoRecorder


def evaluate(
    env, agent, algorithm, video, num_episodes, L, step, test_env=False, eval_mode=None
):
    episode_rewards = []
    for i in range(num_episodes):
        obs = env.reset()
        video.init(enabled=(i == 0))
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
                # obs = obs.reshape((84, 84, 3))
                video.record(env)
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
                    # attrib_grid = make_obs_grad_grid(torch.sigmoid(mask))
                    # agent.writer.add_image(
                    #     prefix + "/smooth_attrib", attrib_grid, global_step=step
                    # )

                episode_step += 1

        if L is not None:
            _test_env = f"_test_env_{eval_mode}" if test_env else ""
            video.save(f"{step}{_test_env}.mp4")
            L.log(f"eval/episode_reward{_test_env}", episode_reward, step)
        episode_rewards.append(episode_reward)

    return np.mean(episode_rewards)


from env.wrappers import VideoWrapper, FrameStack_carla
import os


def load_dataset_for_carla(max_episode_steps=20000, n_frames=10000):
    path_to_dataset = os.path.join(__file__[:-20], "datasets", "noisy_dataset_carla")
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


def main_sgqn_carla(args):
    # Set seed
    utils.set_seed_everywhere(args.seed)

    episode_length = args.episode_length
    frame_skip = 1
    max_episode_steps = (episode_length + frame_skip - 1) // frame_skip

    load_dataset_for_carla()

    env = CarlaEnv(
        True,
        2000,
        0,
        1,
        "pixel",
        False,
        "tesla.cybertruck",
        None,
        False,
        "All",
        max_episode_steps,
    )

    env = FrameStack_carla(env, args.frame_stack)

    test_envs = []
    test_envs_mode = []
    for cond in ["color_easy"]:  # , "color_hard"]:
        if cond == "color_easy":
            # Easy scenario: no traffic, no dyanimc weather, no layers but roads and lighters
            test_env = CarlaEnv(
                True,
                2003,
                0,
                1,
                "pixel",
                False,
                "tesla.cybertruck",
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
                1,
                "pixel",
                True,
                "tesla.cybertruck",
                None,
                False,
                None,
                max_episode_steps,
            )

        # test_env = VideoWrapper(env, cond, 1)
        test_env = FrameStack_carla(test_env, args.frame_stack)

        test_envs.append(test_env)
        test_envs_mode.append(args.eval_mode)

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
    video_dir = utils.make_dir(os.path.join(work_dir, "video"))
    video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)
    utils.write_info(args, os.path.join(work_dir, "info.log"))

    # Prepare agent
    assert torch.cuda.is_available(), "must have cuda enabled"

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

    # create agent
    agent = make_agent(
        obs_shape=cropped_obs_shape, action_shape=env.action_space.shape, args=args
    )

    # Initialize variables
    start_step, episode, episode_reward, done = 0, 0, 0, True

    # Define logger
    L = Logger(work_dir)

    # Start training
    start_time = time.time()
    for step in range(start_step, args.train_steps + 1):
        # EVALUATE:
        if done:
            if step > start_step:
                L.log("train/duration", time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)

            # Evaluate agent periodically
            if step % args.eval_freq == 0:
                print("Evaluating:", work_dir)
                L.log("eval/episode", episode, step)
                evaluate(env, agent, args.algorithm, video, args.eval_episodes, L, step)
                if test_envs is not None:
                    for test_env, test_env_mode in zip(test_envs, test_envs_mode):
                        evaluate(
                            test_env,
                            agent,
                            args.algorithm,
                            video,
                            args.eval_episodes,
                            L,
                            step,
                            test_env=True,
                            eval_mode=test_env_mode,
                        )
                L.dump(step)

            # Save agent periodically
            if step > start_step and step % args.save_freq == 0:
                torch.save(
                    agent.actor.state_dict(),
                    os.path.join(model_dir, f"actor_{step}.pt"),
                )
                torch.save(
                    agent.critic.state_dict(),
                    os.path.join(model_dir, f"critic_{step}.pt"),
                )
                if args.algorithm == "sgsac":
                    torch.save(
                        agent.attribution_predictor.state_dict(),
                        os.path.join(model_dir, f"attrib_predictor_{step}.pt"),
                    )

            L.log("train/episode_reward", episode_reward, step)

            # Reset environment
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0

            episode += 1

            L.log("train/episode", episode, step)

        # TRAIN:
        # Sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # Run training update
        if step >= args.init_steps:
            num_updates = args.init_steps if step == args.init_steps else 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        # Take step
        next_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)

        # Update replay buffer
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        episode_reward += reward
        obs = next_obs
        episode_step += 1

    print("Completed training for", work_dir)


# In[ ]:

# import os

if __name__ == "__main__":
    # os.system("sh ./src/carla_servers_start.sh")
    args = parse_args()
    main_sgqn_carla(args)
