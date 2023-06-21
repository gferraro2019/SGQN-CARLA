#!/usr/bin/env python
# coding: utf-8

# In[1]:
import glob
import math
import os
import queue
import random
import sys
import time

import carla
import gym
import numpy as np
import pygame
from gym import spaces

from utils import (
    clamp,
    draw_image,
    get_actor_name,
    get_font,
    should_quit,
    vector_to_scalar,
)

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
        vehicle_color,
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
        self.vehicle_color = vehicle_color
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
        if "pixel" in self.observations_type:
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
        if self.render_display and "pixel" in self.observations_type:
            self.sync_mode = CarlaSyncMode(
                self.world, self.camera_display, self.camera_vision, fps=20
            )
        elif self.render_display and self.observations_type == "state":
            self.sync_mode = CarlaSyncMode(self.world, self.camera_display, fps=20)
        elif not self.render_display and "pixel" in self.observations_type:
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

        self.observation_space = None

        if self.observations_type == "sgqn_pixel":
            obs = np.zeros((3, 84, 84))
            dx = 0.0
            dy = 0.0
            self.observation_space = spaces.Tuple(
                (
                    spaces.Box(0, 255, shape=obs.shape, dtype=np.uint8),
                    spaces.Box(-np.inf, np.inf, shape=(2,)),
                )
            )
        else:
            # get initial observation
            if self.observations_type == "state":
                obs = self._get_state_obs()

            else:
                obs = np.zeros((3, 84, 84))

            self.obs_dim = obs.shape
            self.observation_space = spaces.Box(
                -np.inf, np.inf, shape=self.obs_dim, dtype="float32"
            )

        # gym environment specific variables
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype="float32")

        # Fix a Waypoint
        self.waypoint = None
        self._fix_waypoint()

    def _fix_waypoint(self):
        transform = self.vehicle.get_transform()
        location = transform.location
        self.waypoint = self.map.get_waypoint(location, project_to_road=True)

    def reset(self):
        self._reset_vehicle()
        self.world.tick()
        self._reset_other_vehicles()
        self.world.tick()
        self.count = 0
        self.collision = False

        # set the car always at the same distance from the waypoint
        self._fix_waypoint()
        transform = self.vehicle.get_transform()
        location = transform.location
        location.x = self.waypoint.transform.location.x - 2
        location.y = self.waypoint.transform.location.y - 2

        # to let the car to stabilize during its falling caused by the reset
        for _ in range(100):
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
                if self.vehicle_color is not None:
                    color = self.vehicle_color

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
        return (
            next_obs,
            np.mean(rewards),
            done,
            info,
        )

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
        if self.render_display and "pixel" in self.observations_type:
            snapshot, display_image, vision_image = self.sync_mode.tick(timeout=2.0)
        elif self.render_display and self.observations_type == "state":
            snapshot, display_image = self.sync_mode.tick(timeout=2.0)
        elif not self.render_display and "pixel" in self.observations_type:
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
            # for sgqn_carla add distances to the state
            next_obs = self._get_pixel_obs(vision_image)
            next_obs = next_obs.reshape(3, 84, 84).astype(np.uint8)
            dx, dy = self._compute_distance_from_waypoint()
            next_obs = (next_obs, dx, dy)

        # increase frame counter
        self.count += 1
        self.time_step += 1

        if self.time_step >= self._max_episode_steps:
            done = True

        return next_obs, reward, done, info

    def _compute_distance_from_waypoint(self):
        transform = self.vehicle.get_transform()
        location = transform.location
        # nearest_wp = self.map.get_waypoint(location, project_to_road=True)

        # dx = np.sqrt(location.x - nearest_wp.transform.location.x) ** 2
        # dy = np.sqrt(location.y - nearest_wp.transform.location.y) ** 2

        dx = np.sqrt(location.x - self.waypoint.transform.location.x) ** 2
        dy = np.sqrt(location.y - self.waypoint.transform.location.y) ** 2

        return dx, dy

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


# In[3]:


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
