#!/usr/bin/env python
# coding: utf-8

import glob
import math
# # from agents.navigation.roaming_agent import RoamingAgent
import os
import queue
import random
import sys
import time

# In[1]:
import carla
import gym
import matplotlib.pyplot as plt
import numpy as np
import pygame
from gym import spaces
from mpmath import csch

from utils import (avoid_list, clamp, draw_image, get_actor_name, get_font,
                   should_quit, vector_to_scalar)

# from agents.navigation.roaming_agent import RoamingAgent
try:
    sys.path.append(
        glob.glob(
            "/home/dcas/g.ferraro/Desktop/CARLA/CARLA_0.9.14/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg"
            # glob.glob(
            #     "../carla/dist/carla-*%d.%d-%s.egg"
            #     % (
            #         sys.version_info.major,
            #         sys.version_info.minor,
            #         "win-amd64" if os.name == "nt" else "linux-x86_64",
            #     )
        )[0]
    )
except IndexError:
    pass


class CarlaEnv(gym.Env):
    """This class define a Carla environment.

    Args:
        gym (_type_): _description_
    """

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
        total_number_waypoint = 500,
        lower_limit_return_=-600,
        visualize_target=False,
        trace_trajectories=True,
        verbose=False,
        image_size=64
    ):
        """This function initialize the Carla enviroment.

        Args:
            render (boolean): whether or not to show the display window of the environment.
            carla_port (int): the number of the port of the server to connect the client.
            changing_weather_speed (_type_): _description_
            frame_skip (int): number of frames to skip.
            observations_type (_type_): _description_
            traffic (boolean): whether or not to add traffic int the environment.
            vehicle_name (str): name of the vehicle. ie.("tesla","c3")
            vehicle_color (tuple(int,int,int)): RGB tuple.
            map_name (str): the name of the map. ie.("map 01")
            autopilot (boolean): whether or not to activate the autopilot for the ego vehicle.
            unload_map_layer (str optional): whether or not to remove some layer from the map. "All" leave everything as it is, "custom" use customized layers. Defaults to None.
            max_episode_steps (int, optional): the lenght of the episode in frames. Defaults to 1000.
            lower_limit_return_ (int, optional): The lowest cumulative reward possible. Defaults to -600.
            visualize_target (bool, optional): whether or not to show the waypoint in the environment. Defaults to False.

        Raises:
            ValueError: _description_
        """
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
        self.trace_trajectories = trace_trajectories
        self.verbose = verbose
        self.actor_list = []
        self.image_size = image_size
        
        print(max_episode_steps)
        self._max_episode_steps = int(max_episode_steps)
        self.total_number_waypoint = total_number_waypoint
        self.current_step = 0

        # used in reward function
        self.previous_steer = 0
        self.previous_distance = 0
        self.wp_is_reached = 0

        self.visualize_target = visualize_target

        # to end the task when the lower limit is reached
        self.lower_limit_return_ = lower_limit_return_
        self.return_ = 0    

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
        self.client.set_timeout(5.0)

        # initialize world and map
        if self.map_name is not None:
            self.world = self.client.load_world(self.map_name)
        else:
            self.world = self.client.get_world()

        self.map = self.world.get_map()

        # unload map layers
        if unload_map_layer is not None:
            if unload_map_layer == "All":
                self.world.unload_map_layer(carla.MapLayer.All)
            elif unload_map_layer == "Custom":
                layers = {
                    "Buildings": carla.MapLayer.Buildings,
                    "Decals": carla.MapLayer.Decals,
                    "Foliage": carla.MapLayer.Foliage,
                    "Ground": carla.MapLayer.Ground,
                    "ParkedVehicles": carla.MapLayer.ParkedVehicles,
                    "Particles": carla.MapLayer.Particles,
                    "Props": carla.MapLayer.Props,
                    "StreetLights": carla.MapLayer.StreetLights,
                    "Walls": carla.MapLayer.Walls,
                    "All": carla.MapLayer.All,
                }
                for layer, value in layers.items():
                    if layer not in ["Walls,", "Ground", "Streetlights", "All"]:
                        self.world.unload_map_layer(value)

        self.world.tick()

        # create vehicle
        self.vehicle = None
        self.vehicles_list = []
        
        # Fix a Waypoint
        self.waypoint = None
        self.counter_waypoint = 0
        self.counter_step_zero_speed = 0
        self.max_distance_from_waypoint = 2

        # weather
        self.weather = Weather(self.world, self.changing_weather_speed)

        self.observation_space = None

        if self.observations_type == "sgqn_pixel":
            obs = np.zeros((3, self.image_size, self.image_size))
            state = np.zeros(9, dtype=np.float32)
            self.observation_space = spaces.Tuple(
                (
                    spaces.Box(0, 1, shape=obs.shape, dtype=np.float32),
                    spaces.Box(-np.inf, np.inf, shape=state.shape, dtype="float32"),
                )
            )
        else:
            # get initial observation
            if self.observations_type == "state":
                obs = self._get_state_obs()

            else:
                obs = np.zeros((3, self.image_size, self.image_size))

            self.obs_dim = obs.shape
            self.observation_space = spaces.Box(
                -np.inf, np.inf, shape=self.obs_dim, dtype="float32"
            )

        # gym environment specific variables
        self.action_space = spaces.Tuple(
            (
                spaces.Box(0, 1.0, shape=(1,), dtype="float32"),
                spaces.Box(-1, 1, shape=(1,), dtype="float32"),
            )
        )

        self.bike = None
        self.bonus = 0

    def spawn_sensors(self):
        
        # collision detection
        self.collision = False
        sensor_blueprint = self.world.get_blueprint_library().find(
            "sensor.other.collision"
        )
        self.collision_sensor = self.world.spawn_actor(
            sensor_blueprint, carla.Transform(), attach_to=self.vehicle
        )
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        
        self.actor_list.append(self.collision_sensor)

        # lane invasion detector
        self.lane_invasion = False
        sensor_blueprint = self.world.get_blueprint_library().find(
            "sensor.other.lane_invasion"
        )
        self.lane_invasion_sensor = self.world.spawn_actor(
            sensor_blueprint, carla.Transform(), attach_to=self.vehicle
        )
        self.lane_invasion_sensor.listen(lambda event: self._on_lane_invasion(event))
        self.n_lane_invasions = 0
        self.actor_list.append(self.lane_invasion_sensor)
        
    
    def spawn_cameras(self):
        # initialize blueprint library
        blueprint_library = self.world.get_blueprint_library()

        # spawn camera for rendering
        if self.render_display:
            location = carla.Location(x=1.6, z=1.7)
            self.camera_display = self.world.spawn_actor(
                blueprint_library.find("sensor.camera.rgb"),
                carla.Transform(location, carla.Rotation(yaw=0.0)
                ),
                attach_to=self.vehicle,
            )
            self.actor_list.append(self.camera_display)

        # spawn camera for pixel observations
        if "pixel" in self.observations_type:
            bp = blueprint_library.find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(self.image_size))
            bp.set_attribute("image_size_y", str(self.image_size))
            bp.set_attribute("fov", str(self.image_size))
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
    
    def destroy_prevoius_actors(self):
        if len(self.actor_list)>0:
            # # remove old vehicles and sensors (in case they survived)
            # self.actor_list = self.world.get_actors()
            # for vehicle in self.actor_list.filter("*vehicle*"):
            #     print("Warning: removing old vehicle")
            #     vehicle.destroy()
            # for sensor in self.actor_list.filter("*sensor*"):
            #     print("Warning: removing old sensor")
            #     sensor.destroy()
            for actor in self.actor_list:
                actor.destroy()
                
            self.vehicle = None
            self.actor_list.clear()
            assert len(self.actor_list)==0 , f"list still contains something {self.actor_list}"
    
    def generate_list_waypoints(self,start_waypoint,number_waypoints):
        waypoints = []
        waypoints.append(start_waypoint)
        temp_waypoint = start_waypoint
        for _ in range(number_waypoints):
            temp_waypoint = temp_waypoint.next(1)[0]
            waypoints.append(temp_waypoint)
        return waypoints
    
    def reset(self):        
        # to avoid influnces from the former episode (angular momentun preserved)
        self.destroy_prevoius_actors()
        
        if self.trace_trajectories:
            self.trace=[]
            self.waypoints_trace = []
            
        self.list_skipped_waypoints = []

        self._reset_vehicle()
        self.world.tick()

        self._reset_other_vehicles()
        self.world.tick()

        self.spawn_cameras()
        self.spawn_sensors()

        self.bonus = 0
        self.previous_distance = 0
        self.collision = False
        self.lane_invasion = False
        
        self.waypoint = self.map.get_waypoint(self.vehicle.get_location(),project_to_road=True, lane_type=(carla.LaneType.Driving ))
        
        self.waypoints = self.generate_list_waypoints(self.waypoint,self.total_number_waypoint)
        self.current_waypoint_idx = 0
        
        self.wp_is_reached =False
        self.counter_waypoint=0
        self.counter_step_zero_speed=0

        if self.bike is not None:
            self.bike.destroy()

        transform = carla.Transform()
        if self.visualize_target == True:
            blueprint_library = self.world.get_blueprint_library()
            veichles = blueprint_library.filter("vehicle.*.*")

            bikes_blueprints = [
                v for v in veichles if v.get_attribute("number_of_wheels").as_int() == 2
            ]
            bike_blueprint = bikes_blueprints[0]
            bike_blueprint.set_attribute("color", "0,255,0")
            transform.location.y = self.waypoint.transform.location.y
            transform.location.x = self.waypoint.transform.location.x
            transform.location.z = 0

            transform.rotation.yaw = -180

            self.bike = self.world.try_spawn_actor(bike_blueprint, transform)

            self.world.tick()

            print(
                f"distance = { np.sqrt((transform.location.x - self.vehicle.get_transform().location.x)**2+(transform.location.y - self.vehicle.get_transform().location.y)**2)}"
            )

        print(f"distance = { np.sqrt((self.waypoint.transform.location.x - self.vehicle.get_transform().location.x)**2 + (self.waypoint.transform.location.y - self.vehicle.get_transform().location.y)**2)}")

        # self._fix_waypoint()  # second time for placing the global waypoint

        # to let the car to stabilize during its falling caused by the reset
        for _ in range(30):
            obs, _, _, _ = self.step([0, 0])
        self.current_step = 0

        self.return_ = 0
        return obs

    def generate_waypoints(self):
        wp_list = []
        for i in range(100):
            wp = self.world.get_map().get_waypoint_xodr(0, -2, i)
            if wp is not None:
                wp_list.append(wp)
        points = np.array(
            [(x.transform.location.x, x.transform.location.y) for x in wp_list]
        )

    def _reset_vehicle(self):
        # choose random spawn point
        init_transforms = self.world.get_map().get_spawn_points()
        vehicle_init_transform = random.choice(init_transforms)

        # create the vehicle
        if self.vehicle is None:
            blueprint_library = self.world.get_blueprint_library()
            vehicle_blueprint = blueprint_library.find("vehicle." + self.vehicle_name)
            if vehicle_blueprint.has_attribute("color"):
                if self.vehicle_color is not None:
                    color = self.vehicle_color
                else:
                    color = random.choice(
                        vehicle_blueprint.get_attribute("color").recommended_values
                    )

                vehicle_blueprint.set_attribute("color", color)

            # spawn vehicle
            self.vehicle = self.world.spawn_actor(vehicle_blueprint, vehicle_init_transform)
            self.actor_list.append(self.vehicle)

            

    def _reset_other_vehicles(self):
        #TODO add machines to actor_list
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
    
    def generate_waypoint_from_lane(self,n_lane,density_wp=8,plot=False,remove=[19,20,21,22,10,23,0,1,2,3,72,101,102,118,119,120,121,122,123,18,16,17,13,12,11,10]):

        def solve(points):
            def key(x):
                atan = math.atan2(x[1], x[0])
                return (atan, x[1]**2+x[0]**2) if atan >= 0 else (2*math.pi + atan, x[0]**2+x[1]**2)

            return sorted(points, key=key)
        
        lanes = self.getLanes([n_lane],plot,density_wp)
        lane=lanes[n_lane]
        
        print(f"before reducing...{len(lane['x'])}")

        
        if plot:
            k=0
            plt.figure()
            plt.scatter(lane["x"],lane["y"])
            for x,y in zip(lane["x"],lane["y"]):
                plt.text(x+2, y, k)
                k+=1
            plt.show()
        
        
        
        xs =[]
        ys = []
        k = 0
        for x,y in zip(lane["x"],lane["y"]):
            if k not in remove:
                # print(k,x,y)
                xs.append(x)
                ys.append(y)
            else:
                print("removed",k)
            k+=1
        
        print(f"after reducing...{len(xs)}")
  
        points = [ (x,y) for x,y in zip(xs,ys)]

        lane_reduced = np.asarray(solve(points))
        
        if plot:
            k=0
            plt.figure()
            plt.scatter(lane_reduced[:,0],lane_reduced[:,1])
            for x,y in zip(lane_reduced[:,0],lane_reduced[:,1]):
                plt.text(x+2, y, k)
                k+=1
            plt.show()
    
                
        return lane_reduced


    def getLanes(self, lane_idx,plot=False,density_wp=8):
        """Plot waypoints in the map according to the lane id

        Args:
            lane_idx (list): list of lane IDs
        
        Retunrs:
            lanes : dict of lanes
        """
        import matplotlib.pyplot as plt

        topology = self.map.generate_waypoints(density_wp)  # self.map.get_topology()
        print(len(topology))
        lane_m5 = {"x": [], "y": [], "junction_id": [], "road_id": []}
        lane_m4 = {"x": [], "y": [], "junction_id": [], "road_id": []}
        lane_m3 = {"x": [], "y": [], "junction_id": [], "road_id": []}
        lane_m2 = {"x": [], "y": [], "junction_id": [], "road_id": []}
        lane_m1 = {"x": [], "y": [], "junction_id": [], "road_id": []}
        lane_0 = {"x": [], "y": [], "junction_id": [], "road_id": []}
        lane_p1 = {"x": [], "y": [], "junction_id": [], "road_id": []}
        lane_p2 = {"x": [], "y": [], "junction_id": [], "road_id": []}
        lane_p3 = {"x": [], "y": [], "junction_id": [], "road_id": []}
        lane_p4 = {"x": [], "y": [], "junction_id": [], "road_id": []}
        lane_p5 = {"x": [], "y": [], "junction_id": [], "road_id": []}

        lanes = {
            -5: lane_m5,
            -4: lane_m4,
            -3: lane_m3,
            -2: lane_m2,
            -1: lane_m1,
            0: lane_0,
            1: lane_p1,
            2: lane_p2,
            3: lane_p3,
            4: lane_p4,
            5: lane_p5,
        }

        for wp in topology:
            # print(wp.lane_id)
            if "Driving" in str(wp.lane_type):
                lanes[wp.lane_id]["x"].append(wp.transform.location.x)
                lanes[wp.lane_id]["y"].append(wp.transform.location.y)
                lanes[wp.lane_id]["road_id"].append(wp.road_id)
                lanes[wp.lane_id]["junction_id"].append(wp.junction_id)
                
        if plot:
            plt.figure()
            plotted = []
            for id_lane in lanes:
                if id_lane in lane_idx:
                    # print(id_lane)
                    color = (
                        np.array(
                            (
                                random.randint(0, 255),
                                random.randint(0, 255),
                                random.randint(0, 255),
                            )
                        )
                        / 255
                    )
                    try:
                        plt.scatter(lanes[id_lane]["x"], lanes[id_lane]["y"], color=color)
                        plotted.append(id_lane)
                    except:
                        print(f"not valid lane {id_lane}")
                        plotted.pop()
            plt.legend(plotted, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
            plt.show()
        return lanes

    def step(self, action):
        rewards = []
        next_obs, done, info = None, False, {}
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
                self.font.render("Frame: %d" % self.current_step, True, (255, 255, 255)),
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
        reward, done, info = self._get_reward(throttle_brake, steer)

        # update cumulative reward to interupt if the lower limit is reached
        self.return_ += reward

        if self.observations_type == "state":
            next_obs = self._get_state_obs()
        else:
            # for sgqn_carla add distances to the state
            next_obs = self._get_pixel_obs(vision_image)
            next_obs = next_obs.reshape(3, self.image_size, self.image_size)
            state = self._get_state_obs()
            next_obs = (next_obs, state)

        # increase frame counter
        self.current_step += 1

        if self.current_step >= self._max_episode_steps or float(self.return_) <= float(
            self.lower_limit_return_
        ):
            done = True

        return next_obs, reward, done, info

    def _compute_distance_from_waypoint(self):
        transform = self.vehicle.get_transform()
        location = transform.location
        # nearest_wp = self.map.get_waypoint(location, project_to_road=True)

        # dx = np.sqrt(location.x - nearest_wp.transform.location.x) ** 2
        # dy = np.sqrt(location.y - nearest_wp.transform.location.y) ** 2

        dx = np.sqrt((location.x - self.waypoint.transform.location.x) ** 2)
        dy = np.sqrt((location.y - self.waypoint.transform.location.y) ** 2)

        return dx, dy

    def _get_pixel_obs(self, vision_image):
        bgra = np.array(vision_image.raw_data).reshape(self.image_size, self.image_size, 4)
        bgr = bgra[:, :, :3]
        rgb = np.flip(bgr, axis=2)
        return rgb/255

    def _get_state_obs(self):
        """This funciton return a state of 9 elements:
            dx_pos,
            dy_pos,
            dz_pos,
            delta_pitch,
            delta_yaw,
            delta_roll,
            acceleration,
            angular_velocity,
            velocity.

        Returns:
            np.array: the state
        """
        transform = self.vehicle.get_transform()
        location = transform.location
        rotation = transform.rotation
        dx_pos = np.abs(location.x - self.waypoint.transform.location.x)
        dy_pos = np.abs(location.y - self.waypoint.transform.location.y)
        dz_pos = np.abs(location.z - self.waypoint.transform.location.z)
        delta_pitch = self.waypoint.transform.rotation.pitch - rotation.pitch
        delta_yaw = self.waypoint.transform.rotation.yaw - rotation.yaw
        delta_roll = self.waypoint.transform.rotation.roll - rotation.roll
        acceleration = vector_to_scalar(self.vehicle.get_acceleration())
        angular_velocity = vector_to_scalar(self.vehicle.get_angular_velocity())
        velocity = vector_to_scalar(self.vehicle.get_velocity())
        #completed_wp = self.counter_waypoint/self.total_number_waypoint
        #completed_percentage_frame = (self.current_step+1)/self._max_episode_steps
        
        return np.array(
            [
                round(dx_pos, 4),
                round(dy_pos, 4),
                round(dz_pos, 4),
                round(delta_pitch / 360, 4),
                round(delta_yaw / 360, 4),
                round(delta_roll / 360, 4),
                round(acceleration, 4),
                round(angular_velocity, 4),
                round(velocity, 4),
                #round(completed_wp,4),
                #round(completed_percentage_frame,4),                
            ],
            dtype=np.float32,
        )

    def compute_alpha_between_lines(self,plot=False):
        p0 = self.waypoint.previous(2)
        p1 = self.waypoint.next(2)
        pcar = (self.vehicle.get_transform().location.x,self.vehicle.get_transform().location.y)
        
        m0 = (p1[0]-p0[0])/(p1[1]-p0[1]+0.001) 
        m1 = (pcar[0]-p0[0])/(pcar[1]-p0[1]+0.001) 
        tan_alpha = abs((m0-m1)/(1+m0*m1))
        alpha = math.atan(tan_alpha)*180/np.pi
        
        if alpha <0:
            assert "alpha less than zero"
        
        if plot:
            plt.scatter(p0[0],p0[1])
            plt.scatter(p1[0],p1[1])
            plt.scatter(pcar[0],pcar[1])
            
        return alpha


    def plot_trajectories(self):
        x_trace = [p.x for p in self.trace]
        y_trace = [p.y for p in self.trace]
        x_waypoints = [p.x for p in self.waypoints_trace]
        y_waypoints = [p.y for p in self.waypoints_trace]
        x_skipped_waypoints = [p.transform.location.x for p in self.list_skipped_waypoints]
        y_skipped_waypoints = [p.transform.location.y for p in self.list_skipped_waypoints]
        # veichle trace
        plt.scatter(x_trace,y_trace,c="blue")
        # list waypoints
        plt.scatter(x_waypoints,y_waypoints,c="orange")
        # list skipped waypoints
        plt.scatter(x_skipped_waypoints,y_skipped_waypoints,c="red")
        #starting poistion veichle
        plt.scatter(x_trace[0],y_trace[0],c="white")
        #starting waypoint
        plt.scatter(x_waypoints[0],y_waypoints[0],c="black")
        # current aimed waypoint
        plt.scatter(self.waypoint.transform.location.x,self.waypoint.transform.location.y,c="green")
        
        #plt.show()


    def has_skipped_waypoints(self,vehicle_location,num_waypoints):
        """This function it checks if the vheicle has skipped some waypoints.
        The number of waypoint skippeble is defined by the num_waypoints parameter.

        Args:
            num_waypoints (_type_): number of skippeble waypoints.

        Returns:
            skipped (bool): the boolean indicates whether the car went too far or just skipped some waypoint.
        """
        
        distances = []
        waypoints = []
        temp_waypoint = self.waypoint        
        skipped = False
        
        
    
        # for i in range(num_waypoints):
        #     temp_waypoint = temp_waypoint.next(1)[0]
        #     if temp_waypoint is not None:
        #         waypoints.append(temp_waypoint)
        #         distance = np.sqrt(
        #             (vehicle_location.x - temp_waypoint.transform.location.x) ** 2
        #             + (vehicle_location.y - temp_waypoint.transform.location.y) ** 2
        #         )
        #         distances.append(distance)
                
        for i in range(num_waypoints):
            temp_waypoint = self.waypoints[self.current_waypoint_idx + i+1]
            if temp_waypoint is not None:
                waypoints.append(temp_waypoint)
                distance = np.sqrt(
                    (vehicle_location.x - temp_waypoint.transform.location.x) ** 2
                    + (vehicle_location.y - temp_waypoint.transform.location.y) ** 2
                )
                distances.append(distance)
        
        distances = np.array(distances)
        argmin = np.argmin(distances)
        
        
        
        if distances[argmin] < self.max_distance_from_waypoint:
            skipped=True
            if self.verbose:
                print(f"\nskipped previous WP: {argmin+1}, it continues for the {argmin+2}")
            # self.waypoint = waypoints[argmin]
            # self.previous_distance = distances[argmin]

            # update list skipped
            self.list_skipped_waypoints.append(self.waypoint)
            for i in range(argmin):
                self.list_skipped_waypoints.append(waypoints[i])
        else:
            self.list_skipped_waypoints.append(self.waypoint)
            self.list_skipped_waypoints += waypoints
        
        return skipped, waypoints[argmin], distances[argmin], argmin
                
    def   _get_reward(self, throttle, steer):
        info_dict = dict()
        info_dict["looped"] = False
        goal, done, total_reward = False, False, 0
        vehicle_location = self.vehicle.get_location()
        
        if self.trace_trajectories:
            self.trace.append(vehicle_location)
            self.waypoints_trace.append(self.waypoint.transform.location)
        
        
        distance = np.sqrt(
            (vehicle_location.x - self.waypoint.transform.location.x) ** 2
            + (vehicle_location.y - self.waypoint.transform.location.y) ** 2
        )
        
        vehicle_velocity = self.vehicle.get_velocity()
        speed = round(
            3.6 * np.linalg.norm(np.array([vehicle_velocity.x, vehicle_velocity.y])), 3
        )
        
        if speed <=0:
            self.counter_step_zero_speed+=1
            # total_reward = -0.01
        else:
            self.counter_step_zero_speed =0
        
        #the best till now
        # total_reward = ((self.counter_waypoint-1)/(self.current_step+1))*self._max_episode_steps - 0
        
        #total_reward = (self.counter_waypoint/(self.current_step+1))*self._max_episode_steps -1
        #rew1 = (k*step/(steps))-1000
        # total_reward = self.counter_waypoint - self.current_step/1000 - 0.7 + self.bonus
        
        # total_reward += self.counter_waypoint*1000/self._max_episode_steps
        
        
        if  self.wp_is_reached:
            self.previous_distance = distance    
            self.wp_is_reached = False

        diff =   distance - self.previous_distance 
        if diff < 0:
            total_reward = 1 
        else:
            total_reward = -1 


        total_reward += (-diff - abs(steer))*speed

        self.remaining_distance = self.total_number_waypoint - self.counter_waypoint
        self.previous_distance = distance    
        
        if distance <= 0.3:
            self.wp_is_reached = True
            # total_reward += 1e3#self.counter_waypoint
            # self.bonus +=1

            #update waypoint
            self.counter_waypoint+=1
            self.current_waypoint_idx+=1
            #self.waypoint = self.waypoint.next(1.)[0]
            self.waypoint = self.waypoints[self.current_waypoint_idx]
            if self.counter_waypoint +1 >= self.total_number_waypoint:
                done = True
                goal = True
        
        # elif distance > 1 + self.previous_distance and distance < self.max_distance_from_waypoint:
        #     total_reward = -1#-0.01 * distance
            
        elif distance >= self.max_distance_from_waypoint or self.counter_step_zero_speed == 300:
            #self.plot_trajectories()
            self.counter_step_zero_speed = 0
            # total_reward = -1#-0.01 * distance
            done = True


        info_dict["distance"] = self.remaining_distance
        info_dict["goal"] = goal
        info_dict["#WP"] = self.counter_waypoint

        return total_reward, done, info_dict

    def _get_follow_waypoint_reward(self, location):
        nearest_wp = self.map.get_waypoint(location, project_to_road=True)
        distance = np.sqrt(
            (location.x - nearest_wp.transform.location.x) ** 2
            + (location.y - nearest_wp.transform.location.y) ** 2
        )
        return -distance

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
        # other_actor = get_actor_name(event.other_actor)
        self.collision = True
        # self._reset_vehicle(from_fixed_point=True)

    def _on_lane_invasion(self, event):
        # self.lane_invasion_event = event
        # print(event.crossed_lane_markings)
        self.lane_invasion = True  # len(event.crossed_lane_markings) * -100

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
