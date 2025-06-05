import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import time
from lidar_sensor import VLP16Sensor
import json
from scipy.spatial.transform import Rotation as R
from randomize_obstacles import randomize_environment


class Jackal_Env(gym.Env):
    def __init__(self, xml_file="jackal_obstacles.xml", render_mode=None,
                 use_lidar=True, num_lidar_rays_h=360, num_lidar_rays_v=16,
                 lidar_max_range=10.0):
        super().__init__()

        # Load the MuJoCo model
        self.xml_file = xml_file
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)
        self.metadata = {"render_modes": ["human", "rgb_array"],
                         "render_fps": int(1.0 / (self.model.opt.timestep))}

        self.lidar_viz_enabled = False
        self.num_lidar_rays_h = num_lidar_rays_h
        self.lidar_max_range = lidar_max_range
        # LiDAR configuration
        self.use_lidar = use_lidar
        if self.use_lidar:
            self.lidar = VLP16Sensor(
                self.model,
                self.data,
                lidar_name="velodyne",
                horizontal_resolution=360.0/num_lidar_rays_h,  # Convert ray count to degrees
                rotation_rate=10,  # Hz
                max_range=lidar_max_range
                # return_type='ranges'  # Just return ranges for the gym environment
            )

        # Define state (observation) and action spaces
        basic_obs_size = self.model.nq + self.model.nv  # joint positions + velocities

        if self.use_lidar:
            lidar_obs_size = (num_lidar_rays_h * num_lidar_rays_v)
            self.observation_space = spaces.Dict({
                'state': spaces.Box(low=-np.inf, high=np.inf, shape=(basic_obs_size,), dtype=np.float32),
                'lidar': spaces.Box(low=0, high=lidar_max_range,
                                    shape=(num_lidar_rays_v, num_lidar_rays_h),
                                    dtype=np.float32)
            })
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(basic_obs_size,), dtype=np.float32
            )

        # For differential drive, we only need 2 actions (left and right)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )

        self.viewer = None
        self.render_mode = render_mode

        # Get the indices of the actuators for the left and right wheels
        self.left_front = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_actuator")
        self.left_rear = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rear_left_actuator")
        self.right_front = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_actuator")
        self.right_rear = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rear_right_actuator")

        # Read reward config
        with open('rewards.json', 'r') as file:
            self.rewards = json.load(file)

        self.floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        self.robot_geom_ids = []
        self.obstacle_geom_ids = []
        self.goal_id = []

        for i in range(self.model.ngeom):
            if self.model.geom_group[i] == 2:
                if i != self.floor_geom_id:
                    self.robot_geom_ids.append(i)
            elif self.model.geom_group[i] == 1:
                self.obstacle_geom_ids.append(i)
            elif self.model.geom_group[i] == 3:
                self.goal_id.append(i)

        # Print for debugging:
        print(f"obstacle geom ids: {self.obstacle_geom_ids}")
        print(f"robot geom ids: {self.robot_geom_ids}")
        print(f"floor geom id: {self.floor_geom_id}")

        self.goal_position = self.extract_goal_position()
        print(f"Goal position: {self.goal_position}")

        self.roll_pitch_threshold = 0.6

        # Variables for tracking position and distance
        self.prev_x = 0.0
        self.prev_y = 0.0
        self.total_distance_traveled = 0.0
        self.total_spin_accumulated = 0.0  # Track total rotation

        self.episode_rewards = []  # Stores complete episode rewards
        self.current_episode_rewards = {
            'alignment': 0.0,
            'distance': 0.0,
            'goal': 0.0,
            'collision': 0.0,
            'distance_traveled': 0.0,
            'spin': 0.0,
            'total': 0.0
        }

    def extract_goal_position(self):
        self.goal_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal_geom"
        )
        if self.goal_geom_id == -1:
            raise ValueError("Could not find geom named 'goal_geom'")
        return self.data.geom_xpos[self.goal_geom_id].copy()

    def _check_collision(self, group1, group2):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            geom1 = contact.geom1
            geom2 = contact.geom2

            # Check if contact is between any geom in group1 and any geom in group2
            if (geom1 in group1 and geom2 in group2) or \
               (geom2 in group1 and geom1 in group2):
                return True

        return False

    def _check_roll_pitch(self):
        quaternion = self.data.qpos[3:7]  # Get the quaternion (qw, qx, qy, qz)

        rotation = R.from_quat(quaternion[[1, 2, 3, 0]])

        roll, pitch, yaw = rotation.as_euler('xyz')

        # Check absolute values against the threshold
        if abs(roll) > self.roll_pitch_threshold or abs(pitch) > self.roll_pitch_threshold:
            print(
                f"TERMINATION: Roll/Pitch too extreme! Roll: {np.degrees(roll):.2f} deg, Pitch: {np.degrees(pitch):.2f} deg")
            return True
        return False

    def step(self, action):
        # Initialize reward and flags
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        # Extract current state
        current_x = self.data.qpos[0]
        current_y = self.data.qpos[1]

        linear_velocity = self.data.qvel[0]

        # Get angular velocity (spinning)
        # Angular velocity around Z-axis
        angular_velocity = abs(self.data.qvel[5])
        timestep = self.model.opt.timestep
        step_rotation = angular_velocity * timestep  # Rotation this step
        self.total_spin_accumulated += step_rotation

        # Set actuators (left and right wheel speeds)
        self.data.ctrl[self.left_front] = action[0]
        self.data.ctrl[self.left_rear] = action[0]
        self.data.ctrl[self.right_front] = action[1]
        self.data.ctrl[self.right_rear] = action[1]

        mujoco.mj_step(self.model, self.data)

        # Compute observations
        state_obs = np.concatenate([self.data.qpos.flat, self.data.qvel.flat])
        if self.use_lidar:
            lidar_obs = self.lidar.update()['ranges']
            observation = {
                'state': state_obs.astype(np.float32),
                'lidar': lidar_obs.astype(np.float32)
            }
        else:
            observation = state_obs.astype(np.float32)

        # Get new position after step
        new_x = self.data.qpos[0]
        new_y = self.data.qpos[1]

        # Calculate distance traveled this step
        step_distance = np.sqrt((new_x - current_x) **
                                2 + (new_y - current_y)**2)
        self.total_distance_traveled += step_distance

        # Calculate distance to goal
        goal_x, goal_y = self.goal_position[:2]
        distance_to_goal = np.sqrt((new_x - goal_x)**2 + (new_y - goal_y)**2)

        # Distance-based shaping (reward for getting closer)
        prev_distance = np.sqrt((self.prev_x - goal_x)
                                ** 2 + (self.prev_y - goal_y)**2)
        distance_reduction = prev_distance - distance_to_goal

        robot_to_goal_vec_x_world = goal_x - new_x
        robot_to_goal_vec_y_world = goal_y - new_y

        quaternion_after_step = self.data.qpos[3:7]
        rotation_after_step = R.from_quat(quaternion_after_step[[1, 2, 3, 0]])
        _, _, current_heading_rad = rotation_after_step.as_euler('xyz')

        cos_heading = np.cos(-current_heading_rad)
        sin_heading = np.sin(-current_heading_rad)

        robot_to_goal_vec_x_local = robot_to_goal_vec_x_world * cos_heading - robot_to_goal_vec_y_world * sin_heading
        robot_to_goal_vec_y_local = robot_to_goal_vec_x_world * sin_heading + robot_to_goal_vec_y_world * cos_heading
        angle_diff = np.arctan2(robot_to_goal_vec_y_local, robot_to_goal_vec_x_local)

        # Initialize reward components
        reward_components = {
            'distance': 0.0,
            'goal': 0.0,
            'collision': 0.0,
            'time_penalty': 0.0
        }

        # Calculate base rewards
        reward_components['alignment'] = self.rewards["alignment_reward"] * np.cos(angle_diff)
        reward_components['forward'] = self.rewards["forward_motion"] * linear_velocity * np.cos(angle_diff)
        reward_components['angular_penalty'] = self.rewards["angular_penalty"] * (abs(angular_velocity) ** 2)
        reward_components['distance'] = self.rewards["distance"] * distance_reduction
        reward_components['time_penalty'] = self.rewards.get("time_penalty", 0.01)

        # Goal-reaching reward
        if distance_to_goal < 1.0:
            reward_components['goal'] = self.rewards["goal_reached"]
            terminated = True

        # 5. Collision penalties
        if self._check_collision(self.robot_geom_ids, self.obstacle_geom_ids):
            reward_components['collision'] = self.rewards["collision_penalty"]
            terminated = True

        if self._check_roll_pitch():
            reward_components['collision'] = self.rewards["collision_penalty"]
            terminated = True

        # Calculate total reward
        total_reward = sum(reward_components.values())
        reward_components['total'] = total_reward

        for key, value in reward_components.items(): 
            if key in self.current_episode_rewards: 
                self.current_episode_rewards[key] += value
            else:
                self.current_episode_rewards[key] = value 

        self.prev_x = current_x
        self.prev_y = current_y

        info['reward_components'] = reward_components.copy()
        info['distance_to_goal'] = distance_to_goal
        info['total_distance_traveled'] = self.total_distance_traveled
        info['total_spin_accumulated'] = self.total_spin_accumulated
        info['angular_velocity'] = angular_velocity

        return observation, total_reward, terminated, truncated, info

    def get_reward_statistics(self):
        """Returns statistics about rewards across all episodes"""
        if not self.episode_rewards:
            # Return empty statistics if no episodes completed
            return {
                component: {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'total': 0.0
                }
                for component in self.current_episode_rewards.keys()
            }

        stats = {}
        for component in self.episode_rewards[0].keys():
            values = [ep[component] for ep in self.episode_rewards]
            stats[component] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'total': np.sum(values)
            }
        return stats

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.current_episode_rewards['total'] != 0:
            print("\n--- Episode Reward Summary ---")
            for component, value in self.current_episode_rewards.items():
                print(f"{component:>15}: {value:10.2f}")
            print(f"{'Total Distance':>15}: {self.total_distance_traveled:10.2f}m")
            print(f"{'Total Spin':>15}: {self.total_spin_accumulated:10.2f}rad")
            self.episode_rewards.append(self.current_episode_rewards.copy())

        # Reset tracking
        self.current_episode_rewards = {
            k: 0.0 for k in self.current_episode_rewards}
        self.total_distance_traveled = 0.0
        self.total_spin_accumulated = 0.0

        # Generate a new XML file with randomized obstacles and goal position
        # randomize_environment(
        #     env_path=self.xml_file,
        #     min_num_obstacles=3,  # Adjust as needed or parameterize
        #     max_num_obstacles=8  # Adjust as needed or parameterize
        # )

        self.prev_x = 0.0
        self.prev_y = 0.0

        # # Load the new model
        # self.model = mujoco.MjModel.from_xml_path(self.xml_file)
        # self.data = mujoco.MjData(self.model)
        # self.metadata = {"render_modes": ["human", "rgb_array"],
        #                  "render_fps": int(1.0 / (self.model.opt.timestep))}

        # Reinitialize LiDAR if enabled since the model has changed
        if self.use_lidar:
            self.lidar = VLP16Sensor(
                self.model,
                self.data,
                lidar_name="velodyne",
                horizontal_resolution=360.0/self.num_lidar_rays_h,  # Convert ray count to degrees
                rotation_rate=10,  # Hz
                max_range=self.lidar_max_range
                # return_type='ranges'  # Just return ranges for the gym environment
            )

        # Reassign the geometry IDs
        self.robot_geom_ids = []
        self.obstacle_geom_ids = []
        self.goal_id = []

        for i in range(self.model.ngeom):
            if self.model.geom_group[i] == 2:
                if i != self.floor_geom_id:
                    self.robot_geom_ids.append(i)
            elif self.model.geom_group[i] == 1:
                self.obstacle_geom_ids.append(i)
            elif self.model.geom_group[i] == 3:
                self.goal_id.append(i)

        # Reset the viewer if it exists
        if self.viewer:
            self.viewer.close()
            self.viewer = None

        # Reset the model and data
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        # Reassign the goal position
        self.goal_position = self.extract_goal_position()
        print(f"goal position: {self.goal_position}")

        # Get the basic observation
        state_obs = np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

        # Get LiDAR observation if enabled
        if self.use_lidar:
            lidar_obs = self.lidar.update()['ranges']
            observation = {
                'state': state_obs.astype(np.float32),
                'lidar': lidar_obs.astype(np.float32)
            }
        else:
            observation = state_obs.astype(np.float32)

        info = {}
        return observation, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(
                    self.model, self.data)
            else:
                self.viewer.sync()

            # Render LiDAR visualization if enabled
            if self.use_lidar and self.lidar_viz_enabled:
                self.lidar_ax = self.lidar.visualize_scan(ax=self.lidar_ax)

        elif self.render_mode == "rgb_array":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(
                    self.model, self.data)
            self.viewer.sync()
            render_output = self.viewer.read_pixels(
                self.model.width, self.model.height, depth=False)
            return render_output[::-1, :, :]  # Flip upside down

        return None

    def close(self):
        if self.viewer:
            self.viewer.close()
        self.viewer = None
