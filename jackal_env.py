
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

        self.initial_x = 0.0
        self.initial_y = 0.0

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

        # Reset the model and data
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.goal_pose = self.extract_goal_pose()
        print(f"Goal pose: {self.goal_pose}")  # [x, y, yaw] [m, m, rad]

        self.roll_pitch_threshold = 0.6

        self.prev_x = 0.0
        self.prev_y = 0.0

    def extract_goal_pose(self):
        self.goal_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal_geom"
        )
        if self.goal_geom_id == -1:
            raise ValueError("Could not find geom named 'goal_geom'")
        pos = self.data.geom_xpos[self.goal_geom_id].copy()
        orientation = self.data.geom_xmat[self.goal_geom_id].copy()
        rotation = R.from_matrix(orientation.reshape(3, 3))
        yaw = rotation.as_euler('xyz')[2]
        return np.array([pos[0], pos[1], yaw])  # [x, y, yaw]

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

        # Store previous action for smooth motion rewards
        if not hasattr(self, 'prev_action'):
            self.prev_action = np.zeros_like(action)

        # Extract current state
        current_x = self.data.qpos[0]
        current_y = self.data.qpos[1]
        current_heading = self.data.qpos[3]
        ang_vel = self.data.qvel[5]  # Angular velocity (rad/s)

        # Set actuators (left and right wheel speeds)
        self.data.ctrl[self.left_front] = action[0]
        self.data.ctrl[self.left_rear] = action[0]
        self.data.ctrl[self.right_front] = action[1]
        self.data.ctrl[self.right_rear] = action[1]

        mujoco.mj_step(self.model, self.data)

        # Compute observations
        state_obs = np.concatenate([self.data.qpos.flat, self.data.qvel.flat])
        if self.use_lidar:
            lidar_data = self.lidar.update()
            lidar_obs = lidar_data['ranges']
            observation = {
                'state': state_obs.astype(np.float32),
                'lidar': lidar_obs.astype(np.float32)
            }
        else:
            observation = state_obs.astype(np.float32)
            lidar_obs = None

        goal_x, goal_y, goal_yaw = self.goal_pose
        distance_to_goal = np.sqrt(
            (current_x - goal_x)**2 + (current_y - goal_y)**2)
        angle_to_goal = np.arctan2(
            goal_y - current_y, goal_x - current_x) - current_heading
        angle_to_goal = (angle_to_goal + np.pi) % (2 *
                                                   np.pi) - np.pi  # Normalize

        # Calculate goal heading difference
        goal_heading_diff = abs((goal_yaw - current_heading + np.pi))
        goal_heading_diff = (goal_heading_diff + np.pi) % (2 * np.pi) - np.pi

        # ========== CORE REWARDS ==========

        # Goal-reaching reward
        if distance_to_goal < 0.4:
            reward += self.rewards["goal_reached"]
            terminated = True
            return observation, reward, terminated, truncated, info

        # Distance-based progress shaping
        prev_distance = np.sqrt((self.prev_x - goal_x)
                                ** 2 + (self.prev_y - goal_y)**2)
        distance_reduction = prev_distance - distance_to_goal
        reward += self.rewards["distance"] * distance_reduction

        # ========== COLLISION AND SAFETY ==========

        # Collision penalties
        if self._check_collision(self.robot_geom_ids, self.obstacle_geom_ids):
            reward += self.rewards["collision_penalty"]
            terminated = True
            return observation, reward, terminated, truncated, info

        if self._check_roll_pitch():
            reward += self.rewards["collision_penalty"]
            terminated = True
            return observation, reward, terminated, truncated, info

        # ========== SPINNING PREVENTION ==========

        # Strong angular velocity penalty
        reward += self.rewards["spin_penalty"] * (abs(ang_vel) ** 1.5)

        # Direct action-based spinning detection and penalty
        wheel_diff = abs(action[0] - action[1])
        wheel_avg = abs(action[0] + action[1]) / 2.0
        forward_motion = (action[0] + action[1]) / 2.0

        # Penalize pure spinning actions
        if wheel_diff > 0.4 and wheel_avg < 0.2:
            reward += -10.0 * wheel_diff

        # Reward forward motion
        if forward_motion > 0.1:
            reward += 0.3

        # ========== LIDAR-BASED NAVIGATION (DIRECTIONAL) ==========

        if self.use_lidar and lidar_obs is not None:
            # Since LiDAR already provides 360° view, use it for intelligent navigation

            # 1. Obstacle avoidance in movement direction
            # Get forward-facing LiDAR readings (±45° from robot's heading)
            num_rays_h = lidar_obs.shape[1]
            forward_start = int(num_rays_h * 7/8)  # -45°
            forward_end = int(num_rays_h * 1/8)    # +45°

            # Handle wrap-around for forward-facing rays
            if forward_start < forward_end:
                forward_ranges = lidar_obs[:, forward_start:forward_end]
            else:
                forward_ranges = np.concatenate([
                    lidar_obs[:, forward_start:],
                    lidar_obs[:, :forward_end]
                ], axis=1)

            min_forward_distance = np.min(forward_ranges)

            # Penalty for obstacles in forward direction (only when moving forward)
            if forward_motion > 0.1 and min_forward_distance < 1.5:
                obstacle_penalty = (1.5 - min_forward_distance) / 1.5
                reward += self.rewards["obstacle_avoidance_reward"] * \
                    (-obstacle_penalty)

            # 2. Path clearance assessment for goal direction
            # Calculate which LiDAR rays are pointing towards the goal
            goal_angle_global = np.arctan2(
                goal_y - current_y, goal_x - current_x)
            goal_angle_relative = goal_angle_global - current_heading
            goal_angle_relative = (
                goal_angle_relative + np.pi) % (2 * np.pi) - np.pi

            # Convert to LiDAR ray index
            goal_ray_index = int(
                (goal_angle_relative + np.pi) / (2 * np.pi) * num_rays_h) % num_rays_h

            # Check clearance in goal direction (±15°)
            goal_ray_spread = int(num_rays_h * 15 / 360)  # ±15° in ray indices
            goal_start = (goal_ray_index - goal_ray_spread) % num_rays_h
            goal_end = (goal_ray_index + goal_ray_spread) % num_rays_h

            if goal_start < goal_end:
                goal_ranges = lidar_obs[:, goal_start:goal_end]
            else:
                goal_ranges = np.concatenate([
                    lidar_obs[:, goal_start:],
                    lidar_obs[:, :goal_end]
                ], axis=1)

            min_goal_clearance = np.min(goal_ranges)

            # Reward for clear path to goal (only when moving toward goal)
            # Moving toward goal
            if forward_motion > 0.1 and np.cos(angle_to_goal) > 0.5:
                if min_goal_clearance > 2.0:  # Clear path
                    reward += 0.5
                elif min_goal_clearance < 1.0:  # Blocked path
                    reward += -0.5

            # 3. Emergency safety (always applied)
            min_distance_all = np.min(lidar_obs)
            if min_distance_all < 0.4:  # Very close to any obstacle
                reward += -5.0

        # ========== NAVIGATION REWARDS (FORWARD-MOTION CONDITIONED) ==========

        # Only reward alignment and heading when robot is moving forward
        if forward_motion > 0.15:
            reward += self.rewards["alignment_reward"] * np.cos(angle_to_goal)
            reward += self.rewards["goal_heading_reward"] * \
                np.exp(-goal_heading_diff**2)

        # ========== MOTION QUALITY ==========

        # Smooth motion reward
        action_change = np.linalg.norm(action - self.prev_action)
        reward += self.rewards["smooth_motion_reward"] * np.exp(-action_change)

        # Jerk penalty
        if action_change > 0.8:
            reward += self.rewards["jerk_penalty"] * action_change

        # ========== PROGRESS REWARDS ==========

        # Path efficiency (only when making progress)
        if distance_reduction > 0 and forward_motion > 0.1:
            efficiency = distance_reduction / (np.linalg.norm(action) + 1e-6)
            reward += self.rewards["path_efficiency_reward"] * efficiency

        # Progress milestones
        if not hasattr(self, 'closest_distance'):
            self.closest_distance = distance_to_goal

        if distance_to_goal < self.closest_distance - 0.5:
            reward += self.rewards["progress_milestone_reward"]
            self.closest_distance = distance_to_goal

        # ========== PENALTIES ==========

        # Still penalty
        if abs(distance_reduction) < 0.05:
            reward += self.rewards["still_penalty"]

        # Time step penalty
        reward += self.rewards["time_step_penalty"]

        # Update previous values
        self.prev_x, self.prev_y = current_x, current_y
        self.prev_action = action.copy()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Generate a new XML file with randomized obstacles and goal position
        randomize_environment(
            env_path=self.xml_file,
            min_num_obstacles=3,  # Adjust as needed or parameterize
            max_num_obstacles=8  # Adjust as needed or parameterize
        )

        # Get initial positions for displacement
        self.initial_x = self.data.qpos[0]
        self.initial_y = self.data.qpos[1]

        self.prev_x = 0.0
        self.prev_y = 0.0

        # Load the new model
        self.model = mujoco.MjModel.from_xml_path(self.xml_file)
        self.data = mujoco.MjData(self.model)
        self.metadata = {"render_modes": ["human", "rgb_array"],
                         "render_fps": int(1.0 / (self.model.opt.timestep))}

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
        self.goal_pose = self.extract_goal_pose()
        print(f"goal pose: {self.goal_pose}")

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
