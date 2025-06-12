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
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
import os


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
            self.observation_space = spaces.Dict({
                'state': spaces.Box(low=-np.inf, high=np.inf, shape=(basic_obs_size,), dtype=np.float32),
                'lidar': spaces.Box(low=0, high=1.0, shape=(num_lidar_rays_h,), dtype=np.float32)
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

        # Initialize/reset distance tracking with current robot position
        self.prev_robot_x = self.data.qpos[0]
        self.prev_robot_y = self.data.qpos[1]
        self.prev_x = self.data.qpos[0]
        self.prev_y = self.data.qpos[1]

        self.goal_pose = self.extract_goal_pose()
        print(f"Goal pose: {self.goal_pose}")  # [x, y, yaw] [m, m, rad]

        self.roll_pitch_threshold = 0.6

        self.prev_x = 0.0
        self.prev_y = 0.0

        # Two-stage goal achievement
        self.position_threshold = 0.6  # meters for position achievement
        # radians for orientation achievement (~11 degrees)
        self.orientation_threshold = 0.2
        self.position_achieved = False

        # Distance tracking for efficiency penalty
        self.total_distance_traveled = 0.0
        self.prev_robot_x = 0.0
        self.prev_robot_y = 0.0

        # Saving reward history for plotting after training
        self.reward_dictionary = {
            "episodes": [],
            "total_reward": [],
            "distance_progress": [],
            "alignment": [],
            "distance_traveled_penalty": [],
        }

    def get_reward_history(self):
        return self.reward_dictionary

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

    def _preprocess_lidar(self, lidar_ranges):
        """
        Compress LiDAR from 16x360 → 360 by taking the minimum range across elevations.
        Optionally normalize and weight the values for better learning.
        """
        # Take the minimum range across vertical channels (axis 0)
        compressed_ranges = np.min(lidar_ranges, axis=0)  # Shape: (360,)

        # Normalize to [0, 1]
        normalized = compressed_ranges / self.lidar_max_range
        normalized = np.clip(normalized, 0.0, 1.0)

        # Optional inverse weighting for obstacle emphasis
        weighted = 1.0 - normalized  # Close = high value

        # Optional smoothing (Gaussian blur)
        smoothed = gaussian_filter1d(weighted, sigma=1.0)
        # If the file doesn't exist, plot and save the profile
        if not os.path.exists("lidar_profile.png"):
            plt.plot(smoothed)
            plt.title("BEV LiDAR Profile")
            plt.xlabel("Azimuth Angle Index (0° to 360°)")
            plt.ylabel("Inverse Depth")
            plt.grid()
            plt.savefig("lidar_profile.png")
        return smoothed.astype(np.float32)  # Shape: (360,)

    def _extract_lidar_features(self, lidar_ranges):
        """Extract meaningful features from LiDAR data"""

        # Flatten the 2D LiDAR data
        flat_ranges = lidar_ranges.flatten()

        # Basic features
        min_distance = np.min(flat_ranges)
        mean_distance = np.mean(flat_ranges)

        # Directional features (divide into sectors)
        num_sectors = 8
        sector_size = len(flat_ranges) // num_sectors
        sector_mins = []

        for i in range(num_sectors):
            start_idx = i * sector_size
            end_idx = start_idx + sector_size
            sector_min = np.min(flat_ranges[start_idx:end_idx])
            sector_mins.append(sector_min)

        # Obstacle density (percentage of close readings)
        close_threshold = 2.0  # meters
        obstacle_density = np.sum(
            flat_ranges < close_threshold) / len(flat_ranges)

        # Combine features
        features = np.array([min_distance, mean_distance,
                            obstacle_density] + sector_mins)

        return features

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

    def smooth_action(self, action, alpha=0.8):
        # a_smooth alpha * prev_action + (1 - alpha) * a_raw
        if not hasattr(self, 'prev_action'):
            self.prev_action = np.zeros_like(action)
        smoothed_action = alpha * self.prev_action + (1 - alpha) * action
        # Clip to action space limits
        smoothed_action = np.clip(
            smoothed_action, self.action_space.low, self.action_space.high)
        return smoothed_action

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
        # Quaternion (qw, qx, qy, qz)
        current_orientation = self.data.qpos[3:7]
        rotation = R.from_quat(current_orientation[[1, 2, 3, 0]])
        current_heading = rotation.as_euler('xyz')[2]  # Yaw (rad)
        ang_vel = self.data.qvel[5]  # Angular velocity (rad/s)
        reward += -0.2 * abs(ang_vel)
        # Track total distance traveled for efficiency penalty
        distance_step = 0.0
        if hasattr(self, 'prev_robot_x') and hasattr(self, 'prev_robot_y'):
            distance_step = np.sqrt((current_x - self.prev_robot_x)**2 +
                                    (current_y - self.prev_robot_y)**2)
            self.total_distance_traveled += distance_step

        # Update previous robot position for distance tracking
        self.prev_robot_x = current_x
        self.prev_robot_y = current_y

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

        # Calculate goal heading difference (for final orientation alignment)
        goal_heading_diff = abs(
            (goal_yaw - current_heading + np.pi) % (2 * np.pi) - np.pi)

        # ========== TWO-STAGE GOAL ACHIEVEMENT ==========

        # Stage 1: Check if position is achieved
        if distance_to_goal < self.position_threshold:
            if not self.position_achieved:
                self.position_achieved = True
                # Partial reward for reaching position
                reward += self.rewards["goal_reached"] * 0.5
                print(f"Position achieved! Now aligning orientation...")

            # Stage 2: Check orientation alignment when at position
            if goal_heading_diff < self.orientation_threshold:
                reward += self.rewards["goal_reached"] * \
                    0.5  # Complete the reward
                terminated = True
                print(f"Goal fully achieved! Position and orientation aligned.")
                print(
                    f"Total distance traveled: {self.total_distance_traveled:.2f}m")
                return observation, reward, terminated, truncated, info
        else:
            self.position_achieved = False

        # ========== PROGRESS TOWARD GOAL (Core Learning Signal) ==========
        prev_distance = np.sqrt((self.prev_x - goal_x)
                                ** 2 + (self.prev_y - goal_y)**2)
        # distance_reduction = prev_distance - distance_to_goal
        distance_reduction = np.clip(
            prev_distance - distance_to_goal, -0.5, 0.5)

        reward += self.rewards["distance_progress"] * distance_reduction

        # ========== DIRECTIONAL ALIGNMENT ==========
        if not self.position_achieved:
            # When far from goal, encourage pointing toward goal position
            reward += self.rewards["alignment"] * np.cos(angle_to_goal)
        else:
            # When at goal position, encourage aligning with goal orientation
            orientation_alignment = np.cos(goal_heading_diff)
            # Higher weight for final alignment
            reward += self.rewards["alignment"] * 2.0 * orientation_alignment

        # ========== SAFETY (Collision Avoidance) ==========
        if self._check_collision(self.robot_geom_ids, self.obstacle_geom_ids):
            reward += self.rewards["collision_penalty"]
            terminated = True
            return observation, reward, terminated, truncated, info

        if self._check_roll_pitch():
            reward += self.rewards["collision_penalty"]
            terminated = True
            return observation, reward, terminated, truncated, info

        # Emergency obstacle avoidance (LiDAR-based)
        if self.use_lidar and lidar_obs is not None:
            min_distance_all = np.min(lidar_obs)
            if min_distance_all < 0.3:  # Very close to obstacle
                reward += -10.0

        # ========== TIME EFFICIENCY ==========
        reward += self.rewards["time_penalty"]

        # ========== DISTANCE EFFICIENCY PENALTY ==========
        # Penalize total distance traveled to encourage efficient paths
        reward += self.rewards["distance_traveled_penalty"] * distance_step

        # Update tracking variables
        self.prev_x, self.prev_y = current_x, current_y
        if not hasattr(self, 'prev_action'):
            self.prev_action = np.zeros_like(action)
        self.prev_action = action.copy()

        # Add distance info for monitoring
        info['total_distance_traveled'] = self.total_distance_traveled
        info['distance_to_goal'] = distance_to_goal
        info['position_achieved'] = self.position_achieved

        print(f"Current Reward: {reward}")

        # Update reward history for plotting
        self.reward_dictionary["episodes"].append(
            len(self.reward_dictionary["total_reward"]) + 1)
        self.reward_dictionary["total_reward"].append(reward)
        self.reward_dictionary["distance_progress"].append(
            self.rewards["distance_progress"] * distance_reduction)
        self.reward_dictionary["alignment"].append(
            self.rewards["alignment"] * (np.cos(angle_to_goal) if not self.position_achieved else 2.0 * np.cos(goal_heading_diff)))
        self.reward_dictionary["distance_traveled_penalty"].append(
            self.rewards["distance_traveled_penalty"] * distance_step)

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset goal achievement tracking
        self.position_achieved = False

        # Reset distance tracking
        self.total_distance_traveled = 0.0
        self.prev_robot_x = 0.0
        self.prev_robot_y = 0.0

        # Generate a new XML file with randomized obstacles and goal position
        randomize_environment(
            env_path=self.xml_file,
            min_num_obstacles=3,  # Adjust as needed or parameterize
            max_num_obstacles=8  # Adjust as needed or parameterize
        )

        # Get initial positions for displacement
        self.initial_x = self.data.qpos[0]
        self.initial_y = self.data.qpos[1]

        # Reset tracking variables
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
                    self.model, self.data, show_left_ui=False, show_right_ui=False)
                # Set camera to show birds-eye view
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                self.viewer.cam.lookat[:] = [
                    # Look at the robot
                    self.data.qpos[0], self.data.qpos[1], 0.5]
                self.viewer.cam.distance = 16.0  # Distance from the robot
                self.viewer.cam.azimuth = 0.0
                self.viewer.cam.elevation = -90.0
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
