
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

        # Initialize reward plots
        plt.ion()
        self.fix, self.ax = plt.subplots(
            nrows=2, ncols=3, figsize=(12, 8), tight_layout=True)
        self.ax = self.ax.flatten()

        self.component_names = [
            "Distance Reward",
            "Spin Penalty",
            "Alignment Reward",
            "Still Penalty",
            "Total Reward"
        ]
        self.reward_history = {name: [] for name in self.component_names}
        self.lines = {}
        for i, comp in enumerate(self.component_names):
            self.lines[comp], = self.ax[i].plot(
                [], [], "-o", markersize=2, linewidth=1, label=comp)
            self.ax[i].set_title(comp)
            self.ax[i].set_xlabel("Time Steps")
            self.ax[i].set_ylabel("Reward")
            self.ax[i].legend()
            self.ax[i].grid()
        plt.show()
        plt.pause(0.001)

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
        """Preprocess LiDAR data for better neural network training"""

        # 1. Normalize to [0, 1] range
        normalized_ranges = lidar_ranges / self.lidar_max_range

        # 2. Apply inverse distance weighting for better obstacle representation
        # Objects closer than 1m get exponentially higher values
        inverse_ranges = 1.0 - normalized_ranges
        weighted_ranges = np.where(normalized_ranges < 0.1,
                                   inverse_ranges ** 2,
                                   inverse_ranges)

        # 3. Optional: Apply Gaussian smoothing to reduce noise
        from scipy import ndimage
        smoothed_ranges = ndimage.gaussian_filter(weighted_ranges, sigma=0.5)

        return smoothed_ranges

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

        # ========== 1. GOAL SUCCESS (Terminal) ==========
        if distance_to_goal < 0.4:
            reward += self.rewards["goal_reached"]
            terminated = True
            return observation, reward, terminated, truncated, info

        # ========== 2. PROGRESS TOWARD GOAL (Core Learning Signal) ==========
        prev_distance = np.sqrt((self.prev_x - goal_x)
                                ** 2 + (self.prev_y - goal_y)**2)
        distance_reduction = prev_distance - distance_to_goal
        reward += self.rewards["distance_progress"] * distance_reduction

        # ========== 3. DIRECTIONAL ALIGNMENT (Navigation Guidance) ==========
        # Reward pointing toward goal (works regardless of motion)
        reward += self.rewards["alignment"] * np.cos(angle_to_goal)

        # ========== 4. SAFETY (Collision Avoidance) ==========
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

        # ========== 5. PREVENT EXCESSIVE SPINNING ==========
        # Only penalize when spinning without purpose (not aligned AND spinning fast)
        if abs(angle_to_goal) < 0.2 and abs(ang_vel) > 1.0:  # Well aligned but still spinning
            reward += self.rewards["excessive_spin_penalty"] * abs(ang_vel)

        # ========== 6. TIME EFFICIENCY ==========
        reward += self.rewards["time_penalty"]

        # Update tracking variables
        self.prev_x, self.prev_y = current_x, current_y
        if not hasattr(self, 'prev_action'):
            self.prev_action = np.zeros_like(action)
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
