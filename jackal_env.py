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

        self.goal_position = self.assign_goal_position()
        print(f"Goal position: {self.goal_position}")

        self.roll_pitch_threshold = 0.6

    def assign_goal_position(self):
        self.goal_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal_geom"
        )
        if self.goal_geom_id == -1:
            raise ValueError("Could not find geom named 'goal_geom'")
        self.goal_position = self.data.geom_xpos[self.goal_geom_id]

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
        # Initialize reward FIRST
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        x_vel = self.data.qvel[0]
        angular_vel = self.data.qvel[5]
        current_x = self.data.qpos[0]
        current_y = self.data.qpos[1]

        # Calculate actual displacement
        displacement = np.sqrt((current_x - self.prev_x)
                               ** 2 + (current_y - self.prev_y)**2)
        self.prev_x = current_x
        self.prev_y = current_y

        assert len(action) == 2, "Action should be [left_speed, right_speed]"

        # Set actuators
        self.data.ctrl[self.left_front] = action[0]
        self.data.ctrl[self.left_rear] = action[0]
        self.data.ctrl[self.right_front] = action[1]
        self.data.ctrl[self.right_rear] = action[1]

        mujoco.mj_step(self.model, self.data)

        # Basic observation
        state_obs = np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

        # LiDAR observation (still collected for the agent, but not for reward)
        if self.use_lidar:
            lidar_obs = self.lidar.update()['ranges']
            observation = {
                'state': state_obs.astype(np.float32),
                'lidar': lidar_obs.astype(np.float32)
            }

            frontal_sector = np.concatenate([
                lidar_obs[0, :30],    # Left-front
                lidar_obs[0, 330:360]  # Right-front
            ])
        else:
            observation = state_obs.astype(np.float32)

        if self._check_collision(self.robot_geom_ids, self.obstacle_geom_ids):
            reward += self.rewards["collision"]
            terminated = True

        if self._check_roll_pitch():
            reward += self.rewards["collision"]
            terminated = True

        # Rewards
        # Time step penalty
        reward += self.rewards["time_step"]

        # Displacement penalty (dont take too long for a path to goal)
        total_displacement = np.sqrt(
            (current_x - self.initial_x)**2 + (current_y - self.initial_y)**2
        )
        reward += total_displacement * self.rewards["displacement_penalty"]

        # Reward inverse distance to goal

        # Terminate if goal is reached
        reached_goal = self._check_collision(self.robot_geom_ids, self.goal_id)
        if reached_goal:
            reward += self.rewards["goal_reached"]
            terminated = True

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

        # Reassign the goal position
        self.goal_position = self.assign_goal_position()

        # Reset the viewer if it exists
        if self.viewer:
            self.viewer.close()
            self.viewer = None

        # Reset the model and data
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

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

        # Close any open matplotlib figures
        plt.close('all')
