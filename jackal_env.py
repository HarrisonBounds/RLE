import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer


class Jackal_Env(gym.Env):
    def __init__(self, xml_file="test.xml", render_mode=None):
        super().__init__()

        # Load the MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)
        self.metadata = {"render_modes": [
            "human", "rgb_array"], "render_fps": self.model.opt.timestep / 1000}

        # Define state (observation) and action spaces
        obs_size = self.model.nq + self.model.nv  # joint positions + velocities
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        # For differential drive, we only need 2 actions (left and right)
        # Even though the model has 4 actuators
        # action_size = self.model.nu  # number of actuators
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

    def step(self, action):
        # Map the 2D action (left, right) to the 4 wheel actuators
        # If you're using equality constraints in the XML file, we technically only need to
        # set the front actuators, but we'll set all four for completeness

        # Ensure action is the right shape
        assert len(action) == 2, "Action should be [left_speed, right_speed]"

        # Set left side actuators
        self.data.ctrl[self.left_front] = action[0]
        self.data.ctrl[self.left_rear] = action[0]

        # Set right side actuators
        self.data.ctrl[self.right_front] = action[1]
        self.data.ctrl[self.right_rear] = action[1]

        # Simulate one step
        mujoco.mj_step(self.model, self.data)

        # Get the new observation
        observation = np.concatenate(
            [self.data.qpos.flat, self.data.qvel.flat]
        )

        reward = 0.0
        terminated = False
        truncated = False
        info = {}  # info dict for the same format as gymnasium

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        observation = np.concatenate(
            [self.data.qpos.flat, self.data.qvel.flat]
        )
        info = {}  # info dict for the same format as gymnasium

        return observation, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(
                    self.model, self.data)
            else:
                self.viewer.sync()
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
