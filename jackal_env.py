import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer

class Jackal_Env(gym.Env):
    def __init__(self, xml_file="test.xml", render_mode=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)

        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": self.model.opt.timestep / 1000}

        #Define state (observation) and action spaces
        obs_size = self.model.nq + self.model.nv # joint positions + velocities
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        action_size = self.model.nu #number of actuators
        self.action_space = spaces.Box(low=-1, high=1, shape=(action_size,), dtype=np.float32)

        self.viewer = None
        self.render_mode = render_mode

    def step(self, action):
        # Apply the action to the model
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        # Get the new observation
        observation = np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

        reward = 0.0

        terminated = False
        truncated = False

        return observation, reward, terminated, truncated
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        observation = np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

        return observation

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            else:
                self.viewer.sync()
        elif self.render_mode == "rgb_array":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            render_output = self.viewer.read_pixels(self.model.width, self.model.height, depth=False)
            return render_output[::-1, :, :] # Flip upside down
        return None
    
    def close(self):
        if self.viewer:
            self.viewer.close()
        self.viewer = None