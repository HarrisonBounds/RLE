import gymnasium as gym
from jackal_env import Jackal_Env
import time
import mujoco
from ppo import PPOAgent
import numpy as np

#Set up Env
env = Jackal_Env(
    xml_file="jackal_obstacles.xml",
    render_mode="human",
    use_lidar=True
)

#Get observation and action dimensions
print("Observation space: ", env.observation_space)
obs_space = env.observation_space
action_space = env.action_space
state_dim = 0
for key, space in obs_space.spaces.items():
        state_dim += np.prod(space.shape)

action_dim = np.prod(action_space.shape)
print(f"State dim: {state_dim}, Action Dim: {action_dim}")

#Declare PPO Agent
agent = PPOAgent(state_dim, action_dim)

#Reset env
observation, info = env.reset()


# Drive forward
while True:
    action, log_prob = agent.select_action(observation)
    new_observation, reward, terminated, truncated, info = env.step(action)

    observation = new_observation

    if terminated or truncated:
        break

    env.render()

env.close()

