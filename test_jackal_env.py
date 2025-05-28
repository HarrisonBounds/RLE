import gymnasium as gym
from jackal_env import Jackal_Env
import time
import mujoco
from ppo import PPOAgent
import numpy as np
import torch

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

total_steps = 2048 
current_steps = 0

max_episodes = 5
episode_reward = 0
episode_steps = 0
episode_count = 0
terminated_episode = False
truncated_episode = False
done_flag = False

#Drive forward
while True:
    #Select action and take environment step
    action, log_prob = agent.select_action(observation)
    new_observation, reward, terminated, truncated, info = env.step(action)

    #Gather state information
    current_processed_state = np.concatenate([observation['state'].flatten(), observation['lidar'].flatten()])
    new_processed_state = np.concatenate([new_observation['state'].flatten(), new_observation['lidar'].flatten()])

    if terminated or truncated:
        done = True
        break

    #Store experience
    agent.buffer.store(
        current_processed_state,
        action,
        reward,
        new_processed_state,
        done_flag,
        log_prob
    )

    #Update env information
    observation = new_observation
    episode_reward += reward
    episode_steps += 1
    current_steps += 1

    env.render()

    if current_steps >= total_steps:
        #Update agent
        agent.update()

        current_steps = 0

        if terminated or truncated:
            episode_count += 1

            if episode_count >= max_episodes and current_steps == 0:
                print(f"Reached max_episodes_for_test ({max_episodes}). Stopping.")
                break
            
            # Reset environment for a new episode
            observation, info = env.reset()
            episode_reward = 0
            episode_steps = 0

env.close()

