import gymnasium as gym
from jackal_env import Jackal_Env
import time
import mujoco
from ppo import PPOAgent
import numpy as np
import torch
import os



TOTAL_TIMESTEPS = 1_000_000   
STEPS_PER_BATCH = 2048                

# Logging & Saving
LOG_INTERVAL_EPISODES = 10   # Log training progress every X episodes
SAVE_MODEL_INTERVAL_STEPS = 100000 # Save model every X total timesteps
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

#---------------------------------------------------------------------------------------------------

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

agent.actor.to(DEVICE)
agent.critic.to(DEVICE)
agent.device = DEVICE

#Reset env and get first state
observation, info = env.reset()
processed_state = np.concatenate([observation['state'].flatten(), observation['lidar'].flatten()])

global_step = 0         
episode_reward_sum = 0      
episode_steps = 0           
episode_count = 0 

#---------------------------------------------------------------------------------------------------

#Drive forward
while global_step < TOTAL_TIMESTEPS:
    batch_steps = 0

    while batch_steps < STEPS_PER_BATCH:
        #Select action and take environment step
        action, log_prob = agent.select_action(observation)

        new_observation, reward, terminated, truncated, info = env.step(action)
        env.render()

        #Gather state information
        new_processed_state = np.concatenate([new_observation['state'].flatten(), new_observation['lidar'].flatten()])

        if terminated or truncated:
            done_flag = True
        else:
            done_flag = False

        #Store experience
        agent.buffer.store(
            processed_state,
            action,
            reward,
            new_processed_state,
            done_flag,
            log_prob
        )

        #Update env information
        episode_reward_sum += reward
        episode_steps += 1
        global_step += 1      
        batch_steps += 1

        # Update current observation for the next step
        observation = new_observation
        processed_state = new_processed_state

        if terminated or truncated:
            episode_count += 1

            # Log episode stats
            if episode_count % LOG_INTERVAL_EPISODES == 0:
                print(f"Episode {episode_count} | Total T: {global_step} | Steps: {episode_steps} | Reward: {episode_reward_sum:.2f}")

            # Reset environment for a new episode
            observation, info = env.reset()

            # Process the new initial observation for the next step
            processed_state = np.concatenate([observation['state'].flatten(), observation['lidar'].flatten()])

            # Reset episode specific counters
            episode_reward_sum = 0 
            episode_steps = 0

            break

        if len(agent.buffer.states) > 0: 
            agent.update()

        

env.close()

