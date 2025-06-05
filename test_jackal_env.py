import gymnasium as gym
from jackal_env import Jackal_Env
import time
import mujoco
from ppo import PPOAgent
import numpy as np
import torch
import os

# --- Training Hyperparameters ---
TOTAL_TIMESTEPS = 1_000_000
STEPS_PER_BATCH = 512 # Number of environment steps to collect before a PPO update

# --- Logging & Saving ---
LOG_INTERVAL_EPISODES = 10
SAVE_MODEL_INTERVAL_STEPS = STEPS_PER_BATCH
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Device configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ---------------------------------------------------------------------------------------------------

# Set up Env
env = Jackal_Env(
    xml_file="jackal_obstacles.xml",
    render_mode=None,
    use_lidar=True
)

# Get observation and action dimensions
obs_space = env.observation_space
action_space = env.action_space

state_dim = 0
for key, space in obs_space.spaces.items():
    state_dim += np.prod(space.shape)
action_dim = np.prod(action_space.shape)
print(f"State dim: {state_dim}, Action Dim: {action_dim}")

# --- Declare PPO Agent ---
agent = PPOAgent(
    state_dim,
    action_dim,
    lr_actor=1e-4,
    lr_critic=1e-4,
    gamma=0.995,
    epsilon_clip=0.2,
    K_epochs=10,
    entropy_coef=0.01,
    gae_lambda=0.98,
    action_space_low=action_space.low,
    action_space_high=action_space.high
)

agent.actor.to(DEVICE)
agent.critic.to(DEVICE)

# Reset env and get first observation
raw_observation, info = env.reset()
processed_state = np.concatenate(
    [raw_observation['state'].flatten(), raw_observation['lidar'].flatten()])

# --- Training Loop Variables ---
global_step = 0
episode_reward_sum = 0
episode_steps_this_episode = 0 
episode_count = 0
batch_number = 0

# ---------------------------------------------------------------------------------------------------

print("\n--- Starting PPO Training ---")
try:
    while global_step < TOTAL_TIMESTEPS:
        steps_collected_in_this_segment = 0 
        
        while steps_collected_in_this_segment < STEPS_PER_BATCH:
            action, log_prob = agent.select_action(processed_state)

            raw_new_observation, reward, terminated, truncated, info = env.step(action)
            env.render()

            new_processed_state = np.concatenate(
                [raw_new_observation['state'].flatten(), raw_new_observation['lidar'].flatten()])

            done_flag = True if terminated or truncated else False

            agent.buffer.store(
                processed_state,
                action,
                info['reward_components'], 
                new_processed_state,
                done_flag,
                log_prob
            )

            episode_reward_sum += reward
            episode_steps_this_episode += 1
            global_step += 1 
            steps_collected_in_this_segment += 1

            processed_state = new_processed_state

            if terminated or truncated:
                episode_count += 1
                print(
                    f"Episode {episode_count} | Total T: {global_step} | Steps: {episode_steps_this_episode} | Reward: {episode_reward_sum:.2f}")

                raw_observation, info = env.reset()
                processed_state = np.concatenate(
                    [raw_observation['state'].flatten(), raw_observation['lidar'].flatten()])

                episode_reward_sum = 0
                episode_steps_this_episode = 0
                
               
                break 

        if len(agent.buffer.states) > 0:
            batch_number += 1
            batch_component_summary, actual_batch_size = agent.update() # agent.update() will clear the buffer
            
            print(f"\n--- PPO Batch Update {batch_number} Completed ({actual_batch_size} steps) ---")
            
            print("Reward Contributions for Current Batch:")
            for component, total_value in batch_component_summary.items():
                print(f"  {component:<15}: {total_value:.2f}")

        if global_step % SAVE_MODEL_INTERVAL_STEPS == 0 and global_step > 0:
            torch.save(agent.actor.state_dict(), os.path.join(
                MODEL_DIR, f"actor_step_{global_step}.pth"))
            torch.save(agent.critic.state_dict(), os.path.join(
                MODEL_DIR, f"critic_step_{global_step}.pth"))
            print(f"Models saved at {global_step} timesteps.")

except KeyboardInterrupt:
    print("\nTraining interrupted by user.")
finally:
    env.close()
    print("Environment closed.")