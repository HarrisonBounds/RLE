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
STEPS_PER_BATCH = 512
MAX_STEPS = STEPS_PER_BATCH * 4
             
# --- Logging & Saving ---
LOG_INTERVAL_EPISODES = 10   
SAVE_MODEL_INTERVAL_STEPS = 100000
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Device configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

#---------------------------------------------------------------------------------------------------

# Set up Env
env = Jackal_Env(
    xml_file="jackal_obstacles.xml",
    render_mode="human",
    use_lidar=True # Keeping this as True, so env returns a dictionary
)

# Get observation and action dimensions
print("Observation space: ", env.observation_space)
obs_space = env.observation_space
action_space = env.action_space

# Calculate state_dim correctly for a Dict observation space
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
raw_observation, info = env.reset() # Renamed to raw_observation for clarity
# Process the Dict observation for the agent input
processed_state = np.concatenate([raw_observation['state'].flatten(), raw_observation['lidar'].flatten()])

# --- Training Loop Variables ---
global_step = 0         
episode_reward_sum = 0      
episode_steps = 0           
episode_count = 0 

#---------------------------------------------------------------------------------------------------

print("\n--- Starting PPO Training ---")
try:
    while global_step < TOTAL_TIMESTEPS:
        batch_steps = 0
        
        while batch_steps < STEPS_PER_BATCH:
            print(f"Global step: {global_step}")

            if global_step >= MAX_STEPS:
                truncated = True

            # Pass the ALREADY PROCESSED (flattened/concatenated) state to select_action
            action, log_prob = agent.select_action(processed_state)
            
            #action = [0.5, 0.5] 
            
            # Take environment step
            raw_new_observation, reward, terminated, truncated, info = env.step(action)
            env.render()

            # Process the new raw observation for the agent and buffer
            new_processed_state = np.concatenate([raw_new_observation['state'].flatten(), raw_new_observation['lidar'].flatten()])

            done_flag = True if terminated or truncated else False

            # Store experience in the agent's replay buffer
            agent.buffer.store(
                processed_state, # Store the processed state
                action,
                reward,
                new_processed_state, # Store the new processed state
                done_flag,
                log_prob
            )

            episode_reward_sum += reward
            episode_steps += 1
            global_step += 1      
            batch_steps += 1

            # Update current processed state for the next step
            processed_state = new_processed_state

            if terminated or truncated:
                episode_count += 1

                print(f"Episode {episode_count} | Total T: {global_step} | Steps: {episode_steps} | Reward: {episode_reward_sum:.2f}")

                # Reset environment and get new raw observation
                raw_observation, info = env.reset()
                # Process the new initial raw observation for the next step
                processed_state = np.concatenate([raw_observation['state'].flatten(), raw_observation['lidar'].flatten()])

                # Reset episode specific counters
                episode_reward_sum = 0 
                episode_steps = 0

                break

        if len(agent.buffer.states) > 0: 
            agent.update()

        if global_step % SAVE_MODEL_INTERVAL_STEPS == 0:
            torch.save(agent.actor.state_dict(), os.path.join(MODEL_DIR, f"actor_step_{global_step}.pth"))
            torch.save(agent.critic.state_dict(), os.path.join(MODEL_DIR, f"critic_step_{global_step}.pth"))
            print(f"Models saved at {global_step} timesteps.")

except KeyboardInterrupt:
    print("\nTraining interrupted by user.")
finally:
    env.close()
    print("Environment closed.")