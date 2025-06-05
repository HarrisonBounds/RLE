import gymnasium as gym
from jackal_env import Jackal_Env
import mujoco
import numpy as np
import torch
import os
import time

from ppo import PPOAgent 

TRAINED_ACTOR_PATH = "./models/actor_step_XXXXXX.pth" 

DEPLOY_RENDER_MODE = "human" 
NUM_EVAL_EPISODES = 10 

# --- Device configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ---------------------------------------------------------------------------------------------------

def deploy_policy():
    # Set up Env
    env = Jackal_Env(
        xml_file="jackal_obstacles.xml",
        render_mode=DEPLOY_RENDER_MODE,
        use_lidar=True
    )

    obs_space = env.observation_space
    action_space = env.action_space

    state_dim = 0
    for key, space in obs_space.spaces.items():
        state_dim += np.prod(space.shape)
    action_dim = np.prod(action_space.shape)
    print(f"State dim: {state_dim}, Action Dim: {action_dim}")

  
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

    # Load the trained actor's state_dict
    try:
        agent.actor.load_state_dict(torch.load(TRAINED_ACTOR_PATH, map_location=DEVICE))
        print(f"Successfully loaded trained actor from: {TRAINED_ACTOR_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {TRAINED_ACTOR_PATH}. Please check the path.")
        return # Exit if model not found
    
    # Set the actor to evaluation mode (important for networks with Dropout, BatchNorm, etc.)
    agent.actor.eval() 

    # --- Evaluation Loop ---
    print(f"\n--- Starting Policy Deployment for {NUM_EVAL_EPISODES} Episodes ---")
    
    total_rewards = []
    total_steps = []

    for episode_num in range(NUM_EVAL_EPISODES):
        raw_observation, info = env.reset()
        processed_state = np.concatenate(
            [raw_observation['state'].flatten(), raw_observation['lidar'].flatten()])
        
        episode_reward = 0
        episode_steps = 0
        
        while True:
            action, _ = agent.select_action(processed_state, evaluate=True) 

            raw_new_observation, reward, terminated, truncated, info = env.step(action)
            env.render() 

            new_processed_state = np.concatenate(
                [raw_new_observation['state'].flatten(), raw_new_observation['lidar'].flatten()])
            
            episode_reward += reward
            episode_steps += 1
            processed_state = new_processed_state

            if terminated or truncated:
                print(f"Episode {episode_num + 1} finished in {episode_steps} steps with reward: {episode_reward:.2f}")
                total_rewards.append(episode_reward)
                total_steps.append(episode_steps)
                break
        
        time.sleep(1)

    env.close()
    print("\n--- Deployment Evaluation Complete ---")
    print(f"Average reward over {NUM_EVAL_EPISODES} episodes: {np.mean(total_rewards):.2f}")
    print(f"Average steps per episode: {np.mean(total_steps):.2f}")

if __name__ == "__main__":
    deploy_policy()