import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer
import torch
from torch.optim import Adam
from jackal_env import Jackal_Env
from ppo import Actor, Critic, ReplayBuffer, ppo_update

# Initialize environment, actor, critic, and optimizers
env = Jackal_Env(xml_file="jackal_velodyne.xml", render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
actor = Actor(state_dim, action_dim)
critic = Critic(state_dim)
optimizer_actor = Adam(actor.parameters(), lr=1e-4)
optimizer_critic = Adam(critic.parameters(), lr=1e-3)
replay_buffer = ReplayBuffer()

# Hyperparameters
ppo_epochs = 10
mini_batch_size = 64
total_timesteps = 100000
clip_param = 0.2
gamma = 0.99
gae_lambda = 0.95
print_interval = 1000
save_interval = 5000

# Main training loop
state, _ = env.reset()
for t in range(total_timesteps):
    # 1. Get action from the actor
    state_tensor = torch.tensor(state, dtype=torch.float32)
    action_mean, action_std = actor(state_tensor)
    dist = torch.distributions.Normal(action_mean, action_std)
    action = dist.sample()
    log_prob = dist.log_prob(action).sum(dim=-1) #get log prob

    # Clip the action to the action space bounds
    action = np.clip(action.detach().numpy(), env.action_space.low, env.action_space.high)

    # 2. Take a step in the environment
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # 3. Store the transition in the replay buffer
    replay_buffer.store(state, action, reward, next_state, done, log_prob.detach().numpy()) #store log prob

    state = next_state

    # 4. If the episode is done, reset the environment
    if done:
        state, _ = env.reset()

    # 5. If enough data is collected, update the policy
    if len(replay_buffer.states) >= mini_batch_size:
        batch = replay_buffer.get_batch()
        ppo_update(actor, critic, optimizer_actor, optimizer_critic, batch, clip_param, ppo_epochs, mini_batch_size, gamma, gae_lambda)

    # Print and Save
    if (t + 1) % print_interval == 0:
        avg_reward = np.mean(replay_buffer.rewards) if replay_buffer.rewards else 0
        print(f"Timestep: {t + 1}, Average Reward: {avg_reward:.2f}")

    if (t+1) % save_interval == 0:
        torch.save(actor.state_dict(), f'actor_{t+1}.pth')
        torch.save(critic.state_dict(), f'critic_{t+1}.pth')

env.close()