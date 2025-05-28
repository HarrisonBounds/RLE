import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
<<<<<<< HEAD
from torch.distributions import Normal
=======
>>>>>>> origin/feat/ppo
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim) 
        self.log_std = nn.Parameter(torch.zeros(1, action_dim)) 
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_mean = torch.tanh(self.fc3(x)) 
        action_std = torch.exp(self.log_std)
        return action_mean, action_std #Output probability distribution

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x) #Output scalar of expected rewards
        return value

#Store data to update policy   
class ReplayBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = [] #store log probs of actions

    def store(self, state, action, reward, next_state, done, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)

    def get_batch(self):
        batch = {
            'states': torch.tensor(np.array(self.states), dtype=torch.float32),
            'actions': torch.tensor(np.array(self.actions), dtype=torch.float32),
            'rewards': torch.tensor(np.array(self.rewards), dtype=torch.float32),
            'next_states': torch.tensor(np.array(self.next_states), dtype=torch.float32),
            'dones': torch.tensor(np.array(self.dones), dtype=torch.float32),
            'log_probs': torch.tensor(np.array(self.log_probs), dtype=torch.float32), #added log probs
        }
        self.clear()
        return batch

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []




class PPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor=3e-4, lr_critic=3e-4, gamma=0.99,
                 epsilon_clip=0.2, K_epochs=10, entropy_coef=0.01, gae_lambda=0.95):
        
        self.gamma = gamma
        self.epsilon_clip = epsilon_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        self.gae_lambda = gae_lambda

        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = ReplayBuffer()

    def select_action(self, observation):
        
        state_data = observation['state'].flatten()
        lidar_data = observation['lidar'].flatten()
        processed_state = np.concatenate([state_data, lidar_data])

        state_tensor = torch.tensor(processed_state, dtype=torch.float32).unsqueeze(0)

        #Forward Pass
        with torch.no_grad(): 
            action_mean, action_std = self.actor(state_tensor)
        
        #Create normal distribution and sample from it
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)

        return action.squeeze(0).numpy(), log_prob.item()

    def compute_advantages_and_returns(self, rewards, values, next_values dones):
        #Squeeze out the batch dimension
        values = values.squeeze()
        next_values = next_values.squeeze()

        #TD Errors
        deltas = rewards + self.gamma * next_values * (1 - dones) - values

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(values)

        last_advantage = 0
        last_return = 0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                last_advantage = 0
                last_return = 0

            advantages[t] = deltas[t] + self.gamma * self.gae_lambda * last_advantage
            last_advantage = advantages[t]

            returns[t] = rewards[t] + self.gamma * last_return
            last_return = returns[t]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages.unsqueeze(1), returns.unsqueeze(1)

    def update(self):
       
        # This method will involve:
        # 1. Getting a batch of data from the buffer.
        # 2. Computing advantages and returns.
        # 3. Iterating K_epochs times to update actor and critic.
        # 4. Calculating the PPO loss for the actor (clipped objective + entropy).
        # 5. Calculating the loss for the critic (MSE loss).
        # 6. Performing backpropagation and optimization steps.
        pass




