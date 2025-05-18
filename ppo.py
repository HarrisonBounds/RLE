import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

