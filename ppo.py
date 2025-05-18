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


def ppo_update(actor, critic, optimizer_actor, optimizer_critic, batch, clip_param, ppo_epochs, mini_batch_size, gamma, gae_lambda):
    states = batch['states']
    actions = batch['actions']
    rewards = batch['rewards']
    next_states = batch['next_states']
    dones = batch['dones']
    old_log_probs = batch['log_probs'] #get old log probs

    # Calculate Advantages and Returns (GAE)
    values = critic(states).squeeze()
    next_values = critic(next_states).squeeze()
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(rewards.size(0))):
        if t == rewards.size(0) - 1:
            nextnonterminal = 1.0 - dones[t]
            next_value = next_values[t]
        else:
            nextnonterminal = 1.0 - dones[t]
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        returns[t] = rewards[t] + gamma * next_value * nextnonterminal


    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    #convert to tensors
    returns = returns.detach()

    for _ in range(ppo_epochs): #ppo epochs
        # Mini-batch iteration
        for index in range(0, states.size(0), mini_batch_size):
            #get mini batch
            state_batch = states[index: index + mini_batch_size]
            action_batch = actions[index: index + mini_batch_size]
            adv_batch = advantages[index: index + mini_batch_size]
            return_batch = returns[index: index + mini_batch_size]
            old_log_prob_batch = old_log_probs[index: index + mini_batch_size]

            #get new action log probs
            action_mean, action_std = actor(state_batch)
            dist = torch.distributions.Normal(action_mean, action_std)
            new_log_probs = dist.log_prob(action_batch).sum(dim=-1)


            # Calculate probability ratio
            ratio = torch.exp(new_log_probs - old_log_prob_batch)

            # Clipped objective
            surr1 = ratio * adv_batch
            surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * adv_batch
            actor_loss = -torch.min(surr1, surr2).mean()


            # Value function loss
            value_pred = critic(state_batch).squeeze()
            critic_loss = F.mse_loss(value_pred, return_batch).mean()

            # Optimize actor
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            # Optimize critic
            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()



