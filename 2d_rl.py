import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# --- 1. Custom 2D Robot Environment ---

class CustomRobotEnv(gym.Env):
    """
    A 2D environment for a robot to navigate from a start to a goal state.
    Features:
    - Robot and goal spawn in random locations each episode.
    - Simulated LIDAR for obstacle detection (walls only for simplicity).
    - Continuous action space (linear and angular velocity).
    - Continuous observation space (LIDAR readings, relative goal position).
    - Reward based on distance to goal.
    """
    metadata = {'render_modes': ['human', 'none'], 'render_fps': 30}

    def __init__(self, render_mode='none', workspace_size=10.0, lidar_beams=16, lidar_range=5.0):
        super().__init__()

        self.render_mode = render_mode
        self.workspace_size = workspace_size  # Size of the square workspace (e.g., 10x10 units)
        self.lidar_beams = lidar_beams        # Number of LIDAR beams
        self.lidar_range = lidar_range        # Max range of LIDAR
        self.robot_radius = 0.2               # Robot collision radius
        self.goal_radius = 0.3                # Goal reach radius
        self.dt = 0.1                         # Time step for physics simulation

        # State: [robot_x, robot_y, robot_theta, goal_x, goal_y]
        # Observations: [lidar_readings (lidar_beams), goal_relative_x, goal_relative_y, goal_relative_angle]
        # Lidar readings are normalized to [0, 1] by dividing by lidar_range
        # Goal relative position is normalized by dividing by workspace_size
        low_obs = [0.0] * lidar_beams + [-1.0, -1.0, -math.pi]
        high_obs = [1.0] * lidar_beams + [1.0, 1.0, math.pi]
        self.observation_space = spaces.Box(low=np.array(low_obs, dtype=np.float32),
                                            high=np.array(high_obs, dtype=np.float32),
                                            dtype=np.float32)

        # Action: [linear_velocity, angular_velocity]
        self.action_space = spaces.Box(low=np.array([-1.0, -math.pi/2], dtype=np.float32),
                                       high=np.array([1.0, math.pi/2], dtype=np.float32),
                                       dtype=np.float32)

        self.robot_state = None  # [x, y, theta]
        self.goal_state = None   # [x, y]

        #Generate random obstacles
        self.num_obstacles = 0
        self.obstacle_array = []

        if self.render_mode == 'human':
            import pygame
            pygame.init()
            self.screen_size = 700
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("2D Robot Navigation")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 24)

    def _get_obs(self):
        """Generates the observation from the current robot and goal state."""
        rx, ry, rtheta = self.robot_state
        gx, gy = self.goal_state

        # 1. Simulated LIDAR readings
        lidar_readings = self._simulate_lidar(rx, ry, rtheta)

        # 2. Relative goal position
        dx = gx - rx
        dy = gy - ry
        dist_to_goal = math.sqrt(dx**2 + dy**2)
        angle_to_goal = math.atan2(dy, dx) - rtheta
        angle_to_goal = math.atan2(math.sin(angle_to_goal), math.cos(angle_to_goal)) # Normalize angle to [-pi, pi]

        # Normalize observations
        normalized_lidar = lidar_readings / self.lidar_range
        normalized_dx = dx / self.workspace_size
        normalized_dy = dy / self.workspace_size
        # angle_to_goal is already in [-pi, pi]
        # normalized_angle_to_goal = angle_to_goal / math.pi # No need to normalize if goal is relative to robot angle

        obs = np.concatenate([normalized_lidar,
                              np.array([normalized_dx, normalized_dy, angle_to_goal])]).astype(np.float32)
        return obs

    def _get_info(self):
        """Provides auxiliary information (e.g., actual distance to goal)."""
        rx, ry, _ = self.robot_state
        gx, gy = self.goal_state
        dist_to_goal = math.sqrt((gx - rx)**2 + (gy - ry)**2)
        return {"distance_to_goal": dist_to_goal}

    def _simulate_lidar(self, rx, ry, rtheta):
        """
        Simulates LIDAR beams. For simplicity, only detects walls.
        Assumes the workspace is a square from (0,0) to (workspace_size, workspace_size).
        """
        readings = np.full(self.lidar_beams, self.lidar_range, dtype=np.float32)
        angles = np.linspace(-math.pi/2, math.pi/2, self.lidar_beams) # Front 180 degrees

        for i, angle_offset in enumerate(angles):
            beam_angle = rtheta + angle_offset
            min_dist = self.lidar_range

            # Check intersection with walls
            # Wall 1: x = 0
            if math.cos(beam_angle) < 0.001: # Avoid division by zero if beam is vertical
                dist_x = float('inf')
            else:
                dist_x = (0 - rx) / math.cos(beam_angle)
            if dist_x > 0 and dist_x < min_dist:
                intersect_y = ry + dist_x * math.sin(beam_angle)
                if 0 <= intersect_y <= self.workspace_size:
                    min_dist = dist_x

            # Wall 2: x = workspace_size
            if math.cos(beam_angle) > -0.001: # Avoid division by zero
                dist_x = float('inf')
            else:
                dist_x = (self.workspace_size - rx) / math.cos(beam_angle)
            if dist_x > 0 and dist_x < min_dist:
                intersect_y = ry + dist_x * math.sin(beam_angle)
                if 0 <= intersect_y <= self.workspace_size:
                    min_dist = dist_x

            # Wall 3: y = 0
            if math.sin(beam_angle) < 0.001: # Avoid division by zero if beam is horizontal
                dist_y = float('inf')
            else:
                dist_y = (0 - ry) / math.sin(beam_angle)
            if dist_y > 0 and dist_y < min_dist:
                intersect_x = rx + dist_y * math.cos(beam_angle)
                if 0 <= intersect_x <= self.workspace_size:
                    min_dist = dist_y

            # Wall 4: y = workspace_size
            if math.sin(beam_angle) > -0.001: # Avoid division by zero
                dist_y = float('inf')
            else:
                dist_y = (self.workspace_size - ry) / math.sin(beam_angle)
            if dist_y > 0 and dist_y < min_dist:
                intersect_x = rx + dist_y * math.cos(beam_angle)
                if 0 <= intersect_x <= self.workspace_size:
                    min_dist = dist_y

            for cx, cy, cr in self.obstacle_array:
                # Vector from robot to circle center
                dx = cx - rx
                dy = cy - ry

                # Project the vector onto the beam direction
                a = dx * math.cos(beam_angle) + dy * math.sin(beam_angle)

                # Squared distance from robot to circle center
                b_sq = dx**2 + dy**2

                 # Discriminant of the quadratic equation
                discriminant = a**2 - b_sq + cr**2

                # Check for intersection
                if discriminant >= 0:
                    # Calculate the two possible distances
                    dist1 = a - math.sqrt(discriminant)
                    dist2 = a + math.sqrt(discriminant)

                    # Check for the valid, positive intersection
                    if dist1 > 0:
                        min_dist = min(min_dist, dist1)
                    if dist2 > 0:
                        min_dist = min(min_dist, dist2)

            readings[i] = min(min_dist, self.lidar_range)
        return readings

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomly spawn robot within the workspace
        # Ensure robot does not start too close to an edge, or too close to the goal if possible
        min_spawn_dist = self.robot_radius * 2
        max_coord = self.workspace_size - min_spawn_dist
        self.robot_state = np.array([
            random.uniform(min_spawn_dist, max_coord),
            random.uniform(min_spawn_dist, max_coord),
            random.uniform(-math.pi, math.pi) # Random initial orientation
        ], dtype=np.float32)

        # Randomly spawn goal, ensuring it's not too close to the robot
        while True:
            self.goal_state = np.array([
                random.uniform(min_spawn_dist, max_coord),
                random.uniform(min_spawn_dist, max_coord)
            ], dtype=np.float32)
            dist_to_start = math.sqrt((self.robot_state[0] - self.goal_state[0])**2 +
                                      (self.robot_state[1] - self.goal_state[1])**2)
            if dist_to_start > self.goal_radius * 5: # Ensure a reasonable distance
                break

        self.num_obstacles = random.randint(2, 5)
        self.obstacle_array = []

        for _ in range(self.num_obstacles):
            self.obstacle_array.append((random.uniform(0.0, self.workspace_size), random.uniform(0.0, self.workspace_size), random.uniform(0.2, 0.7)))

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self.render()

        return observation, info

    def step(self, action):
        linear_vel, angular_vel = action # Action is [linear_velocity, angular_velocity]

        # Clamp velocities to reasonable limits
        linear_vel = np.clip(linear_vel, -1.0, 1.0)
        angular_vel = np.clip(angular_vel, -math.pi/2, math.pi/2)

        # Update robot state (simple differential drive model)
        rx, ry, rtheta = self.robot_state

        # Update position
        new_rx = rx + linear_vel * math.cos(rtheta) * self.dt
        new_ry = ry + linear_vel * math.sin(rtheta) * self.dt
        new_rtheta = rtheta + angular_vel * self.dt
        new_rtheta = math.atan2(math.sin(new_rtheta), math.cos(new_rtheta)) # Normalize angle

        # Collision detection with walls (simple boundary check)
        collided = False
        if not (self.robot_radius <= new_rx <= self.workspace_size - self.robot_radius and
                self.robot_radius <= new_ry <= self.workspace_size - self.robot_radius):
            collided = True

        for ox, oy, or_ in self.obstacle_array:
            dist_to_obstacle = math.sqrt((new_rx - ox)**2 + (new_ry - oy)**2)
            if dist_to_obstacle < self.robot_radius + or_:
                collided = True
                break
            
            # Penalize collision by setting velocities to zero or reversing slightly
            new_rx = np.clip(new_rx, self.robot_radius, self.workspace_size - self.robot_radius)
            new_ry = np.clip(new_ry, self.robot_radius, self.workspace_size - self.robot_radius)
            linear_vel = 0 # Stop movement if collided
            # No change to angular velocity for now, could also stop rotation

        self.robot_state = np.array([new_rx, new_ry, new_rtheta], dtype=np.float32)

        # Calculate reward
        info = self._get_info()
        distance_to_goal = info["distance_to_goal"]

        reward = -distance_to_goal  # Negative reward for distance, encourages getting closer
        if distance_to_goal < self.goal_radius:
            reward += 100.0 # Large positive reward for reaching goal
            terminated = True
        else:
            terminated = False

        # Termination condition (e.g., time limit)
        truncated = False # No time limit in this basic example, can be added

        # Add collision penalty if needed
        if collided:
            reward -= 50.0 # Penalize collisions

        observation = self._get_obs()

        if self.render_mode == 'human':
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'human':
            import pygame
            self.screen.fill((255, 255, 255)) # White background

            scale = self.screen_size / self.workspace_size

            # Draw workspace boundary
            pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, self.screen_size, self.screen_size), 2)

            # Draw robot
            robot_color = (0, 0, 255) # Blue
            robot_pos_scaled = (int(self.robot_state[0] * scale), int(self.robot_state[1] * scale))
            pygame.draw.circle(self.screen, robot_color, robot_pos_scaled, int(self.robot_radius * scale))

            # Draw robot orientation (a line from center)
            orientation_len = self.robot_radius * scale * 1.5
            end_x = robot_pos_scaled[0] + orientation_len * math.cos(self.robot_state[2])
            end_y = robot_pos_scaled[1] + orientation_len * math.sin(self.robot_state[2])
            pygame.draw.line(self.screen, (255, 0, 0), robot_pos_scaled, (int(end_x), int(end_y)), 2)

            # Draw goal
            goal_color = (0, 255, 0) # Green
            goal_pos_scaled = (int(self.goal_state[0] * scale), int(self.goal_state[1] * scale))
            pygame.draw.circle(self.screen, goal_color, goal_pos_scaled, int(self.goal_radius * scale), 2) # Outline

            # Draw LIDAR beams (for visualization, if desired)
            lidar_color = (255, 165, 0) # Orange
            readings = self._simulate_lidar(self.robot_state[0], self.robot_state[1], self.robot_state[2])
            angles = np.linspace(-math.pi/2, math.pi/2, self.lidar_beams) # Front 180 degrees
            for i, angle_offset in enumerate(angles):
                beam_angle = self.robot_state[2] + angle_offset
                beam_len = readings[i] * scale
                end_x = robot_pos_scaled[0] + beam_len * math.cos(beam_angle)
                end_y = robot_pos_scaled[1] + beam_len * math.sin(beam_angle)
                pygame.draw.line(self.screen, lidar_color, robot_pos_scaled, (int(end_x), int(end_y)), 1)

            # Draw Obstacles
            obstacle_color = (0, 0, 0)
            for ox, oy, r in self.obstacle_array:
                pygame.draw.circle(
                    self.screen, 
                    obstacle_color, 
                    (int(ox * scale), int(oy * scale)), 
                    int(r * scale), 2
                )

            # Display info
            info_text = self.font.render(f"Dist: {self._get_info()['distance_to_goal']:.2f}", True, (0, 0, 0))
            self.screen.blit(info_text, (10, 10))

            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])

    def close(self):
        if self.render_mode == 'human':
            import pygame
            pygame.quit()


# --- 2. Soft Actor-Critic (SAC) Implementation ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, terminated, truncated):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, terminated, truncated)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, terminated, truncated = map(np.stack, zip(*batch))
        return state, action, reward, next_state, terminated, truncated

    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample() # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing action bounds, correct log_prob computation for tanh squashing
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6).sum(1, keepdim=True)
        return action, log_prob, mean, log_std # Also return mean and log_std for debugging/metrics

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

class SAC:
    def __init__(self, env, hidden_dim=256, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, target_update_interval=1):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_scale = torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32, device=device)
        self.action_bias = torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32, device=device)

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.target_update_interval = target_update_interval

        # Actor Network
        self.actor = Actor(self.state_dim, self.action_dim, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Critic Networks (Q-functions)
        self.critic1 = Critic(self.state_dim, self.action_dim, hidden_dim).to(device)
        self.critic2 = Critic(self.state_dim, self.action_dim, hidden_dim).to(device)
        self.critic1_target = Critic(self.state_dim, self.action_dim, hidden_dim).to(device)
        self.critic2_target = Critic(self.state_dim, self.action_dim, hidden_dim).to(device)

        # Initialize target networks with same weights as main networks
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(param.data)

        self.critic_optimizer = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr)

        # Automatic entropy tuning (alpha)
        self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        self.updates = 0

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        if evaluate is False:
            action, _, _, _ = self.actor.sample(state)
        else:
            _, _, action_mean, _ = self.actor.sample(state) # For deterministic action during evaluation
            action = torch.tanh(action_mean) # Use tanh to squash the mean

        # Scale action to environment's action space
        scaled_action = action * self.action_scale + self.action_bias
        return scaled_action.detach().cpu().numpy().flatten()

    def update_parameters(self, replay_buffer, batch_size):
        # Sample a batch from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, terminated_batch, truncated_batch = replay_buffer.sample(batch_size)

        state_batch = torch.FloatTensor(state_batch).to(device)
        action_batch = torch.FloatTensor(action_batch).to(device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(device)
        terminated_batch = torch.FloatTensor(terminated_batch).unsqueeze(1).to(device)
        truncated_batch = torch.FloatTensor(truncated_batch).unsqueeze(1).to(device)

        # 1. Update Critic (Q-functions)
        with torch.no_grad():
            next_action, log_prob_next_action, _, _ = self.actor.sample(next_state_batch)
            q1_target_next = self.critic1_target(next_state_batch, next_action)
            q2_target_next = self.critic2_target(next_state_batch, next_action)
            min_q_target_next = torch.min(q1_target_next, q2_target_next) - self.alpha * log_prob_next_action
            # Handle terminated states: if terminated, future Q-value is 0
            # Truncated means the episode ended for reasons other than reaching goal/failing,
            # so we still consider future value for those.
            next_q_value = reward_batch + (1 - terminated_batch) * self.gamma * min_q_target_next

        q1_current = self.critic1(state_batch, action_batch)
        q2_current = self.critic2(state_batch, action_batch)

        critic1_loss = nn.functional.mse_loss(q1_current, next_q_value)
        critic2_loss = nn.functional.mse_loss(q2_current, next_q_value)
        critic_loss = critic1_loss + critic2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 2. Update Actor
        action_reparam, log_prob_reparam, _, _ = self.actor.sample(state_batch)
        q1_actor = self.critic1(state_batch, action_reparam)
        q2_actor = self.critic2(state_batch, action_reparam)
        min_q_actor = torch.min(q1_actor, q2_actor)

        actor_loss = ((self.alpha * log_prob_reparam) - min_q_actor).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 3. Update temperature parameter (alpha)
        alpha_loss = -(self.log_alpha * (log_prob_reparam + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        # 4. Soft update target networks
        self.updates += 1
        if self.updates % self.target_update_interval == 0:
            for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return critic_loss.item(), actor_loss.item(), alpha_loss.item(), self.alpha


# --- 3. Training Loop ---

def train_sac(env, agent, replay_buffer, num_episodes=5000, max_steps_per_episode=200,
              batch_size=256, updates_per_step=1):
    total_rewards = []
    avg_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False
        step_count = 0

        while not terminated and not truncated and step_count < max_steps_per_episode:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, terminated, truncated)

            state = next_state
            episode_reward += reward
            step_count += 1

            # Update agent if buffer is full enough
            if len(replay_buffer) > batch_size:
                for _ in range(updates_per_step):
                    critic_loss, actor_loss, alpha_loss, alpha = agent.update_parameters(replay_buffer, batch_size)

        total_rewards.append(episode_reward)
        avg_rewards.append(np.mean(total_rewards[-100:])) # Average over last 100 episodes

        if (episode + 1) % 100 == 0:
            print(f"Episode: {episode + 1}, Reward: {episode_reward:.2f}, Avg Reward (last 100): {avg_rewards[-1]:.2f}")
            print(f"  Critic Loss: {critic_loss:.4f}, Actor Loss: {actor_loss:.4f}, Alpha: {alpha:.4f}")

    print("Training finished.")
    return total_rewards, avg_rewards

# --- Main Execution ---
if __name__ == "__main__":
    # Environment setup
    env = CustomRobotEnv(render_mode='human', workspace_size=10.0, lidar_beams=16, lidar_range=5.0)

    # SAC Agent setup
    buffer_capacity = 1_000_000
    replay_buffer = ReplayBuffer(buffer_capacity)
    sac_agent = SAC(env, hidden_dim=256, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2) # Alpha=0.2 is a starting point, it will be tuned automatically

    # Training parameters
    num_episodes_to_train = 5000
    max_steps_per_episode = 200 # A reasonable limit for navigation tasks
    training_batch_size = 256
    updates_per_env_step = 1 # Number of gradient updates per environment step

    # Start training
    print("Starting SAC training...")
    rewards, avg_rewards = train_sac(env, sac_agent, replay_buffer,
                                     num_episodes=num_episodes_to_train,
                                     max_steps_per_episode=max_steps_per_episode,
                                     batch_size=training_batch_size,
                                     updates_per_step=updates_per_env_step)

    # Optional: Save the trained model
    # torch.save(sac_agent.actor.state_dict(), 'sac_actor.pth')
    # torch.save(sac_agent.critic1.state_dict(), 'sac_critic1.pth')
    # print("Model saved.")

    env.close()

    # Basic plot of rewards
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label='Episode Reward')
    plt.plot(avg_rewards, label='Average Reward (last 100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('SAC Training Rewards')
    plt.legend()
    plt.grid(True)
    plt.show()
