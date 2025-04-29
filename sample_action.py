import gymnasium as gym
from jackal_env import Jackal_Env
import time
import mujoco

env = Jackal_Env(render_mode="human")

observation = env.reset()
for _ in range(1000):
    action = env.action_space.sample() # Take a random action
    observation, reward, terminated, truncated = env.step(action)

    env.render()

    # print(f"Observation: {observation.shape}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

    time.sleep(0.01)
    if terminated or truncated:
        observation = env.reset()

env.close()