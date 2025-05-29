import gymnasium as gym
from jackal_env import Jackal_Env
import time
import mujoco

env = Jackal_Env(
    xml_file="jackal_obstacles_randomized.xml",
    render_mode="human",
    use_lidar=True
)

count = 0

# Drive forward
while True:
    action = [0.5, 0.5]  # All wheels forward at half speed
    observation, reward, terminated, truncated, info = env.step(action)

    if count == 500:
        break

    count += 1

    # print("Observation keys or shape:", type(observation), observation.keys())
    # print("LiDAR data shape:", observation['lidar'].shape)
    # print("LiDAR data sample:", observation['lidar'][:10])  # first 10 values from LiDAR
    env.render()
