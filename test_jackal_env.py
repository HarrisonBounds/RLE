import gymnasium as gym
from jackal_env import Jackal_Env
import time
import mujoco

env = Jackal_Env(
    xml_file="jackal_velodyne.xml",
    render_mode="human",
    use_lidar=True
)

TOTAL_TIME = 10  # Duration to drive forward
STEP_TIME = 0.01  # Time per step
ITERS_FROM_TIME = int(TOTAL_TIME / STEP_TIME)


def time_to_iters(seconds: float) -> int:
    return int(seconds / STEP_TIME)


# Initial wait period (to start recording or observing)
print("[TEST] Waiting 5 seconds before motion...")
for i in range(time_to_iters(5)):
    observation = env.reset()
    action = [0, 0]  # All wheels stopped
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(0.01)

# Reset before starting actual motion
# observation = env.reset()

# Drive forward
for i in range(ITERS_FROM_TIME):
    # [left_speed, right_speed]
    action = [0.5, 0.5]  # All wheels forward at half speed
    observation, reward, terminated, truncated, info = env.step(action)
    print("Observation keys or shape:", type(observation), observation.keys())
    print("LiDAR data shape:", observation['lidar'].shape)
    print("LiDAR data sample:", observation['lidar'][:10])  # first 10 values from LiDAR
    env.render()
    time.sleep(0.01)


# # Turn left for a few seconds
# for _ in range(ITERS_FROM_TIME // 2):
#     # Left wheels backward, right wheels forward
#     action = [-0.3, 0.3, -0.3, 0.3]
#     observation, reward, terminated, truncated = env.step(action)
#     env.render()
#     time.sleep(0.01)
