import gymnasium as gym
from jackal_env import Jackal_Env
import time
import mujoco

env = Jackal_Env(
    xml_file="jackal_velodyne.xml",
    render_mode="human"
)

TOTAL_TIME = 10  # seconds
STEP_TIME = 0.01  # seconds
ITERS_FROM_TIME = int(TOTAL_TIME / STEP_TIME)

# Drive forward for a few seconds
observation = env.reset()
for _ in range(ITERS_FROM_TIME):
    # [left_front, right_front, left_rear, right_rear]
    action = [0.5, 0.5, 0.5, 0.5]  # All wheels forward at half speed
    observation, reward, terminated, truncated = env.step(action)
    env.render()
    time.sleep(0.01)

# # Turn left for a few seconds
# for _ in range(ITERS_FROM_TIME // 2):
#     # Left wheels backward, right wheels forward
#     action = [-0.3, 0.3, -0.3, 0.3]
#     observation, reward, terminated, truncated = env.step(action)
#     env.render()
#     time.sleep(0.01)
