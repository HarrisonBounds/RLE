import gymnasium as gym
from jackal_env import Jackal_Env
import time
import mujoco

env = Jackal_Env(
    xml_file="jackal_velodyne.xml",
    render_mode="human"
)

TOTAL_TIME = 60  # seconds
STEP_TIME = 0.01  # seconds
ITERS_FROM_TIME = int(TOTAL_TIME / STEP_TIME)

for _ in range(ITERS_FROM_TIME):
    try:
        env.render()
        time.sleep(STEP_TIME)
    except Exception as e:
        print(f"Error: {e}")
        break
env.close()


# observation = env.reset()
# for _ in range(1000):
#     action = env.action_space.sample()  # Take a random action
#     observation, reward, terminated, truncated = env.step(action)

#     env.render()

#     # print(f"Observation: {observation.shape}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

#     time.sleep(0.01)
#     if terminated or truncated:
#         observation = env.reset()

# env.close()
