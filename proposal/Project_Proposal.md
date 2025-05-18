# Reinforcement Learning for Exploration (RLE)

**Team Members:**
- Harrison Bounds
- Sharwin Patil
- Andrew Kwolek
- Logan Boswell

---

### Goal

- Train an RL pipeline to have a mobile robot navigate an area and dynamically avoid obstacles
- Build a 2d occuapncy map from the exploration using ROS2 and Lidar data
- Perform sim-to-real on the robot to have it explore an area in real life

---

### Method

#### Machine Learning Algorithm
- **Proximal Policy Optimization (PPO)**: An Actor-Critic Neural Network Architecture where the Actor produces an action and a critic maxmizes the expected discounted rewards. 

#### Sensing Component
- **3D Lidar**: The lidar is used to detect objects around it by shooting lasers in every direction and measuring the distance . 

---

### Demo and Evaluation Plan

**Demo**: 

- Ideally, we would like to perform sim-to-real to deploy our real robot in a small controlled environment using ROS 2 to navigate and avoid obstacles.

- Due to the time constraints we have, we will focus on preparing the simulation for demo and displaying the robot's capability to explore an area successfully

- Display a map of the explored area in simulation/real world to prove exploration capabilities

**Evaluation**

- We will evaluate our project on the maximum reward per episode. This encourages the robot to gain rewards by performing its specified task which is to explore. This is going to require very specific reward functions. 

- The produced map will also be a sort of evaluation metric of how acccurate the the robot explores a given area
