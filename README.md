# Bipedal Walker PPO Training

This project trains an agent using **Proximal Policy Optimization (PPO)** in the **Bipedal Walker** environment, simulating a robot with 4 joints and 2 legs, aiming to walk across rough terrain. The environment is used in both normal and hardcore modes.

## Table of Contents
0. [About Bipedal Walker](#about-bipedal-walker)
1. [Training Process](#training-process)
    - Normal and Hardcore versions using PPO.
    - Analysis of training logs.
2. [Environment Setup: make_env()](#environment-setup-make_env)
3. [Results and Evaluation](#results-and-evaluation)
4. [Improvements](#improvements)
5. [Credits](#credits)

## 0. About Bipedal Walker
The **Bipedal Walker** environment simulates a bipedal robot, focusing on balance and coordination across uneven terrain.

- **Observation Space**: 24 continuous values including angles, velocities, and LIDAR readings.
- **Action Space**: 4 continuous values controlling hip and knee joints.
- **Rewards**: Positive for forward movement and negative for falls or excessive torque.
- **Termination**: Occurs if the robot falls or exceeds the step limit.

## 1. Training Process
Training was performed with **PPO** for both the normal and hardcore versions of Bipedal Walker.

- **Normal Mode**: Trained for 1 million timesteps with vectorized environments and reward normalization.
- **Hardcore Mode**: Trained for 5 million timesteps, focusing on navigating more difficult terrain.

## 2. Environment Setup: make_env()

The `make_env()` function is designed to prepare the **Bipedal Walker** environment for efficient training, offering several configurable options:
- **Hardcore Mode**: Allows switching between normal and hardcore versions of the environment.
- **Vectorized Operations**: Uses `DummyVecEnv` to enable parallel training and faster performance.
- **Observation & Reward Normalization**: Uses `VecNormalize` to stabilize learning by normalizing both observations and rewards.
- **Frame Stacking**: The last `n` frames are stacked using `VecFrameStack`, providing the agent with temporal context.
- **Video Recording**: If enabled, videos are recorded every 1000 steps and saved in the specified folder.
- **Monitor**: Logs important training metrics, such as rewards and episode lengths, to a designated directory.

This setup ensures the environment is optimized for training PPO models with advanced features like video recording and reward normalization.

## 3. Results and Evaluation
Evaluation was conducted using multiple episodes to determine the average rewards and episode lengths.

- **Normal Mode**: The average reward indicates a well-performing policy with occasional variability.
- **Hardcore Mode**: Training struggles initially but shows improvement over time.

## 4. Improvements
Suggestions for improving agent performance:
- Adjust learning rate and reward structure.
- Increase exploration and training duration.

## 5. Credits
Developed by Oleg Klimov. Adapted for PPO training.
