import torch
import config.config as cfg
import gymnasium as gym
import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


import src.environment
import nn



vec_env = make_vec_env("SoccerEnv-v0", n_envs=4)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=50000)
model.save("models/soccer-joint-marl")


