import torch
import config.config as cfg
import gymnasium as gym
import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure


import src.environment
import nn



vec_env = make_vec_env("SoccerEnv-v0", n_envs=4)

model = PPO("MlpPolicy", vec_env, verbose=1)

out_path = "experiment_out/soccer-joint-marl"
logger = configure(out_path, ["stdout", "csv"])
model.set_logger(logger)

while True:
  model.learn(total_timesteps=100000)
  model.save("models/soccer-joint-marl")



