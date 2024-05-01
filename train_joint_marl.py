import torch
import gymnasium as gym
import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure


import src.environment
import time



import config.config_small_train as cfg
vec_env = make_vec_env("SoccerEnv-v0", env_kwargs={"cfg": cfg}, n_envs=8)

model = PPO("MlpPolicy", vec_env, verbose=1)

out_path = "experiment_out/soccer-joint-marl"
logger = configure(out_path, ["stdout", "csv"])
model.set_logger(logger)

start = time.time()
while True:
  model.learn(total_timesteps=100000)
  model.save("models/soccer-joint-marl")
  print("TIME")
  print(time.time() - start)



