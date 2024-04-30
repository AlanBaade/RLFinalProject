import torch
import config.config as cfg
import gymnasium as gym
import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
import time

import src.environment

vec_env = make_vec_env("SoccerEnv", n_envs=8)





from custom_nn import CustomActorCriticPolicy
from custom_extractor import CustomExtractor
policy_kwargs = dict(
    features_extractor_class=CustomExtractor
)

model = PPO(CustomActorCriticPolicy, vec_env, policy_kwargs=policy_kwargs, verbose=1)

out_path = "experiment_out/soccer-individual-marl"
logger = configure(out_path, ["stdout", "csv"])
model.set_logger(logger)

start = time.time()

while True:
  model.learn(total_timesteps=100000)
  model.save("models/soccer-individual-marl")
  print("TIME")
  print(time.time() - start)


