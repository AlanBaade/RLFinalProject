import torch
import gymnasium as gym
import numpy as np
import time
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
import time

import src.environment

import config.config_small_train as cfg
vec_env = make_vec_env("SoccerEnv-v0", env_kwargs={"cfg": cfg}, n_envs=8)





from individual_marl.custom_nn import CustomActorCriticPolicy
from individual_marl.custom_extractor import CustomExtractor
policy_kwargs = dict(
    features_extractor_class=CustomExtractor
)

model = A2C(CustomActorCriticPolicy, vec_env, policy_kwargs=policy_kwargs, verbose=1,
  n_steps=32,
  learning_rate=0.003)

out_path = "experiment_out/a2c/soccer-individual-marl"
logger = configure(out_path, ["stdout", "csv"])
model.set_logger(logger)

start = time.time()

while True:
  model.learn(total_timesteps=100000)
  model.save("models/a2c/soccer-individual-marl")
  print("TIME")
  print(time.time() - start)


