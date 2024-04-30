import torch
import config.config as cfg
import gymnasium as gym
import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import src.environment

vec_env = make_vec_env(
    "SoccerEnv",
    n_envs=16,
)

from transformer_nn import TransformerActorCriticPolicy
from custom_extractor import CustomExtractor

model = PPO(
    TransformerActorCriticPolicy,
    vec_env,
    verbose=1,
    n_steps=128,
)

while True:
    model.learn(total_timesteps=100000)
    model.save("models/soccer-joint-marl")
