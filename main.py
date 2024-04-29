import torch
import config.config as cfg
import gymnasium as gym
import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


import src.environment




vec_env = make_vec_env('SoccerEnv-v0', n_envs=4)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=250000)
model.save("soccer-env")

del model # remove to demonstrate saving and loading

model = PPO.load("soccer-env")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
    print(rewards)
    print(dones)

