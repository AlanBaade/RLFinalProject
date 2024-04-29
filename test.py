import torch
import config.config as cfg
import gymnasium as gym
import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


import src.environment



model = PPO.load("models/soccer-joint-marl")
env = gym.make('SoccerEnv-v0')
obs, _ = env.reset()
done = False


while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    env.render()




