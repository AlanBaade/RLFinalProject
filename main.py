import torch
import config.config as cfg
import gymnasium as gym
import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


import src.environment



# vec_env = make_vec_env('SoccerEnv-v0', n_envs=4)

# model = PPO("MlpPolicy", vec_env, verbose=1)
# model.learn(total_timesteps=50000)
# model.save("soccer-env")


model = PPO.load("soccer-env")
env = gym.make('SoccerEnv-v0')
obs, _ = env.reset()
done = False


while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    env.render()




