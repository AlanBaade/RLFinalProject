import torch
import gymnasium as gym
import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import time

import src.environment
import src.environment_agent_rew

model = PPO.load("models/agrew/soccer-joint-marl")
import config.config_small_train as cfg

env = gym.make('SoccerEnvAgentRew-v0', cfg=cfg)
obs, _ = env.reset()
done = False

idx = 0
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    if idx % 2 == 0:
        env.render()
        time.sleep(0.5)
    idx += 1
env.render()
print(idx)

