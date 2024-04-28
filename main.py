import torch
import config.config as cfg
import gymnasium as gym
import numpy as np


import src.environment
env = gym.make('SoccerEnv-v0')
env.reset()
env.render()
for i in range(10):
    for j in range(3):
        a,b,done,x = env.step(np.full((cfg.num_offense_players,), 0))
        print(b)
        if done:
            assert False
    env.render()