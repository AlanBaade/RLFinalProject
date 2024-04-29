import torch
import config.config as cfg
import gymnasium as gym
import numpy as np
import time


import src.environment
env = gym.make('SoccerEnv-v0')
env.reset()
env.render()

# t1 = time.time()
for i in range(100):
    for j in range(3):
        a,b,done,c,x = env.step(np.full((cfg.num_offense_players,), 8))
        # print(b)
        if done:
            assert False
    env.render()
# t2 = time.time()
# print(t2-t1)