import torch
import config.config as cfg
import gymnasium as gym
import numpy as np
import time
from src.baseline import baseline_policy

import src.environment
import config.config_small_test as cfg

env = gym.make('SoccerEnv-v0', cfg=cfg)
obs, _ = env.reset()
env.render()

# t1 = time.time()
for i in range(100):
    for j in range(1):
        # obs,b,done,c,x = env.step(np.full((cfg.num_offense_players,), 0))
        obs,b,done,c,x = env.step(baseline_policy(obs, env))
        # print(b)
        if done:
            env.render()
            assert False
    env.render()
# t2 = time.time()
# print(t2-t1)