import torch
import gymnasium as gym
import numpy as np
import time
from src.baseline import baseline_policy

import src.environment
import config.config_large_train as cfg

env = gym.make('SoccerEnv-v0', cfg=cfg)
obs, _ = env.reset()
env.render()

done = False
i = 0
while not done:
    obs,b,done,c,x = env.step(baseline_policy(obs, env))
    if i%4 == 0:
        env.render()
        time.sleep(1.0)
    i+=1
env.render()
