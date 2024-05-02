import torch
import gymnasium as gym
import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import time
import src.environment
import src.environment_agent_rew
from src.baseline import baseline_policy

import config.config_small_train as cfg_small_train
import config.config_small_test as cfg_small_test
import config.config_large_train as cfg_large_train
import config.config_large_test as cfg_large_test

import warnings
warnings.filterwarnings("ignore")

N = 100

def run_one(p, env):
  obs, _ = env.reset()
  done = False
  rew_tot = 0
  while not done:
      action = p(obs, env)
      obs, reward, done, _, _ = env.step(action)
      rew_tot += reward
  return rew_tot

def run_n(p, env):
  global N
  rews = []
  for i in range(N):
    rews.append(run_one(p, env))
  mean = np.mean(rews)
  stdv = np.std(rews) / np.sqrt(len(rews))
  return (mean, stdv)


categories = []
stdvs = []
means = []



env = gym.make('SoccerEnv-v0', cfg=cfg_large_test)
x = run_n(baseline_policy, env)
means.append(x[0])
stdvs.append(x[1])

print(means)
print(stdvs)


