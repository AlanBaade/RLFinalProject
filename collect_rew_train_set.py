import torch
import gymnasium as gym
import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import time
import src.environment
import src.environment_agent_rew

import config.config_small_train as cfg_small_train
import config.config_small_test as cfg_small_test
import config.config_large_train as cfg_large_train
import config.config_large_test as cfg_large_test

import warnings
warnings.filterwarnings("ignore")

N = 100

def run_one(model, env):
  obs, _ = env.reset()
  done = False
  rew_tot = 0
  while not done:
      action, _ = model.predict(obs)
      obs, reward, done, _, _ = env.step(action)
      rew_tot += reward
  return rew_tot

def run_n(model, env):
  global N
  rews = []
  for i in range(N):
    rews.append(run_one(model, env))
  mean = np.mean(rews)
  stdv = np.std(rews) / np.sqrt(len(rews))
  return (mean, stdv)


categories = []
stdvs = []
means = []



categories.append("TRANSFORMER SMALL")
model = PPO.load("models/soccer-joint-transformer")
env = gym.make('SoccerEnv-v0', cfg=cfg_small_train)
x = run_n(model, env)
means.append(x[0])
stdvs.append(x[1])


categories.append("LINEAR SMALL")
model = PPO.load("models/soccer-joint-linear")
env = gym.make('SoccerEnv-v0', cfg=cfg_small_train)
x = run_n(model, env)
means.append(x[0])
stdvs.append(x[1])


categories.append("JOINT A2C")
model = PPO.load("models/a2c/soccer-joint-marl")
env = gym.make('SoccerEnv-v0', cfg=cfg_small_train)
x = run_n(model, env)
means.append(x[0])
stdvs.append(x[1])


categories.append("INDIVIDUAL A2C")
model = PPO.load("models/a2c/soccer-individual-marl")
env = gym.make('SoccerEnv-v0', cfg=cfg_small_train)
x = run_n(model, env)
means.append(x[0])
stdvs.append(x[1])


categories.append("JOINT LARGE")
model = PPO.load("models/soccer-joint-marl-large")
env = gym.make('SoccerEnv-v0', cfg=cfg_large_train)
x = run_n(model, env)
means.append(x[0])
stdvs.append(x[1])


categories.append("INDIV LARGE")
model = PPO.load("models/soccer-individual-marl-large")
env = gym.make('SoccerEnv-v0', cfg=cfg_large_train)
x = run_n(model, env)
means.append(x[0])
stdvs.append(x[1])


categories.append("LINEAR LARGE")
model = PPO.load("models/soccer-joint-linear-large-v2")
env = gym.make('SoccerEnv-v0', cfg=cfg_large_train)
x = run_n(model, env)
means.append(x[0])
stdvs.append(x[1])


categories.append("INDIVIDUAL SMALL")
model = PPO.load("models/soccer-individual-marl")
env = gym.make('SoccerEnv-v0', cfg=cfg_small_train)
x = run_n(model, env)
means.append(x[0])
stdvs.append(x[1])

categories.append("JOINT SMALL")
model = PPO.load("models/soccer-joint-marl")
env = gym.make('SoccerEnv-v0', cfg=cfg_small_train)
x = run_n(model, env)
means.append(x[0])
stdvs.append(x[1])


categories.append("INDIVIDUAL SMALL AREW")
model = PPO.load("models/agrew/soccer-individual-marl")
env = gym.make('SoccerEnv-v0', cfg=cfg_small_train)
x = run_n(model, env)
means.append(x[0])
stdvs.append(x[1])

categories.append("JOINT SMALL AREW")
model = PPO.load("models/agrew/soccer-joint-marl")
env = gym.make('SoccerEnv-v0', cfg=cfg_small_train)
x = run_n(model, env)
means.append(x[0])
stdvs.append(x[1])


print(categories)
print(means)
print(stdvs)



