import torch
import gymnasium as gym


import src.environment
env = gym.make('SoccerEnv-v0')
env.reset()
env.render()