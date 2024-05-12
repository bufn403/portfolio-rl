import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import gymnasium as gym
import random


class CustomEnv(gym.Env):
  def __init__(self):
    super(CustomEnv, self).__init__()
    self.action_space = gym.spaces.Box(low=-1, high=1, shape=(5,))
    self.observation_space = gym.spaces.Dict(
      data=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5, 3), dtype=np.float32),
      weights=gym.spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
    )
  
  def reset(self, *args, **kwargs):
    x = np.random.rand(5, 3)
    x[:, 0] = np.linspace(0, 1, 5)
    return {
      'data': x,
      'weights': np.array([0, 0, 0, 0, 1.0])
    }
  
  def step(self, action):
    action = 2*action - 1
    t = np.linspace(0, 1, 5)
    reward = -100 * ((action - t)**2).sum()

    x = np.random.rand(5, 3)
    x[:, 0] = np.linspace(0, 1, 5)

    finished = np.random.rand() < 0.01
    return {
      'data': x,
      'weights': action
    }, reward, finished, False, {}