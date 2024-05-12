import torch
import torch.nn as nn
import gymnasium as gym
from typing import Callable, Optional
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

class EIEE_CNN_Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, n_output_channels: int = 32):
        super(EIEE_CNN_Extractor, self).__init__(observation_space, n_output_channels)
        n_channels, self.universe_size_w_rfa, data_len = observation_space['data'].shape
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=(1, 5)),
            nn.ReLU(),
            nn.Conv2d(16, n_output_channels, kernel_size=(1, data_len-4)),
            nn.ReLU()
        ).cuda()

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.cnn(observations['data'])
        w = observations['weights'][:, None, :]
        y = torch.cat((x.squeeze(-1), w), dim=1)
        return y






