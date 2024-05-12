import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import stable_baselines3
from portfolio_env_framework import *
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy.typing as npt
import gymnasium as gym

class TrainDataManager(AbstractDataManager):
    def get_obs_space(self) -> gym.spaces.Box:
        return gym.spaces.Dict({
            'data': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3, self.universe_size+1, 15), dtype=np.float32),
            'weights': gym.spaces.Box(low=0, high=1, shape=(self.universe_size+1,), dtype=np.float32)
        })

    def get_data(self) -> tuple[int, int]:
        # read SNP data
        df = pd.read_csv('crsp_snp100_2010_to_2024.csv', dtype='string')
    
        # convert datatypes
        df = df[['date', 'TICKER', 'PRC', 'VOL', 'ASKHI', 'BIDLO', 'FACPR']]
        df.date = pd.to_datetime(df.date)
        df.FACPR = df.FACPR.fillna('0.0')
        df.astype({
            'PRC': float,
            'VOL': float,
            'ASKHI': float,
            'BIDLO': float,
            'FACPR': float
        })
    
        # drop duplicates and nans
        df = df.drop_duplicates(subset=['date', 'TICKER'])
        df.dropna(inplace=True)
    
        # only include stocks that are present in all dates
        ticker_ok = df.TICKER.value_counts() == df.TICKER.value_counts().max()
        def is_max_val_count(ticker: str) -> bool:
          return ticker_ok[ticker]
        ok = df.apply(lambda row: is_max_val_count(row['TICKER']), axis=1)
        df = df[ok]
        df = df[(df.date.dt.year >= 2010) & (df.date.dt.year <= 2019)]
    
        # create stock array
        self.stock_df = df.pivot(index='date', columns='TICKER', values='PRC').astype(float)
        self.high_df = df.pivot(index='date', columns='TICKER', values='ASKHI').astype(float)
        self.low_df = df.pivot(index='date', columns='TICKER', values='BIDLO').astype(float)
        
        # adjust for stock splits
        facpr_df = df.pivot(index='date', columns='TICKER', values='FACPR').astype(float)
        self.stock_df = self.stock_df * (1+facpr_df).cumprod(axis=0)
        self.high_df = self.high_df * (1+facpr_df).cumprod(axis=0)
        self.low_df = self.low_df * (1+facpr_df).cumprod(axis=0)
        # assert np.all(self.stock_df.pct_change().iloc[1:, :] > -1), f"{(self.stock_df.pct_change().iloc[1:, :] <= -1).sum().sum()=}, {np.any(pd.isna(self.stock_df.pct_change().iloc[1:, :]))}"
        self.ret = np.log(self.stock_df.pct_change().iloc[1:, :] + 1)
    
        # get times and dickers
        self.times = df.date.unique()[1:]
        self.tickers = df.TICKER.unique()
        
        self.num_time_periods = len(self.times)-15-1
        self.universe_size = len(self.tickers)
        return self.num_time_periods, self.universe_size
    
    def get_state(self, t: int, w: npt.NDArray[np.float64], port_val: np.float64) -> npt.NDArray[np.float64]:
        # today is self.times[self.t+15]
        s = np.zeros((3, self.universe_size+1, 15))
        # s[1:, :-1] = self.ret[self.t:self.t+15, :].T
        basis = self.stock_df.loc[self.times[t], :].to_numpy().flatten()[:, None]
        s[0, :-1, :] = self.stock_df.loc[self.times[t:t+15], :].to_numpy().T / basis
        s[1, :-1, :] = self.high_df.loc[self.times[t:t+15], :].to_numpy().T / basis
        s[2, :-1, :] = self.low_df.loc[self.times[t:t+15], :].to_numpy().T / basis
        s[:, -1, :] = 1.0
        print(f"{s=}")
        return {'data': s, 'weights': w}

    def get_prices(self, t: int) -> npt.NDArray[np.float64]:
        # today is self.times[self.t+15]
        return np.append(self.stock_df.loc[self.times[t+15], :].to_numpy().flatten(), 1.0)

class DifferentialSharpeRatioReward(AbstractRewardManager):
    def __init__(self, eta: float = 1/252):
        self.eta = eta
        self.initialize_reward()

    def initialize_reward(self):
        self.A, self.B = 0.0, 0.0

    def compute_reward(self, old_port_val: float, new_port_val: float) -> float:
        R = np.log(new_port_val / old_port_val)
        dA = R - self.A
        dB = R ** 2 - self.B
        if self.B - self.A ** 2 == 0:
            D = 0
        else:
            D = (self.B * dA - 0.5 * self.A * dB) / (self.B - self.A ** 2) ** (3 / 2)
        self.A += self.eta * dA
        self.B += self.eta * dB
        return D

class ProfitReward(AbstractRewardManager):
    def __init__(self):
        pass

    def initialize_reward(self):
        pass

    def compute_reward(self, old_port_val: float, new_port_val: float) -> float:
        return new_port_val - old_port_val

class TestDataManager(AbstractDataManager):
    def get_obs_space(self) -> gym.spaces.Box:
        # return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3, self.universe_size+1, 15), dtype=np.float32)
        return gym.spaces.Dict({
            'data': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3, self.universe_size+1, 15), dtype=np.float32),
            'weights': gym.spaces.Box(low=0, high=1, shape=(self.universe_size+1,), dtype=np.float32)
        })

    def get_data(self) -> tuple[int, int]:
        # read SNP data
        df = pd.read_csv('crsp_snp100_2010_to_2024.csv', dtype='string')
    
        # convert datatypes
        df = df[['date', 'TICKER', 'PRC', 'VOL', 'ASKHI', 'BIDLO', 'FACPR']]
        df.date = pd.to_datetime(df.date)
        df.FACPR = df.FACPR.fillna('0.0')
        df.astype({
            'PRC': float,
            'VOL': float,
            'ASKHI': float,
            'BIDLO': float,
            'FACPR': float
        })
    
        # drop duplicates and nans
        df = df.drop_duplicates(subset=['date', 'TICKER'])
        df.dropna(inplace=True)
    
        # only include stocks that are present in all dates
        ticker_ok = df.TICKER.value_counts() == df.TICKER.value_counts().max()
        def is_max_val_count(ticker: str) -> bool:
          return ticker_ok[ticker]
        ok = df.apply(lambda row: is_max_val_count(row['TICKER']), axis=1)
        df = df[ok]
        df = df[(df.date.dt.year >= 2010) & (df.date.dt.year <= 2020)]
    
        # create stock array
        self.stock_df = df.pivot(index='date', columns='TICKER', values='PRC').astype(float)
        self.high_df = df.pivot(index='date', columns='TICKER', values='ASKHI').astype(float)
        self.low_df = df.pivot(index='date', columns='TICKER', values='BIDLO').astype(float)
        
        # adjust for stock splits
        facpr_df = df.pivot(index='date', columns='TICKER', values='FACPR').astype(float)
        self.stock_df = self.stock_df * (1+facpr_df).cumprod(axis=0)
        self.high_df = self.high_df * (1+facpr_df).cumprod(axis=0)
        self.low_df = self.low_df * (1+facpr_df).cumprod(axis=0)
        # assert np.all(self.stock_df.pct_change().iloc[1:, :] > -1), f"{(self.stock_df.pct_change().iloc[1:, :] <= -1).sum().sum()=}, {np.any(pd.isna(self.stock_df.pct_change().iloc[1:, :]))}"
        self.ret = np.log(self.stock_df.pct_change().iloc[1:, :] + 1)
    
        # get times and dickers
        self.times = df.date.unique()[1:]
        self.tickers = df.TICKER.unique()
        
        self.num_time_periods = len(self.times)-15-1
        self.universe_size = len(self.tickers)
    
        # read index data and compute volatilities
        idx_df = pd.read_csv('crsp_snpidx_2010_to_2024.csv', dtype={
          'DATE': 'string',
          'vwretd': float
        })
        idx_df.DATE = pd.to_datetime(idx_df.DATE)
        idx_df['vol_20'] = idx_df.vwretd.rolling(20).std()
        idx_df['vol_60'] = idx_df.vwretd.rolling(60).std()
        idx_df.set_index('DATE', inplace=True)
        self.idx_df = idx_df
        # self.vol_20 = idx_df.vol_20
        # self.vol_60 = idx_df.vol_60
    
        # # get vix data
        # vix_df = pd.read_csv('crsp_vix_2010_to_2024.csv', dtype={
        #   'Date': 'string',
        #   'vix': float
        # })
        # vix_df.Date = pd.to_datetime(vix_df.Date)
        # vix_df.set_index('Date', inplace=True)
        # self.vix_df = vix_df.vix
        
        self.num_time_periods = len(self.times)-15-1
        self.universe_size = len(self.tickers)
        return self.num_time_periods, self.universe_size
    
    def get_state(self, t: int, w: npt.NDArray[np.float64], port_val: np.float64) -> npt.NDArray[np.float64]:
        # today is self.times[self.t+15]
        s = np.zeros((3, self.universe_size+1, 15))
        # s[1:, :-1] = self.ret[self.t:self.t+15, :].T
        basis = self.stock_df.loc[self.times[t], :].to_numpy().flatten()[:, None]
        s[0, :-1, :] = self.stock_df.loc[self.times[t:t+15], :].to_numpy().T / basis
        s[1, :-1, :] = self.high_df.loc[self.times[t:t+15], :].to_numpy().T / basis
        s[2, :-1, :] = self.low_df.loc[self.times[t:t+15], :].to_numpy().T / basis
        s[:, -1, :] = 1.0
        return {'data': s[None, :, :, :], 'weights': w[None, :]}

    def get_prices(self, t: int) -> npt.NDArray[np.float64]:
        # today is self.times[self.t+15]
        return np.append(self.stock_df.loc[self.times[t+15], :].to_numpy().flatten(), 1.0)














from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# class Custom_EIEE_CNN_Extractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 85):
#         super(Custom_EIEE_CNN_Extractor, self).__init__(observation_space, features_dim)
#         n_channels, self.universe_size_w_rfa, data_len = observation_space['data'].shape
#         n_output_channels = 16
#         self.cnn = nn.Sequential(
#             nn.Conv2d(n_channels, 8, kernel_size=(1, 5)),
#             nn.ReLU(),
#             nn.Conv2d(8, n_output_channels, kernel_size=(1, data_len-4)),
#             nn.ReLU()
#         ).cuda()
#         # self.features_dim = self.universe_size_w_rfa

#     def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
#         print(f"{observations['data'].shape=}")
#         x = self.cnn(observations['data'])
#         w = observations['weights'][:, None, :]
#         y = torch.cat((x.squeeze(-1), w), dim=1)
#         print(f"{x.detach().cpu().numpy()=}")
#         print(f"{w.detach().cpu().numpy()=}")
#         print(f"{y.detach().cpu().numpy()=}")
#         print(f"{x.shape=}, {w.shape=}, {y.shape=}")
#         return y

# class Custom_EIEE_CNN_Extractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 85):
#         super(Custom_EIEE_CNN_Extractor, self).__init__(observation_space, features_dim)
#         n_channels, self.universe_size_w_rfa, data_len = observation_space['data'].shape
#         n_output_channels = 16
#         self.cnn = nn.Sequential(
#             nn.Conv1d(n_channels, 8, kernel_size=(5,)),
#             nn.Tanh(),
#             nn.Conv1d(8, n_output_channels, kernel_size=(data_len-4,)),
#             nn.Tanh()
#         ).cuda()
#         self.v_cnn = torch.vmap(self.cnn, in_dims=2, out_dims=2)
#         # self.features_dim = self.universe_size_w_rfa

#     def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
#         # print(f"{observations['data'][:, :, 0, :].shape=}")
#         # print(f"{observations['data'][:, :, 0, :].detach().cpu().numpy()=}")
#         # print(f"{self.cnn(observations['data'][:, :, 0, :]).detach().cpu().numpy()=}")
#         # print(f"{observations['data'][:, :, 1, :].shape=}")
#         # print(f"{observations['data'][:, :, 1, :].detach().cpu().numpy()=}")
#         # print(f"{self.cnn(observations['data'][:, :, 1, :]).detach().cpu().numpy()=}")

#         # print(f"{observations['data'].shape=}")
#         # print(f"{observations['data'].detach().cpu().numpy()=}")
#         x = self.v_cnn(observations['data'])
#         # print(f"{x.shape=}")
#         # print(f"{x.detach().cpu().numpy()=}")
#         w = observations['weights'][:, None, :]
#         y = torch.cat((x.squeeze(-1), w), dim=1)
#         # print(f"{x.detach().cpu().numpy()=}")
#         # print(f"{w.detach().cpu().numpy()=}")
#         # print(f"{y.detach().cpu().numpy()=}")
#         # print(f"{x.shape=}, {w.shape=}, {y.shape=}")
#         return y

class Custom_EIEE_CNN_Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 85):
        super(Custom_EIEE_CNN_Extractor, self).__init__(observation_space, features_dim)
        n_channels, self.universe_size_w_rfa, data_len = observation_space['data'].shape
        n_output_channels = 16
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 4, kernel_size=(5,)),
            nn.ReLU(),
            nn.Conv1d(4, 8, kernel_size=(5,)),
            nn.ReLU(),
            nn.Conv1d(8, n_output_channels, kernel_size=(data_len-8,)),
            nn.ReLU()
        ).cuda()
        self.v_cnn = torch.vmap(self.cnn, in_dims=2, out_dims=2)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.v_cnn(observations['data'])
        w = observations['weights'][:, None, :]
        y = torch.cat((x.squeeze(-1), w), dim=1)
        return y


class Custom_EIEE_Network(nn.Module):
    def __init__(self, observation_space: gym.spaces.Dict, input_features: int = 17, *args, **kwargs):
        super(Custom_EIEE_Network, self).__init__(*args, **kwargs)
        self.observation_space = observation_space
        self.net = nn.Sequential(
            nn.Linear(input_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1, bias=False)
        )
        torch.nn.init.uniform_(self.net[4].weight, a=-0.2, b=0.2)
        self.vec_net = torch.vmap(torch.vmap(self.net))
        # print(f"{self.observation_space['data'].shape=}")
        # self.features_dim = observation_space['data'].shape[1]

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # print(f"network features median={features.median()}, std={features.std()}")
        x = self.vec_net(torch.transpose(features, 1, 2)).squeeze(-1)
        # print(f"x min={x.min()=}, max={x.max()=}, median={x.median()}, xhat std={x.std()}")
        x_hat = nn.functional.tanh(5 * x)
        # print(f"xhat min={x_hat.min()=}, max={x_hat.max()=}, median={x_hat.median()}, xhat std={x_hat.std()}")
        return x_hat
        # return nn.functional.softmax(x, dim=1)


from stable_baselines3.td3.policies import Actor
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.preprocessing import get_action_dim

class CustomActor(Actor):
    def __init__(self, observation_space: gym.spaces.Dict, *args, **kwargs):
        super(CustomActor, self).__init__(observation_space, *args, **kwargs)
        self.mu = Custom_EIEE_Network(observation_space)

class DenseMlp(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 16,
        n_layers: int = 2,
        squash_output: bool = False,
    ):
        super(DenseMlp, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.squash_output = squash_output
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim + input_dim, hidden_dim)
        if output_dim > 0:
            self.output_layer = nn.Linear(hidden_dim + input_dim, output_dim)
        self.output_activation = nn.Tanh()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        out_1 = F.relu(self.layer_1(input_tensor))
        input_2 = torch.cat([out_1, input_tensor], dim=1)
        out_2 = F.relu(self.layer_2(input_2))

        if self.output_dim < 0:
            return out_2

        input_3 = torch.cat([out_2, input_tensor], dim=1)
        out_3 = self.output_layer(input_3)

        if self.squash_output:
            out_3 = self.output_activation(out_3)
        return out_3

class DenseContinuousCritic(BaseModel):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: list[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []
        features_dim = 17 * 85 # TODO: bad idea
        # print(f"{features_dim=}, {action_dim=}, {net_arch=}, {n_critics=}")
        for idx in range(n_critics):
            # q_net = DenseMlp(features_dim + action_dim, 1, net_arch[0])
            q_net = nn.Sequential(
                Custom_EIEE_Network(observation_space, input_features=18),
                # nn.Softmax(dim=1),
                nn.Linear(85, 1)
            )
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with torch.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        # print(f"{obs['data'].shape=}, {actions.shape=}, {features.shape=}, {actions[:, None, :].shape=}, {features.flatten(1, 2).shape=}")
        qvalue_input = torch.cat([features, actions[:, None, :]], dim=1)
        # print(f"{qvalue_input.shape=}")
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with torch.no_grad():
            features = self.extract_features(obs, self.features_extractor)
        # print(f"features mean={features.mean().detach().cpu().numpy()}, std={features.std().detach().cpu().numpy()}")
        # print(f"actions median={actions.median().detach().cpu().numpy()}, mean={actions.mean().detach().cpu().numpy()}, std={actions.std().detach().cpu().numpy()}")
        q1 = self.q_networks[0](torch.cat([features, actions[:, None, :]], dim=1))
        # print(f"q1 mean={q1.mean().detach().cpu().numpy()}, std={q1.std().detach().cpu().numpy()}")
        return q1
        # return self.q_networks[0](torch.cat([features, actions[:, None, :]], dim=1))


from typing import Optional
from stable_baselines3.td3.policies import TD3Policy

class CustomTD3Policy(TD3Policy):
    def __init__(self, observation_space: gym.spaces.Dict, *args, **kwargs):
        super(CustomTD3Policy, self).__init__(observation_space, *args, **kwargs)
        self.observation_space = observation_space

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return CustomActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> DenseContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return DenseContinuousCritic(**critic_kwargs).to(self.device)











# Parallel environments
from stable_baselines3 import DDPG, SAC

train_env = PortfolioEnvWithTCost(dm=TrainDataManager(), rm=ProfitReward(), cp=0.01, cs=0.01)
# vec_env = make_vec_env(PortfolioEnvWithTCost, n_envs=4, env_kwargs={
#     'dm': TrainDataManager(),
#     'rm': DifferentialSharpeRatioReward(),
#     'cp': 0.01, # 0.10/365,
#     'cs': 0.01, # 0.10/365
# })

# Set seeds
random.seed(42)
np.random.seed(42)
# train_env.seed(42)
train_env.action_space.seed(43)
torch.manual_seed(42)

model = DDPG(CustomTD3Policy, train_env, buffer_size=250, verbose=1, policy_kwargs={
  'features_extractor_class': Custom_EIEE_CNN_Extractor,
  'net_arch': [16, 16]
})
model.learn(total_timesteps=1_000, log_interval=1)
# model.save("cnn_portoflio_policy")
