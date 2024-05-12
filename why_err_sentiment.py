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
            'data': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6, self.universe_size, 10), dtype=np.float32),
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
          return ticker_ok[ticker] and ticker not in ['GOOG', 'EXC']
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
        self.ret = np.log(self.stock_df.pct_change().iloc[1:, :] + 1)
    
        # get times and tickers
        self.times = df.date.unique()[1:]
        self.tickers = df.TICKER.unique()

        # read and clean sentiment data
        sec_df = pd.read_csv('sec_sentiment.csv', dtype='string')
        del sec_df['Unnamed: 0']
        sec_df['fdate'] = pd.to_datetime(sec_df['fdate'])
        sec_df.Sentiment = sec_df.Sentiment.astype(float)
        sec_df.lm_negative = sec_df.lm_negative.astype(float)
        sec_df.lm_positive = sec_df.lm_positive.astype(float)
        sec_df.lm_uncertainty = sec_df.lm_uncertainty.astype(float)
        sec_df = sec_df[sec_df.TICKERH.isin(self.tickers)]
        for sec_ticker in sec_df.TICKERH.unique():
            assert sec_ticker in self.tickers
        for df_ticker in self.tickers:
            assert df_ticker in sec_df.TICKERH.unique()
        sec_df.lm_negative = sec_df.groupby('TICKERH')['lm_negative'].transform(lambda v: v.ffill())
        sec_df.lm_positive = sec_df.groupby('TICKERH')['lm_positive'].transform(lambda v: v.ffill())
        sec_df.lm_uncertainty = sec_df.groupby('TICKERH')['lm_uncertainty'].transform(lambda v: v.ffill())

        # fill in missing dates
        date_range = pd.date_range(sec_df.fdate.min(), sec_df.fdate.max(), freq='D')
        full_df = pd.DataFrame({'fdate': list(date_range)}).merge(pd.DataFrame({'TICKERH': list(sec_df.TICKERH.unique())}), how='cross')
        full_df['neg'] = np.nan
        full_df['pos'] = np.nan
        full_df['unc'] = np.nan
        for ticker in sec_df.TICKERH.unique():
            ticker_df = sec_df[sec_df.TICKERH == ticker]
            ticker_df.index = pd.DatetimeIndex(ticker_df.fdate)
            ticker_df = ticker_df.reindex(date_range, fill_value=np.nan)
            full_df.loc[full_df.TICKERH == ticker, 'neg'] = ticker_df.lm_negative.ffill().values
            full_df.loc[full_df.TICKERH == ticker, 'pos'] = ticker_df.lm_positive.ffill().values
            full_df.loc[full_df.TICKERH == ticker, 'unc'] = ticker_df.lm_uncertainty.ffill().values
        
        # create pivot tables
        self.neg_sent_df = full_df.pivot(index='fdate', columns='TICKERH', values='neg').astype(float)
        self.pos_sent_df = full_df.pivot(index='fdate', columns='TICKERH', values='pos').astype(float)
        self.unc_sent_df = full_df.pivot(index='fdate', columns='TICKERH', values='unc').astype(float)
        
        self.num_time_periods = len(self.times)-15-1
        self.universe_size = len(self.tickers)
        print(f"{self.universe_size=}")
        return self.num_time_periods, self.universe_size
    
    def get_state(self, t: int, w: npt.NDArray[np.float64], port_val: np.float64) -> npt.NDArray[np.float64]:
        # today is self.times[self.t+10]
        s = np.zeros((6, self.universe_size, 10))
        s[0, :, :] = self.stock_df.loc[self.times[t:t+10], :].to_numpy().T
        s[1, :, :] = self.high_df.loc[self.times[t:t+10], :].to_numpy().T
        s[2, :, :] = self.low_df.loc[self.times[t:t+10], :].to_numpy().T
        s[3, :, :] = self.neg_sent_df.loc[self.times[t:t+10], self.tickers].to_numpy().T
        s[4, :, :] = self.pos_sent_df.loc[self.times[t:t+10], self.tickers].to_numpy().T
        s[5, :, :] = self.unc_sent_df.loc[self.times[t:t+10], self.tickers].to_numpy().T
        assert np.all(~np.isnan(s)), f"State contains NaNs, {s[0, :, :]=}, {s[1, :, :]=}, {s[2, :, :]=}, {s[3, :, :]=}, {s[4, :, :]=}, {s[5, :, :]=}"
        return {'data': s, 'weights': w}

    def get_prices(self, t: int) -> npt.NDArray[np.float64]:
        # today is self.times[self.t+10]
        return np.append(self.stock_df.loc[self.times[t+10], :].to_numpy().flatten(), 1.0)


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


from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class Custom_EIEE_CNN_Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 984):
        super(Custom_EIEE_CNN_Extractor, self).__init__(observation_space, features_dim)
        n_channels, self.universe_size, data_len = observation_space['data'].shape
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 6, kernel_size=(1, 3)),
            nn.ReLU(),
            nn.Conv2d(6, 8, kernel_size=(1, 3)),
            nn.ReLU(),
            nn.Conv2d(8, 10, kernel_size=(1, 3)),
            nn.ReLU(),
            nn.Conv2d(10, 12, kernel_size=(1, data_len-6)),
        ).cuda()

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.cnn(observations['data'])
        return x.flatten(start_dim=1)


# Parallel environments
from stable_baselines3 import DDPG, SAC
from stable_baselines3.common.noise import NormalActionNoise

train_env = PortfolioEnvWithTCost(dm=TrainDataManager(), rm=ProfitReward(), cp=0.01, cs=0.01)

# Set seeds
random.seed(42)
np.random.seed(42)
train_env.action_space.seed(43)
torch.manual_seed(42)

model = DDPG('MultiInputPolicy', train_env, buffer_size=3*10**5, verbose=1, policy_kwargs={
  'features_extractor_class': Custom_EIEE_CNN_Extractor,
  # 'net_arch': [50, 50],
}, action_noise=NormalActionNoise(mean=0, sigma=0.10*np.ones(83)))
model.learn(total_timesteps=10**10, log_interval=1)




