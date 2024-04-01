import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from abc import abstractmethod
import pandas_ta as ta
from tensorly.decomposition import Tucker
import torch
import jax
from functools import partial
import jax.numpy as jnp


class AbstractPortfolioEnv(gym.Env):
  """
  Abstract class that provides functionality for all environments to load 
  the same data. Environments should inerhit from this class and implement
  the interface specified by gym.Env. 
  """
  def get_data(self, sample = "IN") -> tuple:
    # read SNP data
    df = pd.read_csv('crsp_snp100_2010_to_2024.csv', dtype='string')
    # convert datatypes
    df = df[['date', 'TICKER', 'PRC', 'VOL', 'ASKHI', 'BIDLO']]
    df.date = pd.to_datetime(df.date)
    df.astype({
      'PRC': float,
      'VOL': float,
      'ASKHI': float,
      'BIDLO': float
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
    if sample == "IN":
      df = df[(df.date.dt.year >= self.start_year) & (df.date.dt.year <= self.end_year)]
    else:
      # In out of sample, we need lookback
      df = df[(df.date.dt.year >= self.start_year - 1) & (df.date.dt.year <= self.end_year)]

    # create numpy array
    pivot_df = df.pivot(index='date', columns='TICKER', values='PRC')
    stock_array = pivot_df.values.astype(float)
    ret_array = np.log(1 + np.diff(stock_array, axis=0) / stock_array[:-1, :])
    # get times and dickers
    times = df.date.unique()[1:]
    tickers = df.TICKER.unique()
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
    vol_20 = idx_df.loc[times].vol_20.values
    vol_60 = idx_df.loc[times].vol_60.values

    # get vix data
    vix_df = pd.read_csv('crsp_vix_2010_to_2024.csv', dtype={
      'Date': 'string',
      'vix': float
    })
    vix_df.Date = pd.to_datetime(vix_df.Date)
    vix_df.set_index('Date', inplace=True)
    vix = vix_df.loc[times].vix.values
    return times, tickers, stock_array, ret_array, vol_20, vol_60, vix


class BasicEnv(AbstractPortfolioEnv):
  """
  https://icaps23.icaps-conference.org/papers/finplan/FinPlan23_paper_4.pdf
  Deep RL for Portfolio Optimization 
  by Sood, Papasotiriou, Vaiciulis, Balch
  """
  def __init__(self, T: int = 100, start_year: int = 2010, end_year: int = 2019):
    # Set constants
    self.eta = 1/252
    self.T = T
    self.start_year = start_year
    self.end_year = end_year
    # Get data
    self.times, self.tickers, self.price, self.ret, self.vol_20, self.vol_60, self.vix = self.get_data()
    self.universe_size = len(self.tickers)

    # Box for continuous spaces
    # TODO: bounds are bad
    self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.universe_size + 1,), dtype=np.float32)
    self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.universe_size + 1, self.T+1), dtype=np.float32)
  

  def __compute_state(self) -> np.ndarray:
    """
    stock and index data just before the beginning of day t
    """
    s = np.zeros((self.universe_size + 1, self.T + 1))
    s[:, 0] = self.w
    s[1:, :-1] = self.ret[self.t-self.T:self.t, :].T
    s[-1, 1] = self.vol_20[self.t-1]
    s[-1, 2] = self.vol_20[self.t-1] / self.vol_60[self.t-1]
    s[-1, 3] = self.vix[self.t-1]
    return s
  
  def step(self, action: np.ndarray) -> tuple:
    """
    action taken at the beginning of day t.
    returns and indicators from close of previous day.
    reward and next state computed at end of day.
    """
    action = action / action.sum()
    self.w = action

    # liquidate everything
    port_val = self.v[:-1] @ self.price[1+self.t-1, :] + self.v[-1]
    # reassign shares and cash according to new weights
    self.v[:] = 0.0
    self.v[:-1] = port_val * self.w[:-1] / self.price[1+self.t-1, :]
    self.v[-1] = port_val * self.w[-1]
    # create next state
    self.t += 1
    next_state = self.__compute_state()
    # compute reward
    new_port_val = self.v[:-1] @ self.price[1+self.t-1, :] + self.v[-1]
    assert new_port_val > 0, f"{new_port_val=}, {self.v=}"
    R = np.log(new_port_val / port_val)
    dA = R - self.A
    dB = R**2 - self.B
    if self.B - self.A**2 == 0:
      D = 0
    else:
      D = (self.B * dA - 0.5 * self.A * dB) / (self.B - self.A**2)**(3/2)
    self.A += self.eta * dA
    self.B += self.eta * dB
    return next_state, D, (self.t == len(self.times)), False, {
      'port_val': new_port_val,
    }

  def reset(self, *args, **kwargs) -> tuple[np.ndarray, dict]:
    # initial time
    self.t = self.T
    # portfolio weights (final is cash weight)
    self.w = np.zeros(self.universe_size+1, dtype=float)
    self.w[-1] = 1.0
    # portfolio shares (final is raw cash)
    self.v = np.zeros(self.universe_size+1, dtype=float)
    self.v[-1] = 1.0
    # initialize moving averages
    self.A, self.B = 0.0, 0.0
    self.state = self.__compute_state()
    return self.state.copy(), {}
  

""" JAX ENVIRONMENTS BELOW """

class MPT(AbstractPortfolioEnv):
  """
  Deep Reinforcement Learning for Stock Portfolio Optimization by connecting
  with modern portfolio theory. Needs 3 technical indicators:
    1. Moving Average: MA_w(t) = sum(P_t) / w, w = 28
    2. Relative Strength Index: RSI_t(w) = 100 * [1 - ( / (1 + 
          [EMA_w(max(P_t - P_{t-1}, 0)) / EMA_w(min(P_t - P_{t-1}, 0))]))]
        - EMA_w(t) = alpha * P(t) + (1-alpha) * EMA_w(t - 1), alpha = 2/(1+w), w = 14
    3. Moving Average Convergence Divergance: MACD_w(t) = sum(EMA_k(i)) - sum(EMA_d(i))
       where k = 26, d = 12, w = 9
  by Jang, Seong

  Let M = number of days in dataset, T = 252, and N = universe size
  """
  def __init__(self, start_year: int = 2010, end_year: int = 2019, sample = "IN"):
    # Set constants
    self.start_year, self.end_year = start_year, end_year
    self.w1, self.w2, self.w3 = 28, 14, 9
    self.m = 28
    self.t = self.m
    self.eta = 1/252
    # Get data
    self.times, self.tickers, self.price, _, _, _, _ = self.get_data(sample)
    self.universe_size = len(self.tickers)
    # portfolio weights (final is cash weight)
    self.w = jnp.zeros(self.universe_size+1, dtype=float)
    self.w = self.w.at[-1].set(1)
    self.conv3d = torch.nn.Conv3d(in_channels=4, out_channels=32, kernel_size=(1, 3, 1))
    self.relu = torch.nn.ReLU()
    self.tucker = Tucker(rank=(5, 5, 5, 5), init="random")

    self.A, self.B = 1e-9, -1e-9

    # portfolio shares (final is raw cash)
    self.v = jnp.zeros(self.universe_size+1, dtype=float)
    self.v = self.v.at[-1].set(1)

    df = (pd.DataFrame(self.price, columns=self.tickers))
    df["Date"] = pd.Series(self.times)
    df = df.dropna()
    # If in sample, start 1 year in front, else use sliver of training period as lookback
    start_year = start_year + 1 if sample == "IN" else start_year
    self.__pivot = df[df["Date"] > pd.to_datetime(f"{start_year}-01-01")].index[0] - self.m
    df = df.set_index("Date")
    mp = {ticker: pd.DataFrame(df[ticker]).rename(columns={ticker: "close"}) for ticker in self.tickers}
    # SMA df
    sma = {ticker: pd.DataFrame(mp[ticker].ta.sma(self.w1)).rename(columns={"SMA_28": ticker}) for ticker in self.tickers}
    sma_df = (pd.concat(sma.values(), axis=1))
    # RSI df
    rsi = {ticker: pd.DataFrame(mp[ticker].ta.rsi(self.w2)).rename(columns={"RSI_14": ticker}) for ticker in self.tickers}
    rsi_df = (pd.concat(rsi.values(), axis=1))
    # MACD df
    macd = {ticker: pd.DataFrame(mp[ticker].ta.macd(self.w3, 26, 12)["MACD_9_26_12"]).rename(columns={"MACD_9_26_12": ticker}) for ticker in self.tickers}
    macd_df = (pd.concat(macd.values(), axis=1))

    df = df[self.__pivot:]
    sma_df = sma_df[self.__pivot:]
    rsi_df = rsi_df[self.__pivot:]
    macd_df = macd_df[self.__pivot:]

    # Compute F = V @ Corr
    V = np.array([np.array(x.T) for x in [df, sma_df, rsi_df, macd_df]])
    Corr = np.array([np.corrcoef(x) for x in V])
    self.F = np.einsum('aki,akj->akij', V, Corr)

    # Box for continuous spaces
    self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.universe_size + 1,), dtype=np.float32)
    self.observation_space = gym.spaces.Box(low=-3000, high=3000, shape=(5, 5, 5, 5), dtype=np.float32)
  

  # @jax.jit
  # TODO: either jax jit or torch jit this
  def __compute_state(self) -> np.ndarray:
    """
    Technical indicators and correlations data just before day t
    """
    f = self.F[:, :, self.t - self.m : self.t, :]
    f = torch.Tensor(np.expand_dims(f, axis=0))
    f = self.relu(self.conv3d(f)).detach().numpy().squeeze(axis=0)
    core, _ = self.tucker.fit_transform(f)
    return core 

  def step(self, action: np.ndarray) -> tuple:
    """
    action taken at the beginning of day t.
    returns and indicators from close of previous day.
    reward and next state computed at end of day.
    """
    action = action / action.sum()
    self.w = action

    # liquidate everything
    port_val = self.v[:-1] @ self.price[1+self.t-1, :] + self.v[-1]
    # reassign shares and cash according to new weights
    self.v[:] = 0.0
    self.v[:-1] = port_val * self.w[:-1] / self.price[1+self.t-1, :]
    self.v[-1] = port_val * self.w[-1]
    # create next state
    self.t += 1
    next_state = self.__compute_state()
    # compute reward
    new_port_val = self.v[:-1] @ self.price[1+self.t-1, :] + self.v[-1]
    assert new_port_val > 0, f"{new_port_val=}, {self.v=}"
    R = np.log(new_port_val / port_val)
    dA = R - self.A
    dB = R**2 - self.B
    if self.B - self.A**2 == 0:
      D = 0
    else:
      D = (self.B * dA - 0.5 * self.A * dB) / (self.B - self.A**2)**(3/2)
    self.A += self.eta * dA
    self.B += self.eta * dB
    return next_state, D, (self.t == len(self.times)), False, {
      'port_val': new_port_val,
    } 

      
  
  def reset(self, *args, **kwargs) -> tuple[np.ndarray, dict]:
    """
    Resets the environment
    """
    self.t = self.m
    # Portfolio weights (final is cash weight)
    self.w = jnp.zeros(self.universe_size+1, dtype=float)
    self.w = self.w.at[-1].set(1)

    # Portfolio shares (final is raw cash)
    self.v = jnp.zeros(self.universe_size+1, dtype=float)
    self.v = self.v.at[-1].set(1)

    self.state = self.__compute_state() 
    self.A, self.B = 0.0, 0.0
    return self.state.copy(), {}
