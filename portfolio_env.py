import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import gymnasium as gym
from abc import abstractmethod

class AbstractPortfolioEnv(gym.Env):
  """
  Abstract class that provides functionality for all environments to load 
  the same data. Environments should inerhit from this class and implement
  the interface specified by gym.Env. 
  """
  def get_data(self) -> tuple:
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
    df = df[(df.date.dt.year >= self.start_year) & (df.date.dt.year <= self.end_year)]
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
  def __init__(self, T: int = 252, start_year: int = 2010, end_year: int = 2019):
    # Set constants
    self.T, self.t = T, T
    self.start_year = start_year
    self.end_year = end_year
    self.m = 28
    # MA parameters
    self.w1 = 28
    # RSI parameters
    self.w2 = 14
    self.alpha = 2 / (1 + self.w2)
    # MACD constants
    self.k, self.d, self.w3 = 26, 12, 9
    # Get data
    self.times, self.tickers, self.price, _, _, _, _ = self.get_data()
    # Now 28 day moving averages for all stocks
    self.ma = self.price.rolling(self.w1).mean()
    # EMAs
    self.ema_14 = self.price.rolling(14).ewa(self.alpha)
    self.ema_k = self.price.rolling(self.k).ewa(self.alpha)
    self.ema_d = self.price.rolling(self.d).ewa(self.alpha)
    self.universe_size = len(self.tickers)
    # Box for continuous spaces
    # TODO: bounds are bad
    self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.universe_size + 1,), dtype=np.float32)
    self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.universe_size + 1, self.T+1), dtype=np.float32)
  
  
  def __compute_state(self) -> np.ndarray:
    """
    Computes V: (4, m, n) and Cor: (4, n, n) in the paper and returns 
    F: (4, n, m, n). Returns stock
    and technical indicators just before day t
    """
    # TODO: from self.prices, build F from V and Cor and
    prices = self.price[self.t - self.m: self.t]
    # Calculate MA
    ma = self.price.rolling(self.m + self.w1 + 1).mean()[self.t - self.m: self.t]
    # Calculate RSI
    change = self.price[self.m + self.w2 + 1].diff()
    change_up, change_down = change.clip(lower=0, upper=0)
    up = change_up.rolling(self.w2).ewm(self.alpha)[self.t - self.m: self.t]
    down = change_down.rolling(self.w2).ewm(self.alpha).abs()[self.t - self.m: self.t]
    rsi = 100 * (1 - (1) / (1 + (up / down)))
    # Calculate MACD
    ema_k = self.price.rolling(self.k).ewm(self.alpha)
    ema_d = self.price.rolling(self.k).ewm(self.alpha)
    macd = (ema_k - ema_d).rolling(9).sum()
    V = [prices, ma, rsi, macd]
    Cor = [prices.corr(), ma.corr(), rsi.corr(), macd.corr()]
    F = np.einsum('aki,akj->akij', V, Cor)
    return F

  def step(self, action: np.ndarray) -> tuple:
    """
    action taken at the beginning of day t.
    returns and indicators from close of previous day.
    reward and next state computed at end of day.
    """
    action = action / action.sum()
    self.w = action
    
    # Create next state
    next_state = None

    # Compute reward


    return next_state, D, (self.t == len(self.times)), False, {
      'port_val': new_port_val,
    } 
    pass
  
  def reset(self, *args, **kwargs) -> tuple[np.ndarray, dict]:
    """
    Resets the environment
    """
    self.t = self.T
    self.state = self.__compute_state() 
    return self.state.copy(), {}
