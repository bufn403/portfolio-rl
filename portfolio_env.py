import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import gymnasium as gym


class BasicPortfolioEnv(gym.Env):
  def __init__(self, T: int = 100):
    # set constants
    self.eta = 1/252
    self.T = T

    # get data
    self.times, self.tickers, self.price, self.ret, self.vol_20, self.vol_60, self.vix = self.__get_data()
    self.universe_size = len(self.tickers)

    # set spaces
    # TODO: bounds are bad
    self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.universe_size+1,), dtype=np.float32)
    self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.universe_size+1, self.T+1), dtype=np.float32)
  
  # @staticmethod
  def __get_data(self) -> tuple:
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

  def __compute_state(self) -> np.ndarray:
    """
    stock and index data just before the beginning of day t
    """
    s = np.zeros((self.universe_size+1, self.T+1))
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
