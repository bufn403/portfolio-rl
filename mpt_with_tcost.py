import numpy as np
import pandas as pd
import gymnasium as gym
import pandas_ta as ta
from tensorly.decomposition import Tucker
import torch
from portfolio_env_with_tcost import AbstractPortfolioEnvWithTCost
from typing import Tuple, Optional
import numpy.typing as npt

class MPTWithTCost(AbstractPortfolioEnvWithTCost):

    def get_obs_space(self) -> gym.spaces.Box:
        pass

    def get_data(self) -> Tuple[int, int]:
        pass

    def get_state(self) -> npt.NDArray[float]:
        pass

    def get_prices(self) -> npt.NDArray[float]:
        pass

    