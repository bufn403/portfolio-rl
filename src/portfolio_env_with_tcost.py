from abc import abstractmethod
import numpy as np
import numpy.typing as npt
import gymnasium as gym


class AbstractPortfolioEnvWithTCost(gym.Env):
    def __init__(self, w_lb=0.0, w_ub=1.0, cp=0.0, cs=0.0, logging=True, sample="IN"):
        # set constants
        self.eta = 1 / 252
        self.cp, self.cs = cp, cs
        self.logging = logging
        self.sample = sample

        # get data, set problem size
        self.num_time_periods, self.universe_size = self.get_data()

        # set spaces
        assert w_lb <= w_ub
        self.observation_space = self.get_obs_space()
        self.action_space = gym.spaces.Box(
            low=w_lb,
            high=w_ub,
            shape=(self.universe_size + 1,),
            dtype=np.float32
        )

    @abstractmethod
    def get_obs_space(self) -> gym.spaces.Box:
        """Result is assigned to ``self.observation_space``"""
        pass

    @abstractmethod
    def get_data(self) -> tuple[int, int]:
        """
        This abstract function loads/fetches state data and stores it on the environment.
        This will be called during initialization. The properties assigned here should be
        accessed into the __compute_state() method. Note that the data should provide for
        one more than the number of time periods desired (for the initial state).
        :return: (number of time periods, number of stock tickers)
        """
        pass

    @abstractmethod
    def get_state(self) -> npt.NDArray[float]:
        """
        Computes and returns the new state at time ``self.t`` (to be used for
        calculating weights at the start of time period ``self.t+1``).
        When ``self.t == 0``, it should output the initial state.
        """
        pass

    @abstractmethod
    def get_prices(self) -> npt.NDArray[float]:
        """
        Obtains the security prices at time ``self.t`` (at the
        beginning of time period ``self.t+1``). When ``self.t == 0``, it
        should output the initial prices.
        """
        pass

    def initialize_reward(self):
        self.A, self.B = 0.0, 0.0

    def compute_reward(self) -> float:
        R = np.log(self.new_port_val / self.port_val)
        dA = R - self.A
        dB = R ** 2 - self.B
        if self.B - self.A ** 2 == 0:
            D = 0
        else:
            D = (self.B * dA - 0.5 * self.A * dB) / (self.B - self.A ** 2) ** (3 / 2)
        self.A += self.eta * dA
        self.B += self.eta * dB
        return D

    def find_mu(self, w_old: npt.NDArray[float], w_new = npt.NDArray[float]) -> float:
        cp, cs = self.cp, self.cs

        def f(mu: float) -> float:
            return ((1 - cp * w_new[-1] - (cs + cs - cs*cp) * (w_new[:-1] - mu * w_old[:-1]).clip(min=0).sum()) /
                    (1 - cp * w_old[-1]))

        mu = 0.0
        for _ in range(30):
            mu = f(mu)
        return mu

    def step(self, action: npt.NDArray[float]) -> tuple:
        action = action / action.sum()
        self.w_new = action
        self.t += 1
        self.v_new = self.get_prices()
        self.y = self.v_new / self.v
        self.mu = self.find_mu(self.y * self.w / (self.y * self.w).sum(), self.w_new)
        self.new_port_val = self.port_val * self.mu * (self.y @ self.w)

        self.reward = self.compute_reward()

        self.state = self.get_state()
        self.w = self.w_new
        self.v = self.v_new
        self.port_val = self.new_port_val

        if self.logging:
            info = {
                'port_val': self.port_val
            }
        else:
            info = {}

        finished = (self.t == self.num_time_periods)
        return self.state.copy(), self.reward, finished, False, info

    def reset(self, *args, **kwargs) -> tuple[np.ndarray, dict]:
        # portfolio weights (final is cash weight)
        self.w = np.zeros(self.universe_size + 1, dtype=float)
        self.w[-1] = 1.0

        self.port_val = 1.0

        self.initialize_reward()

        # compute and return initial state
        self.t = 0
        self.state = self.get_state()
        self.v = self.get_prices()
        return self.state.copy(), {}
