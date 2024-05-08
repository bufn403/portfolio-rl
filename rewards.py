import numpy as np
from portfolio_env_framework import AbstractRewardManager


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

