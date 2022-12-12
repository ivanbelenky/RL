"""
RL - Copyright © 2023 Iván Belenky @Leculette
"""


from abc import ABC, abstractmethod

import numpy 


MEAN_ITERS = int(1E4)


class RewardGenerator(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def generate(self, state: int = None) -> float:
        raise NotImplementedError

    def mean(self, state: int = None):
        total = 0
        for _ in range(MEAN_ITERS):
            total += self.generate(state=state)
        return total/MEAN_ITERS


class BernoulliRewardGenerator(RewardGenerator):
    def __init__(self, p: float):
        self.p = p
    
    def generate(self, state: int = None ) -> float:
        return numpy.random.binomial(1, self.p)


class GaussianRewardGenerator(RewardGenerator):
    def __init__(self, mean: float, std: float):
        self.mu = mean
        self.std = std
    
    def generate(self, state: int = None) -> float:
        return numpy.random.normal(self.mu, self.std)

    def mean(self) -> float:
        return self.mu


class UniformRewardGenerator(RewardGenerator):
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high
    
    def generate(self, state: int = None) -> float:
        return numpy.random.uniform(self.low, self.high)

    def mean(self) -> float:
        return (self.low + self.high)/2


class ParetoRewardGenerator(RewardGenerator):
    def __init__(self, alpha: float):
        self.alpha = alpha
    
    def generate(self, state: int = None) -> float:
        return numpy.random.pareto(self.alpha)

    def mean(self):
        return self.alpha/(self.alpha - 1)
    

class TriangularRewardGenerator(RewardGenerator):
    def __init__(self, low: float, high: float, mode: float):
        self.low = low
        self.high = high
        self.mode = mode
    
    def generate(self, state: int = None) -> float:
        return numpy.random.triangular(self.low, self.high, self.mode)

    def mean(self) -> float:
        return (self.low + self.high + self.mode)/3


class PoissonRewardGenerator(RewardGenerator):
    def __init__(self, lam: float):
        self.lam = lam
    
    def generate(self, state: int = None) -> float:
        return numpy.random.poisson(self.lam)

    def mean(self) -> float:
        return self.lam
    