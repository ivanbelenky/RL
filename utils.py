from abc import ABC, abstractmethod
from typing import Any, Union, Sequence, List

import numpy as np


class Policy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, state: int = None) -> int:
        raise NotImplementedError

    @abstractmethod
    def update_policy(self, *args, **kwargs):
        raise NotImplementedError


class _TabularIndexer():
    '''
        Simple proxy for tabular state & actions.
    '''
    def __init__(self, seq: Sequence[Any]):
        self.seq = seq
        self.N = len(seq)
        self.index = {v: i for i, v in enumerate(seq)}
        
    def get_index(self, v) -> Any:
        return self.index[v]
    
    def from_index(self, idx) -> Any:
        for k, i in self.index.items():
            if i == idx:
                return k

    def random(self, value=False):
        rnd_idx = np.random.choice(self.N)
        if value:
            return self.seq[rnd_idx]
        return rnd_idx

class State(_TabularIndexer):
    pass


class Action(_TabularIndexer):
    pass


MEAN_ITERS = int(1E4)


class RewardGenerator():
    DISTRIBUTION = {
        'bernoulli': np.random.binomial,
        'gaussian': np.random.normal,
        'uniform': np.random.uniform,
        'exponential': np.random.exponential,
        'poisson': np.random.poisson,
        'pareto': np.random.pareto,
        'triangular': np.random.triangular,
    }

    @classmethod
    def generate(self, distribution='normal', *args, **kwargs) -> float:
        generator = self.DISTRIBUTION.get(distribution)
        if not generator:
            raise ValueError(f'Invalid distribution: {distribution}')
        return generator(*args, **kwargs)