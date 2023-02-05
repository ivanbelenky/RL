from abc import ABC, abstractmethod
from typing import (
    Any, 
    Union, 
    Sequence, 
    List, 
    Dict,
    Tuple,
    Callable,
    NewType
)

import numpy as np


MAX_STEPS = 1E3
MAX_ITER = int(1E4)
TOL = 1E-6
MEAN_ITERS = int(1E4)


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


class _TabularValues:
    def __init__(self, values: np.ndarray, idx: _TabularIndexer):
        self.v = values
        self.idx = idx
        self.idx_val = {k:v for k,v in zip(idx.index.keys(), values)}

    def values(self):
        return self.idx_val


class Vpi(_TabularValues):
    pass    

class Qpi(_TabularValues):
    pass


VQPi = NewType('VQPi', Tuple[Vpi, Qpi, Policy])
Samples = NewType('Samples', Tuple[int, List[Vpi], List[Qpi], List[np.ndarray]])
Transition = Callable[[Any, Any], Tuple[Tuple[Any, float], bool]]


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


def _typecheck_tabular_idxs(*args):
    for arg in args:
        if not isinstance(arg, Sequence):
            raise TypeError(
                f"Tabular Indexes must be Sequence, not {type(arg)}")
    
 
def _typecheck_transition(transition):
    if not isinstance(transition, Callable):
        raise TypeError(
            f"transition must be a Callable, not {type(transition)}")
    
    if len(transition.__code__.co_varnames) != 2:
        raise TypeError(
            "transition must have 2 arguments, not ",
            len(transition.__code__.co_varnames))
    

def _typecheck_constants(*args):
    for arg in args:
        if not isinstance(arg, (float, int)):
            raise TypeError(
                f"Constants must be float or int, not {type(arg)}")


def _typecheck_booleans(*args):
    for arg in args:
        if not isinstance(arg, bool):
            raise TypeError(
                f"Booleans must be bool, not {type(arg)}")

def _typecheck_policies(*args):
    for arg in args:
        if not isinstance(arg, Policy):
            raise TypeError(
                f"Policies must be Policy, not {type(arg)}")


def _typecheck_all(tabular_idxs=None, callables=None, constants=None,
    booleans=None, policies=None):
    if tabular_idxs:
        _typecheck_tabular_idxs(*tabular_idxs)
    if callables:
        _typecheck_transition(*callables)
    if constants:
        _typecheck_constants(*constants)
    if booleans:
        _typecheck_booleans(*booleans)
    if policies:
        _typecheck_policies(*policies)


def _get_sample_step(samples, n_episodes):
    if samples > n_episodes:
        samples = n_episodes
    if samples > 1E3:
        samples = int(1E3)
    sample_step = int(n_episodes / samples)
    return sample_step