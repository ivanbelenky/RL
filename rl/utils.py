from abc import ABC, abstractmethod
from typing import (
    Any, 
    Sequence, 
    List, 
    Tuple,
    Callable,
    NewType
)

import numpy as np


MAX_STEPS = 1E3
MAX_ITER = int(1E4)
TOL = 5E-5
MEAN_ITERS = int(1E4)
W_INIT = 1E-3

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
        self.revindex = {i: v for i, v in enumerate(seq)}

    def get_index(self, v) -> Any:
        return self.index[v]
    
    def from_index(self, idx) -> Any:
        return self.revindex[idx]

    def random(self, value=False):
        rnd_idx = np.random.choice(self.N)
        if value:
            return self.seq[rnd_idx]
        return rnd_idx


class State(_TabularIndexer):
    pass


class Action(_TabularIndexer):
    pass


class StateAction(_TabularIndexer):
    pass


class _TabularValues:
    def __init__(self, values: np.ndarray, idx: _TabularIndexer):
        self.v = values
        self.idx = idx
        self.idx_val = {k:v for k,v in zip(idx.index.keys(), values)}

    def values(self):
        return self.v


class Vpi(_TabularValues):
    def __str__(self):
        return f'Vpi({self.v[:5]}...)'

class Qpi(_TabularValues):
    def __str__(self):
        return f'Vpi({self.v[:5]}...)'


VQPi = NewType('VQPi', Tuple[Vpi, Qpi, Policy])
Samples = NewType('Samples', Tuple[int, List[Vpi], List[Qpi], List[Policy]])
Transition = Callable[[Any, Any], Tuple[Tuple[Any, float], bool]]


class PQueue:
    def __init__(self, items: List[Tuple[float, Any]]):
        self.items = items
        self._sort()
    
    def _sort(self):
        self.items.sort(key=lambda x: x[0])
    
    def push(self, item, priority):
        self.items.append((priority, item))
        self._sort()
    
    def pop(self):
        return self.items.pop(0)[1]

    def empty(self):
        return len(self.items) == 0

class RewardGenerator:
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

    if transition.__code__.co_argcount != 2:
        raise TypeError(
            f"transition must have two positional arguments,"
            f" not {transition.__code__.co_argcount}")   
 

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


def _typecheck_all(tabular_idxs=None, transition=None, constants=None,
    booleans=None, policies=None):
    if tabular_idxs:
        _typecheck_tabular_idxs(*tabular_idxs)
    if transition:
        _typecheck_transition(transition)
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


def _check_ranges(values, ranges):
    for v, r in zip(values, ranges):
        if v < r[0] or v > r[1]:
            raise ValueError(f"{v} is out of range {r}")


def auto_cardinal(values, n):
    prod = np.array(np.meshgrid(*[values for _ in range(n)]))
    return prod.T.reshape(-1, n)