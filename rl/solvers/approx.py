from abc import ABC, abstractclassmethod
from typing import (
    Union,
    Any
)

import numpy as np
from numpy.linalg import norm as lnorm

from rl.model_free import (
    ModelFree,
    ModelFreePolicy,
    EpsilonSoftPolicy
)
from rl.utils import (
    Policy,
    auto_cardinal,
    _typecheck_all,
    _get_sample_step,
    _check_ranges,
    VQPi,
    Samples,
    Transition,
    Vpi,
    Qpi,
    PQueue,
    MAX_ITER,
    MAX_STEPS,
    TOL,
    W_INIT
)


'''
All of this may change if the policy gradient methods are
similar to this implementation. Adding probably an `optimize`
keyword argument.

SGD and Semi Gradient Linear methods:

All Linear methods of this methods involve using a real value
weight matrix/vector that will be used in conjunction with a
basis function to approximate the value function.

wt+1 = wt - 1/2 * alpha * d[(v_pi - v_pi_hat)^2]/dw
wt+1 = wt + alpha * (v_pi - v_pi_hat)]*d[v_pi_hat]/dw
wt+1 = wt + alpha * (U - v_pi_hat)]*d[v_pi_hat]/dw

Since we dont have v_pi we have to use some estimator:
- MC would imply grabbing full trajectories and using them
- TD since involves bootstraping it will be a semigradient method

Therefore we generate the most abstract SGD method. The two most 
important parts of this methods is the U approximation to the real
value, and the value function approximator, this class should be
differentiable, or hold a gradient method.
'''

class SGDWA(ABC):
    '''Stochastic Gradient Descent Weight-Vector Approximator

    Differentiable Value Function approximator dependent
    on a weight tensor.
    '''

    @abstractclassmethod
    def __call__(self, s: np.ndarray, *args, **kwargs) -> float:
        '''Return the value of the approximation'''
        raise NotImplementedError

    @abstractclassmethod
    def grad(self, s: np.ndarray, *args, **kwargs) -> np.ndarray:
        '''Return the gradient of the approximation'''
        raise NotImplementedError

    @abstractclassmethod
    def update(self, U: float, alpha: float, s: np.ndarray):
        return self.w + alpha * (U - self(s)) * self.grad(s)


class LinearApproximator(SGDWA):
    '''Linear approximator for arbitrary finite dimension state space'''
    
    def __init__(self, k: int, n: int, basis: str='poly'):
        # (n+1)^k features when using poly basis
        # 2n*k features when using fourier basis
        
        self.n = n 
        self.k = k
        self.basis = basis
        self.w = np.ones(n)*W_INIT

        self._validate_attr()
        self._set_basis()
    
    def _validate_attr(self):
        if not isinstance(self.d, int):
            raise TypeError('d must be an integer')
        if not isinstance(self.w, np.ndarray):
            raise TypeError('w must be a numpy array')

    def _set_basis(self, s: np.ndarray) -> None:
        if self.basis == 'poly':
            n_set = np.arange(self.n+1)
            cij = auto_cardinal(n_set, self.k)
            def basis(s):
                xs = [np.prod(s**cj) for cj in cij]
                return np.array(xs)
        if self.basis == 'fourier':
            raise NotImplementedError

        self.basis = basis 

    def grad(self, s: np.ndarray) -> np.ndarray:
        return self.basis(s)

    def __call__(self, s):
        return np.dot(self.w, self.basis(s))