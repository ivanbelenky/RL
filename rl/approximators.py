from abc import ABC, abstractclassmethod
from typing import Optional, Callable

import numpy as np

from rl.utils import auto_cardinal, W_INIT

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
- TD since involves bootstraping (it will be a semigradient method).

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

    def update(self, U: float, alpha: float, s: np.ndarray):
        return self.w + alpha * (U - self(s)) * self.grad(s)


class LinearApproximator(SGDWA):
    '''Linear approximator for arbitrary finite dimension state space'''
    
    def __init__(self, k: int, n: int=None, basis: str='poly', fs_xs:int=None,
        basis_xs: Optional[Callable[[np.ndarray], np.ndarray]]=None):
        '''
        Parameters
        ----------
        k: int
            Dimension of the state space
        n: int
            Order of the basis function, (n+1)^k features for poly and fourier
        basis: str, optional
            Basis function to use, either 'poly' or 'fourier', by default 'poly'
        '''  
        self.n = n
        self.k = k
        self.fs_xs = fs_xs
        self.basis = basis if not basis_xs else basis_xs.__name__
        self._set_basis(basis_xs)

    def set_weights(self, fs: int):
        '''fs: feature shape'''
        self.w = np.ones(fs)*W_INIT

    def _set_basis(self, basis_xs) -> None:
        if basis_xs:
            self.basis_xs = basis_xs
            self.set_weights(self.fs_xs)
            return

        n_set = np.arange(self.n+1)
        cij = auto_cardinal(n_set, self.k)

        # if one of the defined basis is defined fs is set 
        # and it has (n+1)^k value
        self.set_weights((self.n+1)**self.k)
        if self.basis == 'poly':
            def basis(s):
                xs = [np.prod(s**cj) for cj in cij]
                return np.array(xs)
        if self.basis == 'fourier':
            def basis(s):
                xs = [np.cos(np.pi*np.dot(s, cj)) for cj in cij]
                return np.array(xs)

        self.basis_xs = basis 

    def grad(self, s: np.ndarray) -> np.ndarray:
        return self.basis_xs(s)

    def __call__(self, s):
        return np.dot(self.w, self.basis_xs(s))