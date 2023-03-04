import copy
from abc import ABC, abstractmethod
from typing import (
    Optional, 
    Callable, 
    Tuple, 
    Callable, 
    Sequence,
    Union,
    List,
    Any
)

import numpy as np

from rl.utils import (
    Policy, 
    EpisodeStep, 
    W_INIT, 
    MAX_ITER, 
    MAX_STEPS
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
- TD since involves bootstraping (it will be a semigradient method).

Therefore we generate the most abstract SGD method. The two most 
important parts of this methods is the U approximation to the real
value, and the value function approximator, this class should be
differentiable, or hold a gradient method.
'''


class ModelFreeSALPolicy(Policy):
    pass


class ModelFreeSAL:
    '''
    ModelFreeSAL stands for Model Free State-Action less, this is 
    to approximate methods what ModelFree is to tabular ones.

    ModelFreeSAL is used mostly internally for the seek of readability
    on solvers, but can be used standalone as well. The usual case
    for this is when you want to generate arbitrary episodes for a
    specific environment. This class will stand in between of the
    user implemented transitions and the solvers. In difference with 
    tabular ModelFree there is no room for validation previous to 
    runtime executions.
    '''

    def __init__(self, transition: Callable, gamma: float = 1,
        state_caster: Callable=None, action_caster: Callable=None,
        policy: ModelFreeSALPolicy = None): 

        self.policy = policy
        self.transition = transition
        self.gamma = gamma
        self.policy = policy if policy else ModelFreeSALPolicy()

        self._set_state_action_casters(state_caster, action_caster)

    def _set_state_action_casters(self, s_caster, a_caster):
        '''Set the casters for states and actions. If not defined, 
        behold the exceptions if you did not define them correctly.'''
        if not all([isinstance(c, Callable) for c in [s_caster, a_caster]]):
            raise TypeError("Casters must be callable.")
        self.cast_state = s_caster if s_caster else lambda s: s
        self.cast_action = a_caster if a_caster else lambda a: a
        
    def random_sa(self):
        # It could be random if range is given, but for more complicated
        # state action spaces it is going to be difficult to generalize.
        raise NotImplementedError

    def generate_episode(self, s_0: Any, a_0: Any, policy: ModelFreeSALPolicy=None, 
        max_steps: int=MAX_STEPS) -> List[EpisodeStep]:

        policy = policy if policy else self.policy
        episode = []
        end = False
        step = 0
        s_t_1, a_t_1 = s_0, a_0
        while (end != True) and (step < max_steps):
            (s_t, r_t), end = self.transition(s_t_1, a_t_1)
            episode.append((s_t_1, a_t_1, r_t))
            a_t = policy(s_t)
            s_t_1, a_t_1 = s_t, a_t
            step += 1

        return episode

    def step_transition(self, state: Any, action: Any
    ) -> Tuple[Tuple[Any, float], bool]:    
        return self.transition(state, action)


class Approximator(ABC):
    '''Approximator base class that implements caster methods
    as well as defining the basic interface of any approximator.
    It has to be updateable and callable. Updatability implies
    that it can change its inner attributes and hopefully learn. 
    '''
    def __init__(self, state_caster=None, action_caster=None):
        self._set_state_action_casters(state_caster, action_caster)

    def _set_state_action_casters(self, s_caster, a_caster):
        '''Set the casters for states and actions. If not defined, 
        behold the exceptions if you did not define them correctly.'''
        if s_caster:
            if not isinstance(s_caster, Callable):
                raise TypeError("State caster must be callable.")
        if a_caster:
            if not isinstance(a_caster, Callable):
                raise TypeError("Action caster must be callable.")
        self.cast_state = s_caster if s_caster else lambda s: s
        self.cast_action = a_caster if a_caster else lambda a: a

    @abstractmethod
    def __call__(self, s: Any, *args, **kwargs) -> float:
        '''Return the value of the approximation'''
        raise NotImplementedError

    @abstractmethod
    def update(self, *args, **kwargs) -> Union[None, np.ndarray]:
        '''Update the approximator'''
        raise NotImplementedError

    @abstractmethod
    def copy(self, *args, **kwargs) -> Any:
        '''Return a copy of the approximator'''
        raise NotImplementedError


class SGDWA(Approximator):
    '''Stochastic Gradient Descent Weight-Vector Approximator

    Differentiable Value Function approximator dependent
    on a weight vector. Must define a gradient method. 
    '''
    def grad(self, s: Any, *args, **kwargs) -> np.ndarray:
        '''Return the gradient of the approximation'''
        raise NotImplementedError

    def update(self, U: float, alpha: float, s: Any) -> np.ndarray:
        return self.w + alpha * (U - self(s)) * self.grad(s)

    def copy(self):
        return copy.deepcopy(self)
        


class LinearApproximator(SGDWA):
    '''Linear approximator for arbitrary finite dimension state space'''
    
    def __init__(self, k: int, fs:int=None,
        basis: Optional[Callable[[Any], np.ndarray]]=None,
        action_caster: Callable=None):
        '''
        Parameters
        ----------
        k: int
            state space dimensionality
        fs: int
            feature shape, i.e. dimensionality of the function basis
        basis: Callable[[Any], np.ndarray]
        '''  
        self.k = k
        self.fs = fs
        self.basis_name = basis.__name__
        self.basis = basis
        self.w = np.ones(self.fs)*W_INIT
        self._set_state_action_casters(basis, action_caster)
        
    def grad(self, s: Any) -> np.ndarray:
        return self.basis(s)

    def __call__(self, s):
        return np.dot(self.w, self.basis(s))