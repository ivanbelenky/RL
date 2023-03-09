import copy
import time
from time import perf_counter
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
    Transition,
    TransitionException,
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


class Approximator(ABC):
    '''Approximator base class that implements caster methods
    as well as defining the basic interface of any approximator.
    It has to be updateable and callable. Updatability implies
    that it can change its inner attributes and hopefully learn. 
    '''
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


class ModelFreeSLPolicy(Policy):
    '''ModelFreeSLPolicy is for approximated methods what 
    ModelFreePolicy is for tabular methods.

    This policies are thought with tabular actions in mind, since
    the problem of continuous action spaces are a topic of ongoing 
    research and not yet standardized. For each a in the action-space A
    there will exist an approximator.
    '''
    def __init__(self, actions: Sequence[Any], q_hat: Approximator):
        self.actions = actions
        self.A = len(actions)
        self.q_hat = q_hat
    
    def update_policy(self, *args, **kwargs):
        self.q_hat.update(*args, **kwargs)

    def __call__(self, state: Any):
        action_idx = np.argmax([self.q_hat((state, a)) for a in self.actions])
        return self.actions[action_idx]
        

class EpsSoftSALPolicy(ModelFreeSLPolicy):
    def __init__(self, actions: Sequence[Any], q_hat: Approximator,
                 eps: float = 0.1):
        super().__init__(actions, q_hat)
        self.eps = eps

    def __call__(self, state):
        if np.random.rand() < self.eps:
            return np.random.choice(self.actions)
        return super().__call__(state)
    

class ModelFreeSL:
    '''
    ModelFreeSL stands for Model Free Stateless, this is 
    to approximate methods what ModelFree is to tabular ones.

    ModelFreeSL is used mostly internally for the seek of readability
    on solvers, but can be used standalone as well. The usual case
    for this is when you want to generate arbitrary episodes for a
    specific environment. This class will stand in between of the
    user implemented transitions and the solvers. In difference with 
    tabular ModelFree there is no room for validation previous to 
    runtime executions.
    '''

    def __init__(self, transition: Transition, rand_state: Callable,
                 policy: ModelFreeSLPolicy, gamma: float = 1): 
        self.policy = policy
        self.rand_state = rand_state
        self.transition = transition
        self.gamma = gamma
        self._validate_transition()

    def _validate_transition(self):
        start = perf_counter()
        while perf_counter() - start < 2:
            rand_s = self.rand_state()
            rand_a = np.random.choice(self.policy.actions)
            try:
                self.transition(rand_s, rand_a)
            except Exception as e:
                raise TransitionException(
                    f'Transition function is not valid: {e}')
        
    def random_sa(self):
        a = np.random.choice(self.policy.actions)
        s = self.rand_state()
        return s, a

    def generate_episode(self, 
                         s_0: Any, 
                         a_0: Any, 
                         policy: ModelFreeSLPolicy=None, 
                         max_steps: int=MAX_STEPS) -> List[EpisodeStep]:
        '''Generate an episode using given policy if any, otherwise
        use the one defined as the attribute'''
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


class SGDWA(Approximator):
    '''Stochastic Gradient Descent Weight-Vector Approximator

    Differentiable Value Function approximator dependent
    on a weight vector. Must define a gradient method. 
    '''
    def __init__(self, 
                 fs:int=None,
                 basis: Optional[Callable[[Any], np.ndarray]]=None):
        '''
        Parameters
        ----------
        fs: int
            feature shape, i.e. dimensionality of the function basis
        basis: Callable[[Any], np.ndarray], optional
            function basis defaults to identity. If not specified the
            signature must be Callable[[np.ndarray], np.ndarray] otherwise
            it will be probably fail miserably. 
        '''  
        self.fs = fs
        self.basis_name = basis.__name__
        self.basis = basis if basis else lambda x: x
        self.w = np.ones(self.fs)*W_INIT

    def grad(self, x: Any) -> np.ndarray:
        '''Return the gradient of the approximation'''
        return self.basis(x)

    def delta_w(self, U: float, alpha: float, x: Any) -> np.ndarray:
        return alpha * (U - self(x)) * self.grad(x)

    def update(self, U: float, alpha: float, x: Any) -> np.ndarray:
        '''Updates inplace the weight vector and returns it just in case'''
        dw = self.delta_w(U, alpha, x)
        self.w = self.w + dw
        return dw

    def __call__(self, x):
        return np.dot(self.w, self.basis(x))

    def copy(self):
        return copy.deepcopy(self)
        

LinearApproximator = SGDWA
