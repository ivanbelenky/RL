'''
RL - Copyright © 2023 Iván Belenky @Leculette
'''

from typing import (
    Tuple, 
    Union, 
    Sequence, 
    Callable,
    List, 
    Any, 
    NewType, 
)

import numpy as np

from utils import Policy, State, Action
from solvers import (
    first_visit_monte_carlo,
    every_visit_monte_carlo,
    off_policy_first_visit,
    off_policy_every_visit,
    tdn,
    MAX_STEPS,
    MAX_ITER
)

EpisodeStep = NewType(
    'EpisodeStep', Tuple[int, int, float])


class ModelFreePolicy(Policy):
    def __init__(self, A: Union[Sequence[Any], int], S: Union[Sequence[Any], int]):
        if not isinstance(A, int):
            A = len(A)
        if not isinstance(S, int):
            S = len(S)
        self.A = A
        self.S = S
        self.pi = np.ones((S, A))/A

    def __call__(self, state: int):
        return np.random.choice(self.A, p=self.pi[state])

    def pi_as(self, action: int, state: int):
        return self.pi[state, action]
    
    def update_policy(self, q, s):
        qs_mask = q[s] == np.max(q[s])
        self.pi[s] = np.where(qs_mask, 1/qs_mask.sum(), 0)
        
        
class EpsilonSoftPolicy(ModelFreePolicy):
    def __init__(self, A, S, eps):
        super.__init__(A, S)
        self.Ɛ = eps

    def update_policy(self, q, s):
        self.pi[s, :] = self.Ɛ/self.A
        self.pi[s, np.argmax(q)] += 1 - self.Ɛ


class ModelFree:
    '''
    MC is an abstract class that is able to handle arbitrary
    word transition models. 
    
    The interface defines just one necessary implementation
    transition. 
    '''

    VQ_PI_SOLVERS = {
        'first_visit_mc': first_visit_monte_carlo,
        'every_visit_mc': every_visit_monte_carlo, 
        'off_policy_first_visit': off_policy_first_visit,
        'off_policy_first_visit': off_policy_every_visit,
        'temporal_difference': tdn,
    }

    OPTIMAL_POLICY_SOLVERS = {
    }
    
    def __init__(
        self,
        states: Sequence[Any],
        actions: Sequence[Any],
        transition: Callable,
        gamma: float = 0.9,
        policy: ModelFreePolicy = None,
    ):
    
        self.policy = policy
        self.state = State(states)
        self.action = Action(actions)
        self.transition = transition
        self.gamma = gamma
        self.policy = policy if policy else ModelFreePolicy(
            self.action.N, self.state.N)
  
    def random_sa(self, value=False):
        s = self.state.random(value)
        a = self.action.random(value)
        return s, a

    def _to_index(self, state, action):
        if not isinstance(state, int):
            state = self.state.get_index(state)
        if not isinstance(action, int):
            action = self.action.get_index(action)

        return state, action

    def _transition(self, 
        state: Any, 
        action: Any,
        ) -> Tuple[Tuple[Any, Union[float, int]], bool]:
        
        # to help debug ill defined transitions
        try:
            (s, r), end = self.transition(state, action)
        except Exception as e:
            raise Exception(f"Transition method failed: {e}")    
                    
        if not isinstance(end, bool) or not isinstance(r, (float, int)):
            raise Exception(
                "Transition method must return (Any, float), bool"
                f" instead of ({type(s)}, {type(r)}), {type(end)}"
                )
        
        try:
            self.state.get_index(s)
            self.state.get_index(state)
            self.action.get_index(action)
        except Exception as e:
            raise Exception(
                f"Undeclared state or action in transition method: {e}")

        return (s, r), end

    def _solve(self, method: str = 'first_visit_mc', policy: ModelFreePolicy = None, 
        off_policy: ModelFreePolicy = None, max_episodes: int = MAX_ITER, 
        optimize: bool=False, max_steps: int=MAX_STEPS, exploring_starts=True
        ) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Individual state value functions and action-value functions
        vpi and qpi cannot be calculated for bigger problems. That
        constraint will give rise to parametrizations via DL.
        '''
        policy = policy if policy else self.policy
        solver = self.VQ_PI_SOLVERS.get(method)
        if not solver:
            raise ValueError(f"Method {method} does not exist")
        if not isinstance(policy, ModelFreePolicy):
            raise ValueError(f"Policy must be a ModelFreePolicy")

        if 'off_policy' in method:
            if not off_policy or not isinstance(off_policy, ModelFreePolicy):
                raise ValueError(
                    "Off policy method requires a behavior policy"
                    f"not {type(off_policy)}")
            return solver(self, off_policy, policy, max_episodes, max_steps,
                optimize=optimize)
        return solver(self, policy, max_episodes, max_steps, exploring_starts,
            optimize=optimize)

    def generate_episode(self,
        state_0: Any, 
        action_0: Any,
        policy: ModelFreePolicy = None,
        max_steps: int = MAX_STEPS
        ) -> List[EpisodeStep]:

        policy = policy if policy else self.policy

        episode = []
        end = False
        step = 0
        s_t_1, a_t_1 = state_0, action_0
        
        while (end != True) and (step < max_steps):
            (s_t, r_t), end = self._transition(s_t_1, a_t_1)
            (_s, _a), _r = self._to_index(s_t_1, a_t_1), r_t
            episode.append((_s, _a, _r))

            a_t = policy(self.state.get_index(s_t))
            s_t_1, a_t_1 = s_t, self.action.from_index(a_t)
            
            step += 1

        return episode