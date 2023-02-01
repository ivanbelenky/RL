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
    MAX_STEPS
)

EpisodeStep = NewType(
    'EpisodeStep', Tuple[int, int, float])


class ModelFreePolicy(Policy):

    def __init__(self, A, S):
        self.A = A
        self.S = S
        self.pi = np.ones((S, A))/A

    def __call__(self, state: int):
        return np.random.choice(self.A, p=self.pi[state])

    def update_policy(self, q, s):
        q_mask = q == q[np.max(q)]
        self.pi[s, q_mask] = 1/q_mask.sum()


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
        'first_visit': first_visit_monte_carlo,
        'every_visit': every_visit_monte_carlo, 
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
  
    def random_sa(self):
        s = self.state.random()
        a = self.action.random()
        return s, a

    def _to_index(self, state, action):
        if isinstance(state, State):
            state = state.get_index()
        if isinstance(action, Action):
            action = action.get_index()

        return state, action

    def _transition(self, 
        state: Any, 
        action: Any,
        ) -> Tuple[Tuple[Any, float], bool]:
        
        # to help debug ill defined transitions
        try:
            (s, r), end = self.transition(state, action)

        except Exception as e:
            raise Exception(f"Transition method failed: {e}")    
                    
        if not isinstance(end, bool) or not isinstance(r, float):
            raise Exception(
                "Transition method must return (Any, float), bool")
        
        try:
            self.state.get_index(s)
            self.state.get_index(state)
            self.action.get_index(action)
        except Exception as e:
            raise Exception(
                f"Undeclared state or action in transition method: {e}")

        return (s, r), end

    def generate_episode(self,
        policy: ModelFreePolicy,
        state_0: Any, 
        action_0: Any,
        max_steps: int = MAX_STEPS
        ) -> List[EpisodeStep]:

        policy = policy if policy else self.policy

        episode = []
        end = False
        step = 0
        s_t_1, a_t_1 = state_0, action_0
        
        while (end != True) or (step < max_steps):
            (s_t, r_t), end = self._transition(s_t_1, a_t_1)
            _s, _a, _r = self._to_index(s_t_1, a_t_1), r_t
            episode.append((_s, _a, _r))

            a_t = policy(s_t)
            s_t_1, a_t_1 = s_t, a_t
            
            step += 1

        return episode