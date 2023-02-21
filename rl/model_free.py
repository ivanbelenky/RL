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

from rl.utils import (
    Policy, 
    State, 
    Action,
    StateAction, 
    MAX_ITER, 
    MAX_STEPS
)


EpisodeStep = NewType(
    'EpisodeStep', Tuple[int, int, float])


class TransitionException(Exception):
    pass


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
        qs_mask = (q[s] == np.max(q[s]))
        self.pi[s] = np.where(qs_mask, 1.0/qs_mask.sum(), 0)
        
    def _make_deterministic(self):
        self.pi = np.eye(self.A)[np.argmax(self.pi, axis=1)]

        
class EpsilonSoftPolicy(ModelFreePolicy):
    def __init__(self, A, S, eps):
        super().__init__(A, S)
        self.Ɛ = eps

    def update_policy(self, q, s):
        # if there are multiple actions with the same value,
        # then we choose one of them randomly
        max_q = np.max(q[s])
        qs_mask = (q[s] == max_q)
        self.pi[s] = self.Ɛ/self.A
        self.pi[s, qs_mask] += (1 - self.Ɛ)/qs_mask.sum()


class ModelFree:
    '''
    ModelFree is the base holder of the states, actions, and 
    the transition defining an environment.

    ModelFree is used mostly internally for the seek of readability
    on solvers, but can be used standalone as well. The usual case
    for this is when you want to generate arbitrary episodes of a
    specific environment. This class will stand in between of the
    user implemented transitions and validate its correct behavior. 
    '''

    def __init__(self, states: Sequence[Any], actions: Sequence[Any], 
        transition: Callable, gamma: float = 1, policy: ModelFreePolicy = None
    ):
    
        self.policy = policy
        self.states = State(states)
        self.actions = Action(actions)
        self.stateaction = StateAction(
            [(s,a) for s,a in zip(states, actions)])
        self.transition = transition
        self.gamma = gamma
        self.policy = policy if policy else ModelFreePolicy(
            self.actions.N, self.states.N)

        self._validate_transition()
  
    def init_vq(self):
        v = np.zeros(self.states.N) 
        q = np.zeros((self.states.N, self.actions.N))
        return v,q 

    def random_sa(self, value=False):
        s = self.states.random(value)
        a = self.actions.random(value)
        return s, a

    def _to_index(self, state, action):
        state = self.states.get_index(state)
        action = self.actions.get_index(action)

        return state, action

    def _validate_transition(self):
        states = self.states.seq
        actions = self.actions.seq
        sa = [(s,a) for s in states for a in actions]
        
        success, fail_count = True, 0
        for s, a in sa:
            try:
                self.__validate_transition(s, a)
            except Exception as e:
                success = False
                fail_count += 1
                print(f"Warning: {e}") # TODO: change to logger
                
        if not success:
            raise TransitionException(
                f"Transition failed for {fail_count} state-action pairs")

    def __validate_transition(self, state: Any, action: Any,
        ) -> Tuple[Tuple[Any, Union[float, int]], bool]:
        
        try:
            (s, r), end = self.transition(state, action)
        except Exception as e:
            raise TransitionException(f"Transition method failed: {e}")    
                    
        if not isinstance(end, bool) or not isinstance(r, (float, int)):
            raise TransitionException(
                "Transition method must return (Any, float), bool"
                f" instead of ({type(s)}, {type(r)}), {type(end)}"
                )  
        try:
            self.states.get_index(s)
            self.states.get_index(state)
            self.actions.get_index(action)
        except Exception as e:
            raise TransitionException(
                f"Undeclared state or action in transition method: {e}")

        return (s, r), end

    def generate_episode(self, s_0: Any, a_0: Any, policy: ModelFreePolicy = None, 
        max_steps: int=MAX_STEPS) -> List[EpisodeStep]:

        policy = policy if policy else self.policy

        episode = []
        end = False
        step = 0
        s_t_1, a_t_1 = s_0, a_0
        while (end != True) and (step < max_steps):
            (s_t, r_t), end = self.transition(s_t_1, a_t_1)
            (_s, _a), _r = self._to_index(s_t_1, a_t_1), r_t
            episode.append((_s, _a, _r))
            a_t = policy(self.states.get_index(s_t))
            s_t_1, a_t_1 = s_t, self.actions.from_index(a_t)
            
            step += 1

        return episode

    def step_transition(self, state: int, action: int
    ) -> Tuple[Tuple[int, float], bool]:
    
        s, a = self.states.from_index(state), self.actions.from_index(action)
        (s_t, r_t), end = self.transition(s, a)
        s_new = self.states.get_index(s_t)
        return (s_new, r_t), end