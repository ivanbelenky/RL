'''
RL

Copyright © 2023 Iván Belenky

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files, to deal without restriction, 
including without limitation the rights to  use, copy, modify, merge, publish, 
distribute, sublicense, and/or sell copies of this.
'''

from typing import List, Tuple, Union, NewType
from abc import ABC, abstractmethod

import numpy as np

from policy import Policy
from state import State
from action import Action

from solvers import (
    first_visit_monte_carlo,
    every_visit_monte_carlo
)

EpisodeStep = NewType('EpisodeStep', Tuple[Union[int, State], Union[int, Action], float])

MAX_STEPS = 1E3


class MCPolicy(Policy):

    def __call__(self):
        pass

    def update_policy(self):
        pass



class MC(ABC):
    '''
        MC is an abstract class that is able to handle arbitrary
        word transition models. 
        
        The interface defines just one necessary implementation
        transition. 

        In order to solve  
    '''

    VQ_PI_SOLVERS = {
        'first_visit': first_visit_monte_carlo,
        'every_visit': every_visit_monte_carlo, 
    }

    OPTIMAL_POLICY_SOLVERS = {
    }
    
    def __init__(
        self,
        states: Union[int, State],
        actions: Union[int, Action],
        gamma: float = 0.9,
        policy: MCPolicy = None,
    ):
    
        self.policy = policy
        self.states = states
        self.actions = actions
        self.gamma = gamma

        self._validate_attr()

    def _validate_attr(self):
        if isinstance(self.states, int):
            self.states = np.arange(self.states)
        if isinstance(self.actions, Action):
            self.actions = np.arange(self.actions)
        
        if self.policy == None:
            if isinstance(self.action, Action) or isinstance(self.states, State):
                raise ValueError(
                    'If state is not integer, policy must be defined')

    def _cast_sa(self, state, action):
        if isinstance(state, State):
            state = state.get_index()
        if isinstance(action, Action):
            action = action.get_index()

        return state, action

    @abstractmethod
    def transition(self, 
        state: Union[int, State], 
        action: Union[int, Action]
        ) -> Tuple[Tuple[State, float], bool]:
        '''
            The transition method will determine how the world reacts to a
            state action pair 
        '''

        raise NotImplementedError

    def generate_episode(self,
        state_0: Union[State, int], 
        action_0: Union[Action, int],
        policy: MCPolicy,
        max_steps: int = MAX_STEPS
        ) -> List[EpisodeStep]:

        policy = policy if policy else self.policy

        episode = []
        end = False
        step = 0
        s_t_1, a_t_1 = state_0, action_0
        
        while (end != True) or (step < max_steps):
            (s_t, r_t), end = self.transition(s_t_1, a_t_1)
            _s, _a, _r = self._cast_sa(s_t_1, a_t_1), r_t
            episode.append((_s, _a, _r))

            a_t = policy(s_t)
            s_t_1, a_t_1 = s_t, a_t
            
            step += 1

        return episode
    