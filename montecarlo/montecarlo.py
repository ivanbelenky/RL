'''
RL

Copyright © 2023 Iván Belenky

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files, to deal without restriction, 
including without limitation the rights to  use, copy, modify, merge, publish, 
distribute, sublicense, and/or sell copies of this.
'''

from typing import Tuple, Union, NewType
from abc import ABC, abstractmethod

import numpy as np

from policy import Policy
from state import State
from action import Action

from solvers import (
    first_visit_monte_carlo,
)

EpisodeStep = NewType('EpisodeStep', Tuple[Union[int, State], Union[int, Action], float])

class MC(ABC):
    '''
        MC is an abstract class that is able to handle arbitrary
        word transition models. 
        
        The interface defines just one necessary implementation
        transition. 

        In order to solve  
    '''

    VQ_PI_SOLVERS = {
        'first_visit': first_visit_monte_carlo 
    }

    OPTIMAL_POLICY_SOLVERS = {
    }
    
    def __init__(
        self,
        states: Union[int, State],
        actions: Union[int, Action],
        gamma: float = 0.9,
        policy: Policy = None,
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
            

    @property
    def cum_return(self) -> float:
        return np.sum([r for _, r in self.episode])

    @property
    def discounted_return(self) -> float:
        return np.sum(
            [r*(self.gamma**i) for i,(_, r) in enumerate(self.history)])


    @abstractmethod
    def transition(self, 
        state: Union[int, State], 
        action: Union[int, Action]
        ) -> Tuple[EpisodeStep, bool]:
        '''
            The transition method will determine how the world reacts to a
            state action pair 
        '''

        raise NotImplementedError

    def _transition(self, state, action):
        (state, action, reward), end =  self.transition(state, action)
        
        if isinstance(state, State):
            state = state.get_index()
        if isinstance(action, Action):
            action = action.get_index()

        return (state, action, reward), end

        
    def generate_episode(self, state_0, action_0):
        episode = []
        end = False
        while end == False:
            (s, a, r), end = self._transition(s, a)
            self.episode.append((s, a, r))
        
        return episode

    

    def __call__(self, state: int = 0) -> Tuple[int, float]:
        
        return a