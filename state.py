from typing import Any, Union, Sequence, List

import numpy as np

class State():
    '''
        Simple Proxy case to perform all sorts of operations on state 
        spaces that are discrete.
    '''
    def __init__(self, states: Sequence[Any]):
        self.states = states
        self.S = len(states)
        self.index = {s: i for i, s in enumerate(states)}
        
    def get_index(self, state) -> Any:
        return self.index[state]
    
    def from_index(self, idx) -> Any:
        for k, i in self.index.items():
            if i == idx:
                return k

    def random(self):
        return np.random.choice(self.states)
        
