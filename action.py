from typing import Any, Sequence, Union, List

import numpy as np

class Action():
    '''
        Simple Proxy case to perform all sorts of operations on action 
        spaces that are discrete.
    '''
    def __init__(self, actions: Sequence[Any]):
        self.actions = actions
        self.S = len(actions)
        self.index = {s: i for i, s in enumerate(actions)}
        
    def get_index(self, state) -> Any:
        return self.get_index[state]
    
    def from_index(self, idx) -> Any:
        for k, i in self.index.items():
            if i == idx:
                return k

    def random(self):
        return np.random.choice(self.states)
