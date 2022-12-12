"""
RL - Copyright © 2023 Iván Belenky @Leculette
"""

from typing import List, Union, Any
from abc import ABC, abstractmethod

class Policy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, state: int = None) -> int:
        raise NotImplementedError

    @abstractmethod
    def update_policy(self, *args, **kwargs):
        raise NotImplementedError