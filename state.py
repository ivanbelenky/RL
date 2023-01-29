from abc import ABC, abstractclassmethod
from typing import Any, Union, List

class State(ABC):
    @abstractclassmethod
    def get_index(self, state) -> Any:
        raise NotImplementedError
    
    @abstractclassmethod
    def from_index(self, action) -> Any:
        raise NotImplementedError