from abc import ABC, abstractmethod
from typing import Any, Union, List

class State(ABC):
    @abstractmethod
    def get_index(self, state) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def from_index(self, action) -> Any:
        raise NotImplementedError