from abc import ABC, abstractmethod
from typing import Any, Union, List

class Action(ABC):
    @abstractmethod
    def get_index(self, action) -> Any:
        raise NotImplementedError
    
