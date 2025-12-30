from collections.abc import Iterable
from typing import Any, Protocol, Sized, TypeVar

StateT = TypeVar("StateT", bound=Any)
ActionT = TypeVar("ActionT", bound=Any)


class SizedIterable[T: int](Sized, Iterable, Protocol):
    def __len__(self) -> T: ...
    def __getitem__(self, key: int) -> Any: ...
