from collections.abc import Sized
from typing import Any, Iterable, Iterator, Protocol, TypeVar, runtime_checkable

StateT = TypeVar("StateT", bound=Any)
ActionT = TypeVar("ActionT", bound=Any)


@runtime_checkable
class SizedIterable[T: int](Iterable, Sized, Protocol):
    def __len__(self) -> T: ...
    def __iter__(self) -> Iterator[Any]: ...
    def __getitem__(self, key: int) -> Any: ...
