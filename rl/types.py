from collections.abc import Hashable, Iterable, Iterator, Sized
from typing import Protocol, TypeVar, runtime_checkable

StateT = TypeVar("StateT", bound=Hashable)
ActionT = TypeVar("ActionT", bound=Hashable)


@runtime_checkable
class SizedIterable[T](Iterable[T], Sized, Protocol):
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[T]: ...
