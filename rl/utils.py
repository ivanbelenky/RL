from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Literal, NewType, Self, Sequence

import matplotlib.pyplot as plt
import numpy as np

from rl.types import SizedIterable

MAX_STEPS = int(1e3)
MAX_ITER = int(1e4)
TOL = 5e-8
MEAN_ITERS = int(1e4)
W_INIT = 1e-3


class Policy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def update_policy(self, *args, **kwargs):
        raise NotImplementedError


class _TabularIndexer:
    """Simple proxy for tabular state & actions."""

    def __init__(self, seq: SizedIterable[Any] | Sequence[Any]):
        self.seq = seq
        self.N = len(seq)
        self.index = {v: i for i, v in enumerate(seq)}
        self.revindex = {i: v for i, v in enumerate(seq)}

    def get_index(self, v) -> Any:
        return self.index[v]

    def from_index(self, idx) -> Any:
        return self.revindex[idx]

    def random(self, value=False):
        rnd_idx = np.random.choice(self.N)
        if value:
            return self.seq[rnd_idx]
        return rnd_idx

    @classmethod
    def from_indexable(cls, indexable: _TabularIndexable):
        match indexable:
            case np.ndarray():
                return cls(indexable)  # type: ignore NOTE: error will be raised downstream if indexable is a non dimensional array
            case int():
                return cls([i for i in range(indexable)])
            case _:
                raise ValueError(f"Cannot index this type: {type(indexable)}")


class State(_TabularIndexer):
    pass


class Action(_TabularIndexer):
    pass


class StateAction(_TabularIndexer):
    pass


_TabularIndexable = np.ndarray | int


class _TabularValues:
    def __init__(self, values: np.ndarray, idx: _TabularIndexer | _TabularIndexable):
        self.v = values
        if not isinstance(idx, _TabularIndexer):
            self.idx: _TabularIndexer = _TabularIndexer.from_indexable(idx)

        self.idx: _TabularIndexer = idx
        self.idx_val = {k: v for k, v in zip(self.idx.index.keys(), values)}

    def values(self):
        return self.v

    def copy(self) -> Self:
        return deepcopy(self)


class Vpi(_TabularValues):
    def __str__(self):
        return f"Vpi({self.v[:5]}...)"


class Qpi(_TabularValues):
    def __str__(self):
        return f"Vpi({self.v[:5]}...)"


VQPi = NewType("VQPi", tuple[Vpi, Qpi, Policy])
Sample = NewType("Sample", tuple[int, Vpi, Qpi, Policy | None])
Samples = NewType("Samples", list[Sample])
Transition = Callable[[Any, Any], tuple[tuple[Any, float], bool]]
EpisodeStep = NewType("EpisodeStep", tuple[int, int, float])


class TransitionException(Exception):
    pass


class PQueue:
    """Priority Queue"""

    def __init__(self, items: list[tuple[float, Any]]):
        self.items = items
        self._sort()

    def _sort(self):
        self.items.sort(key=lambda x: x[0])

    def push(self, item, priority):
        self.items.append((priority, item))
        self._sort()

    def pop(self):
        return self.items.pop(0)[1]

    def empty(self):
        return len(self.items) == 0


class RewardGeneartor(ABC):
    @classmethod
    @abstractmethod
    def generate(cls, *args, **kwargs):
        raise NotImplementedError


RandomDistributionT = Literal[
    "bernoulli",
    "gaussian",
    "uniform",
    "exponential",
    "poisson",
    "pareto",
    "triangular",
]


class RandomRewardGenerator:
    DISTRIBUTION: dict[RandomDistributionT, Callable] = {
        "bernoulli": np.random.binomial,
        "gaussian": np.random.normal,
        "uniform": np.random.uniform,
        "exponential": np.random.exponential,
        "poisson": np.random.poisson,
        "pareto": np.random.pareto,
        "triangular": np.random.triangular,
    }

    def __init__(self, distribution: RandomDistributionT, *d_args, **d_kwargs):
        self.gen_reward = partial(
            self.DISTRIBUTION[distribution],
            *d_args,
            **d_kwargs,
        )

    def __call__(self):
        return self.gen_reward()

    def mean(self, N: int = 1000) -> float:
        return np.mean([self.gen_reward() for i in range(N)])

    @classmethod
    def generate(
        cls,
        distribution: RandomDistributionT = "gaussian",
        *args: Any,
        **kwargs: Any,
    ) -> float:
        generator = cls.DISTRIBUTION.get(distribution)
        if not generator:
            raise ValueError(f"Invalid distribution: {distribution}")
        return generator(*args, **kwargs)


class UCTNode:
    def __init__(self, state, action, q, n, parent=None, is_terminal=False):
        self.state = state
        self.action = action
        self.q = q
        self.n = n
        self.parent = parent
        self.children: dict[Action, UCTNode] = {}
        self.is_terminal = False

    def add_child(self, child):
        self.children[child.action] = child
        return child


class UCTree:
    def __init__(self, root, Cp=1.0, max_steps=MAX_STEPS, nodes=None):
        if not isinstance(root, UCTNode):
            self.root = UCTNode(root, None, 0, 1, None)
        else:
            self.root = root
        self.Cp = Cp
        self.max_steps = max_steps
        self.nodes = {} if not nodes else nodes

    def max_depth(self):
        stack = [(self.root, 0)]
        max_depth = 0
        while stack:
            node, depth = stack.pop()
            max_depth = max(depth, max_depth)
            for child in node.children.values():
                stack.append((child, depth + 1))
        return max_depth

    def plot(self):
        max_depth = self.max_depth()
        width = 4 * max_depth
        height = max_depth
        stack: list[tuple[UCTNode, int, float, float]] = [(self.root, 0, 0, width)]
        treenodes = []
        lines = []
        while stack:
            node, depth, x, step = stack.pop()
            node_pos = (x + step / 2, height - depth)
            treenodes.append(node_pos)
            if node.children:
                n_childs = len(node.children)
                step = step / n_childs
                for i, child in enumerate(node.children.values()):
                    stack.append((child, depth + 1, x + i * step, step))
                    lines.append(
                        (node_pos, (step / 2 + x + i * step, height - depth - 1))
                    )

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.set_xticks([])  # type: ignore
        ax.set_yticks([])  # type: ignore
        for node in treenodes:
            ax.scatter(node[0], node[1], color="white", s=1)
        for line in lines:
            ax.plot(
                [line[0][0], line[1][0]],
                [line[0][1], line[1][1]],
                color="white",
                linewidth=0.5,
            )
        plt.show()


def _typecheck_tabular_idxs(*args):
    for arg in args:
        if not isinstance(arg, (Sequence, np.ndarray)):
            raise TypeError(f"Tabular Indexes must be Sequence, not {type(arg)}")


def _typecheck_transition(transition):
    if not isinstance(transition, Callable):
        raise TypeError(f"transition must be a Callable, not {type(transition)}")

    if transition.__code__.co_argcount != 2:
        raise TypeError(
            f"transition must have two positional arguments,"
            f" not {transition.__code__.co_argcount}"
        )


def _typecheck_constants(*args: int | float):
    for arg in args:
        if not isinstance(arg, (float, int)):
            raise TypeError(f"Constants must be float or int, not {type(arg)}")


def _typecheck_booleans(*args: bool):
    for arg in args:
        if not isinstance(arg, bool):
            raise TypeError(f"Booleans must be bool, not {type(arg)}")


def _typecheck_policies(*args):
    for arg in args:
        if not isinstance(arg, Policy):
            raise TypeError(f"Policies must be Policy, not {type(arg)}")


def _typecheck_all(
    tabular_idxs=None,
    transition=None,
    constants=None,
    booleans=None,
    policies=None,
):
    if tabular_idxs:
        _typecheck_tabular_idxs(*tabular_idxs)
    if transition:
        _typecheck_transition(transition)
    if constants:
        _typecheck_constants(*constants)
    if booleans:
        _typecheck_booleans(*booleans)
    if policies:
        _typecheck_policies(*policies)


def _get_sample_step(samples, n_episodes):
    if samples > n_episodes:
        samples = n_episodes
    if samples > 1e3:
        samples = int(1e3)
    sample_step = int(n_episodes / samples)
    return sample_step


def _check_ranges(values, ranges):
    for v, r in zip(values, ranges):
        if v < r[0] or v > r[1]:
            raise ValueError(f"{v} is out of range {r}")


def auto_cardinal(values, n, safe=True):
    if (n + 1) ** len(values) > 2.5e6:
        if safe:
            raise ValueError(
                "Too many combinations, may cause memory error,"
                "set safe=False to avoid raising this error"
            )
        else:
            warnings.warn("Too many combinations, may cause memory error")
    prod = np.array(np.meshgrid(*[values for _ in range(n)]))
    return prod.T.reshape(-1, n)


class BasisException(Exception):
    pass


def get_basis(
    basis: Literal["poly", "fourier"],
    cij,
) -> Callable[[np.ndarray], np.ndarray]:
    """get basis function for linear approximator using polynomial or
    fourier base

    Parameters
    ----------
    basis : str
        Basis function to use, either 'poly' or 'fourier'
    cij : np.ndarray
        Coefficients for the basis function

    Returns
    -------
    basis: Callable[[np.ndarray], np.ndarray]
        This function will not work on arbitrary defined states. Just on
        ones defined as sequences or numpy arrays. Any other type will
        raise an error.
    """
    _basis_function = None
    match basis:
        case "poly":

            def _basis_poly(s):
                xs = [np.prod(s**cj) for cj in cij]
                return np.array(xs)

            _basis_function = _basis_poly

        case "fourier":

            def _basis_fourier(s):
                xs = [np.cos(np.pi * np.dot(s, cj)) for cj in cij]
                return np.array(xs)

            _basis_function = _basis_fourier

    def basis_f(s):
        try:
            return _basis_function(s)
        except Exception:
            raise BasisException("State must be a sequence or numpy array")

    return basis_f
