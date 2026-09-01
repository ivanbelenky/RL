import copy
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Any, Self, cast, override

import numpy as np

from rl.utils import (
    MAX_STEPS,
    W_INIT,
    EpisodeStep,
    Policy,
    Transition,
    TransitionException,
)

"""
All of this may change if the policy gradient methods are
similar to this implementation.

SGD and Semi Gradient Linear methods:

All Linear methods of this methods involve using a real value
weight matrix/vector that will be used in conjunction with a
basis function to approximate the value function.

wt+1 = wt - 1/2 * alpha * d[(v_pi - v_pi_hat)^2]/dw
wt+1 = wt + alpha * (v_pi - v_pi_hat)]*d[v_pi_hat]/dw
wt+1 = wt + alpha * (U - v_pi_hat)]*d[v_pi_hat]/dw

Since we dont have v_pi we have to use some estimator:
- MC would imply grabbing full trajectories and using them
- TD since involves bootstraping (it will be a semigradient method).

Therefore we generate the most abstract SGD method. The two most 
important parts of this methods is the U approximation to the real
value, and the value function approximator, this class should be
differentiable, or hold a gradient method.
"""


class Approximator[InputT](ABC):
    """Approximator base class that implements caster methods
    as well as defining the basic interface of any approximator.
    It has to be updateable and callable. Updatability implies
    that it can change its inner attributes and hopefully learn.
    """

    @abstractmethod
    def __call__(self, x: InputT, /) -> float:
        """Return the value of the approximation"""
        raise NotImplementedError

    @abstractmethod
    def update(self, *args, **kwargs) -> None | np.ndarray:
        """Update the approximator"""
        raise NotImplementedError

    def copy(self, *args: Any, **kwargs: Any) -> Self:
        """Return a copy of the approximator"""
        return copy.deepcopy(self)

    def is_differentiable(self) -> bool:
        grad = getattr(self, "grad", None)
        if grad:
            return True
        return False


class DifferentiableApproximator[InputT](Approximator[InputT]):
    @abstractmethod
    def grad(self, *args, **kwargs):
        raise NotImplementedError

    @property
    @abstractmethod
    def w(self) -> Any:
        raise NotImplementedError

    @w.setter
    def w(self, new_val):
        raise NotImplementedError


class ModelFreeTLPolicy[StateT, ActionT](Policy[ActionT]):
    """ModelFreeTLPolicy is for approximated methods what
    ModelFreePolicy is for tabular methods.

    This policies are thought with tabular actions in mind, since
    the problem of continuous action spaces are a topic of ongoing
    research and not yet standardized. For each a in the action-space A
    there will exist an approximator.
    """

    def __init__(
        self,
        actions: Sequence[ActionT],
        q_hat: Approximator[tuple[StateT, ActionT]],
    ):
        self.actions = actions
        self.A = len(actions)
        self.q_hat = q_hat

    def update_policy(self, *args, **kwargs):
        self.q_hat.update(*args, **kwargs)

    @override
    def __call__(self, state: StateT, /) -> ActionT:
        action_idx = cast(
            int, np.argmax([self.q_hat((state, a)) for a in self.actions])
        )
        return self.actions[action_idx]


class EpsSoftSALPolicy[StateT, ActionT](ModelFreeTLPolicy[StateT, ActionT]):
    def __init__(
        self,
        actions: Sequence[ActionT],
        q_hat: Approximator[tuple[StateT, ActionT]],
        eps: float = 0.1,
    ):
        super().__init__(actions, q_hat)
        self.eps = eps

    def __call__(self, state: StateT) -> ActionT:
        if np.random.rand() < self.eps:
            return self.actions[np.random.randint(self.A)]
        return super().__call__(state)


class REINFORCEPolicy[StateT, ActionT](ModelFreeTLPolicy[StateT, ActionT]):
    def __init__(
        self,
        actions: Sequence[ActionT],
        pi_hat: DifferentiableApproximator[tuple[StateT, ActionT]],
    ):
        """Must be a differential approximator"""
        self.actions = actions
        self.pi_hat = pi_hat
        if not isinstance(self.pi_hat, DifferentiableApproximator):
            raise TypeError("Policy approximator pi_hat must be differentiable")

    def grad_lnpi(self, s: StateT, a: ActionT) -> np.ndarray:
        pi_sa = self.pi_sa(s).reshape(-1, 1)
        grad_pi_sa = self.pi_hat.grad((s, a)).reshape(-1, 1)
        grads_pi_sa = np.array([self.pi_hat.grad((s, a_i)) for a_i in self.actions])
        return (grad_pi_sa - grads_pi_sa @ pi_sa).reshape(-1)

    def update_policy(self, c: float, s: StateT, a: ActionT) -> None:
        self.pi_hat.w += c * self.grad_lnpi(s, a)

    def pi_sa(self, s: StateT) -> np.ndarray:
        pi_hat_sa = [self.pi_hat((s, a)) for a in self.actions]
        max_sa = max(pi_hat_sa)
        e_hsa = [np.exp(pi_hat_sa[i] - max_sa) for i in range(len(self.actions))]
        denom = sum(e_hsa)
        pi_sa = np.array([e_hsa[i] / denom for i in range(len(self.actions))])
        return pi_sa

    def __call__(self, s: StateT, /) -> ActionT:
        """default softmax implementation"""
        action_idx = int(np.random.choice(len(self.actions), p=self.pi_sa(s)))
        return self.actions[action_idx]


class ModelFreeTL[StateT, ActionT]:
    """
    ModelFreeTL stands for Model Free Tabular Less, even if we have state,
    to approximate methods what ModelFree is to tabular ones.

    ModelFreeTL is used mostly internally for the seek of readability
    on solvers, but can be used standalone as well. The usual case
    for this is when you want to generate arbitrary episodes for a
    specific environment. This class will stand in between of the
    user implemented transitions and the solvers. In difference with
    tabular ModelFree there is no room for validation previous to
    runtime executions.
    """

    def __init__(
        self,
        transition: Transition[StateT, ActionT],
        rand_state: Callable[[], StateT],
        policy: ModelFreeTLPolicy[StateT, ActionT],
        gamma: float = 1,
    ):
        self.policy = policy
        self.rand_state = rand_state
        self.transition = transition
        self.gamma = gamma
        self._validate_transition()

    def _validate_transition(self) -> None:
        start = perf_counter()
        while perf_counter() - start < 2:
            rand_s = self.rand_state()
            rand_a = self.policy.actions[np.random.randint(self.policy.A)]
            try:
                self.transition(rand_s, rand_a)
            except Exception as e:
                raise TransitionException(f"Transition function is not valid: {e}")

    def random_sa(self) -> tuple[StateT, ActionT]:
        a = self.policy.actions[np.random.randint(self.policy.A)]
        s = self.rand_state()
        return s, a

    def generate_episode(
        self,
        s_0: StateT,
        a_0: ActionT,
        policy: ModelFreeTLPolicy[StateT, ActionT] | None = None,
        max_steps: int = MAX_STEPS,
    ) -> list[EpisodeStep[StateT, ActionT]]:
        """Generate an episode using given policy if any, otherwise
        use the one defined as the attribute"""
        policy = policy if policy else self.policy
        episode: list[EpisodeStep[StateT, ActionT]] = []
        end = False
        step = 0
        s_t_1, a_t_1 = s_0, a_0
        while (not end) and (step < max_steps):
            (s_t, r_t), end = self.transition(s_t_1, a_t_1)
            episode.append((s_t_1, a_t_1, r_t))
            a_t = policy(s_t)
            s_t_1, a_t_1 = s_t, a_t
            step += 1

        return episode

    def step_transition(
        self, state: StateT, action: ActionT
    ) -> tuple[tuple[StateT, float], bool]:
        return self.transition(state, action)


class SGDWA[InputT](DifferentiableApproximator[InputT]):
    """Stochastic Gradient Descent Weight-Vector Approximator
    for MSVE (mean square value error).

    Differentiable Value Function approximator dependent
    on a weight vector. Must define a gradient method. Thought
    to be less of a general case and more oriented toward the
    mean square value error VE, the prediction objective.
    """

    def __init__(self, fs: int, basis: Callable[[InputT], np.ndarray] | None = None):
        """
        Parameters
        ----------
        fs: int
            feature shape, i.e. dimensionality of the function basis
        basis: Callable[[Any], np.ndarray], optional
            function basis defaults to identity. If not specified the
            signature must be Callable[[np.ndarray], np.ndarray] otherwise
            it will be probably fail miserably.
        """
        self.fs = fs

        self.basis_name = (
            "identity" if not basis else getattr(basis, "__name__", repr(basis))
        )
        self.basis = basis if basis else lambda x: x
        self._w = np.ones(self.fs) * W_INIT

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, new_w):
        self._w = new_w
        return self._w

    def grad(self, x: InputT) -> np.ndarray:
        """Return the gradient of the approximation"""
        return self.basis(x)

    def delta_w(self, U: float, alpha: float, x: InputT, g: np.ndarray) -> np.ndarray:
        """g: vector value, either gradient or elegibility trace"""
        return alpha * (U - self(x)) * g

    def et_update(self, U: float, alpha: float, x: InputT, z: np.ndarray) -> np.ndarray:
        """Updates inplace with elegibility traces the weight vector"""
        dw = self.delta_w(U, alpha, x, z)
        self.w = self.w + dw
        return dw

    def update(self, U: float, alpha: float, x: InputT) -> np.ndarray:
        """Updates inplace the weight vector and returns update just in case"""
        dw = self.delta_w(U, alpha, x, self.grad(x))
        self.w = self.w + dw
        return dw

    def __call__(self, x: InputT, /) -> float:
        return np.dot(self.w, self.basis(x))


LinearApproximator = SGDWA
