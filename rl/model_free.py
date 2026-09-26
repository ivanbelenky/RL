"""
RL - Copyright © 2023 Iván Belenky @Leculette
"""

from collections.abc import Hashable, Sized
from typing import Literal, overload

import numpy as np

from rl.types import SizedIterable
from rl.utils import (
    MAX_STEPS,
    Action,
    EpisodeStep,
    Policy,
    State,
    StateAction,
    Transition,
    TransitionException,
)


class ModelFreePolicy(Policy[int]):
    def __init__(self, A: Sized | int, S: Sized | int):
        if not isinstance(A, int):
            A = len(A)
        if not isinstance(S, int):
            S = len(S)
        self.A = A
        self.S = S
        self.pi = np.ones((S, A)) / A

    def __call__(self, state: int) -> int:
        return np.random.choice(self.A, p=self.pi[state])

    def pi_as(self, action: int, state: int) -> float:
        return self.pi[state, action]

    def update_policy(self, q: np.ndarray, s: int) -> None:
        qs_mask = q[s] == np.max(q[s])
        self.pi[s] = np.where(qs_mask, 1.0 / qs_mask.sum(), 0)

    def _make_deterministic(self) -> None:
        self.pi = np.eye(self.A)[np.argmax(self.pi, axis=1)]


class EpsilonSoftPolicy(ModelFreePolicy):
    def __init__(self, A: Sized | int, S: Sized | int, eps: float):
        super().__init__(A, S)
        self.Ɛ = eps

    def update_policy(self, q: np.ndarray, s: int) -> None:
        # if there are multiple actions with the same value,
        # then we choose one of them randomly
        max_q = np.max(q[s])
        qs_mask = q[s] == max_q
        self.pi[s] = self.Ɛ / self.A
        self.pi[s, qs_mask] += (1 - self.Ɛ) / qs_mask.sum()


class ModelFree[StateT: Hashable, ActionT: Hashable]:
    """
    ModelFree is the base holder of the states, actions, and
    the transition defining an environment.

    ModelFree is used mostly internally for the seek of readability
    on solvers, but can be used standalone as well. The usual case
    for this is when you want to generate arbitrary episodes of a
    specific environment. This class will stand in between of the
    user implemented transitions and validate its correct behavior.
    """

    def __init__(
        self,
        states: SizedIterable[StateT],
        actions: SizedIterable[ActionT],
        transition: Transition[StateT, ActionT],
        gamma: float = 1,
        policy: ModelFreePolicy | None = None,
    ):
        self.states: State[StateT] = State(states)
        self.actions: Action[ActionT] = Action(actions)
        self.stateaction: StateAction[StateT, ActionT] = StateAction(
            [(s, a) for s in states for a in actions]
        )
        self.transition = transition
        self.gamma = gamma
        self.policy = policy or ModelFreePolicy(self.actions.N, self.states.N)

        self._validate_transition()

    def init_vq(self) -> tuple[np.ndarray, np.ndarray]:
        v = np.zeros(self.states.N)
        q = np.zeros((self.states.N, self.actions.N))
        return v, q

    @overload
    def random_sa(self, value: Literal[False] = False) -> tuple[int, int]: ...

    @overload
    def random_sa(self, value: Literal[True]) -> tuple[StateT, ActionT]: ...

    @overload
    def random_sa(self, value: bool) -> tuple[int, int] | tuple[StateT, ActionT]: ...

    def random_sa(
        self, value: bool = False
    ) -> tuple[int, int] | tuple[StateT, ActionT]:
        if value:
            return self.states.random(True), self.actions.random(True)
        return self.states.random(), self.actions.random()

    def _to_index(self, state: StateT, action: ActionT) -> tuple[int, int]:
        state_idx = self.states.get_index(state)
        action_idx = self.actions.get_index(action)
        return state_idx, action_idx

    def _validate_transition(self) -> None:
        states = self.states.seq
        actions = self.actions.seq
        sa = [(s, a) for s in states for a in actions]

        success, fail_count = True, 0
        for s, a in sa:
            try:
                self.__validate_transition(s, a)
            except Exception as e:
                success = False
                fail_count += 1
                print(f"Warning: {e}")  # TODO: change to logger

        if not success:
            raise TransitionException(
                f"Transition failed for {fail_count} state-action pairs"
            )

    def __validate_transition(
        self,
        state: StateT,
        action: ActionT,
    ) -> tuple[tuple[StateT, float], bool]:
        try:
            (s, r), end = self.transition(state, action)
        except Exception as e:
            raise TransitionException(f"Transition method failed: {e}")

        if not isinstance(end, bool) or not isinstance(r, (float, int)):
            raise TransitionException(
                "Transition method must return (Any, float), bool"
                f" instead of ({type(s)}, {type(r)}), {type(end)}"
            )
        try:
            self.states.get_index(s)
            self.states.get_index(state)
            self.actions.get_index(action)
        except Exception as e:
            raise TransitionException(
                f"Undeclared state or action in transition method: {e}"
            )

        return (s, r), end

    def generate_episode(
        self,
        s_0: StateT,
        a_0: ActionT | None = None,
        policy: Policy[int] | None = None,
        max_steps: int = MAX_STEPS,
    ) -> list[EpisodeStep[int, int]]:
        policy = policy or self.policy
        episode: list[EpisodeStep[int, int]] = []
        end = False
        step = 0
        s_t_1 = s_0
        if a_0 is None:
            a_t_1 = self.actions.from_index(policy(self.states.get_index(s_0)))
        else:
            a_t_1 = a_0
        while (not end) and (step < max_steps):
            (s_t, r_t), end = self.transition(s_t_1, a_t_1)
            (_s, _a), _r = self._to_index(s_t_1, a_t_1), r_t
            episode.append((_s, _a, _r))
            if end:
                break
            a_t = policy(self.states.get_index(s_t))
            s_t_1, a_t_1 = s_t, self.actions.from_index(a_t)

            step += 1

        return episode

    def step_transition(
        self, state: int, action: int
    ) -> tuple[tuple[int, float], bool]:
        s, a = self.states.from_index(state), self.actions.from_index(action)
        (s_t, r_t), end = self.transition(s, a)
        s_new = self.states.get_index(s_t)
        return (s_new, r_t), end
