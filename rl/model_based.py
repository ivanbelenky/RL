"""RL Copyright © 2023 Iván Belenky"""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from rl.solvers.model_based import policy_iteration, value_iteration, vq_pi_iter_naive
from rl.types import SizedIterable
from rl.utils import (
    Action,
    Policy,
    RandomRewardGenerator,
    State,
    StateAction,
    VQPi,
    _TabularValues,
)

PROB_TOL = 1e-3
ESTIMATE_ITERS = int(1e3)


class MarkovReward[S: int, A: int](ABC):
    @property
    @abstractmethod
    def states(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def generate(self, state: int, action: int) -> float:
        raise NotImplementedError

    @abstractmethod
    def r_sas(self, next_state: int) -> float:
        """
        r(s,a,s') = E[Rt|St-1 = s, At-1 = a, St = s']
        """
        raise NotImplementedError

    def r_sa(self, p_s: NDArray, state: int, action: int) -> float:
        """
        r(s,a) = E[Rt|St-1 = s, At-1 = a]
        """
        p = p_s[state][action]
        r = 0
        for i, ps in enumerate(p):
            r += ps * np.mean([self.r_sas(next_state=s) for s in range(self.states)])
        return r


class TabularReward(MarkovReward):
    """
    Tabular reward implements as the name suggests a reward
    per state and action. The reward is a matrix of size SxA.
    This type of reward is used in the case that the world
    in which the agent conducts gives you fixed rewards for
    taking action: a at state: s.
    """

    def __init__(
        self,
        r_sa: np.ndarray,
    ):
        self._states, self._actions = r_sa.shape
        self._r_sa = r_sa

    @property
    def states(self):
        return self._states

    def generate(self, state: int = 0, action: int = 0) -> float:
        return self._r_sa[state][action]

    def r_sa(self, p_s: np.ndarray, state: int, action: int) -> float:
        return self._r_sa[state][action]

    def r_sas(self, next_state: int) -> float:
        return np.mean(self._r_sa[next_state])


class RandomMarkovReward(MarkovReward):
    def __init__(self, reward_generator: RandomRewardGenerator | None):
        self._reward_gen = reward_generator or RandomRewardGenerator("gaussian", 0, 1)

    def generate(self, state: int = 0, action: int = 0) -> float:
        return self._reward_gen()

    def r_sa(self, p_s: np.ndarray, state: int, action: int) -> float:
        return self._reward_gen()

    def r_sas(self, next_state: int) -> float:
        return self._reward_gen()


class MarkovPolicy(Policy):
    """
    Markov Policy is a policy that is defined by a matrix of size SxA.
    This class admits a policy defined by the user or a equally probable
    policy will be created.

    The policy matrix π(a|s) must be a matrix of size SxA where each row
    represents the probability of taking action a at state s. Therefore
    each row must sum to 1 within the specified tolerance 1E-3.
    """

    def __init__(
        self,
        pi_sa: np.ndarray | None = None,
        s: int | None = None,
        a: int | None = None,
    ):
        """
        pi_sa: policy matrix
        s: number of states
        a: number of actions

        pi_sa and s and a are mutually exclusive. If pi_sa is provided then
        s and a are ignored. If pi_sa is not provided then s and a must be
        provided.
        """
        if pi_sa:
            self.pi_sa = pi_sa
            sa: tuple[int, int] = self.pi_sa.shape
            self.s, self.a = sa
            self._validate_attr()
        elif s is not None and a is not None:
            self.s = s
            self.a = a
            # equal probable policy
            self.pi_sa = np.ones((self.s, self.a)) / self.a
        else:
            raise ValueError("Either pi_sa or s and a must be provided")

    def _validate_attr(self):
        if not np.allclose(self.pi_sa.sum(axis=1), 1, atol=PROB_TOL):
            raise ValueError("Each row must sum to 1")

    def update_policy(self, q_pi: np.ndarray | _TabularValues):
        """
        Updates the policy based on the Q function: for each state s
        the action a that maximizes Q(s,a) is selected. If there are
        multiple actions that maximize Q(s,a) then the policy is
        updated to be equally probable among those actions.
        """
        q_pi = q_pi if isinstance(q_pi, np.ndarray) else q_pi.v
        self.pi_sa = np.array([self._update_policy(q_pi, s) for s in range(self.s)])

    def _update_policy(self, q_pi: np.ndarray, state: int) -> np.ndarray:
        q_sa = q_pi.T[state]
        max_q = max(q_sa)
        max_q_sa = np.array([q_sa[a] == max_q for a in range(self.a)])
        return max_q_sa / sum(max_q_sa)

    def π(self, state: int):
        """
        π(a|s=state)
        """
        return self.pi_sa[state]

    def __call__(self, state: int) -> int:
        """
        Collapses the policy to a single action, i.e. a sample from the
        random variable that represents the policy.
        """
        return np.random.choice(self.pi_sa[state], p=self.pi_sa[state])


class MDP[S: int, A: int]:
    VQ_PI_SOLVERS = {"iter_n": vq_pi_iter_naive}

    OPTIMAL_POLICY_SOLVERS = {
        "policy_iteration": policy_iteration,
        "value_iteration": value_iteration,
    }

    SAS = tuple[S, A, S]

    def __init__(
        self,
        p_s: np.ndarray[SAS],
        states: SizedIterable[S],
        actions: SizedIterable[A],
        reward_gen: MarkovReward,
        gamma: float = 0.9,
        policy: MarkovPolicy | None = None,
    ):
        self.p_s: np.ndarray[self.SAS] = p_s  # transition function
        self.gamma: int | float = gamma
        self.reward_gen: MarkovReward = reward_gen

        self.states = State(states)
        self.actions = Action(actions)
        self.stateaction = StateAction([(s, a) for s in states for a in actions])

        self.policy: MarkovPolicy = policy or MarkovPolicy(
            s=self.states.N,
            a=self.actions.N,
        )
        self._validate_attr()

    def _validate_attr(self):
        S = self.states.N
        A = self.actions.N
        if self.p_s.shape != (S, A, S):
            raise ValueError(
                "p_s must be of shape "
                + f"(n_states, n_actions, n_states) = ({S}, {A}, {S})"
            )

        for i in range(S):
            if not np.allclose(self.p_s[i].sum(axis=1), 1, atol=PROB_TOL):
                raise ValueError("Each row must sum to 1")

        if self.gamma > 1 or self.gamma < 0:
            raise ValueError("discounted rate gamma has to be in range [0, 1]")

    def r_sa(self, state: int, action: int) -> float:
        return self.reward_gen.r_sa(self.p_s, state, action)

    def r_sas(self, next_s: int) -> float:
        return self.reward_gen.r_sas(next_s)

    def pi_sa(self, state: int) -> np.ndarray:
        return self.policy.pi_sa[state]

    def vq_pi(
        self,
        method: Literal["iter_n"] = "iter_n",
    ) -> VQPi:
        """
        Individual state value functions and action-value functions
        vpi and qpi cannot be calculated for bigger problems. That
        constraint will give rise to parametrizations via DL.
        """
        solver = self.VQ_PI_SOLVERS.get(method)
        if not solver:
            raise ValueError(f"Method {method} does not exist")

        return solver(self, self.policy)

    def optimize_policy(
        self,
        method: Literal["policy_iteration", "value_iteration"] = "policy_iteration",
    ) -> MarkovPolicy:
        """
        Optimal policy is the policy that maximizes the expected
        discounted return. It is the policy that maximizes the
        value function for each possible state.
        """
        solver = self.OPTIMAL_POLICY_SOLVERS.get(method)

        if not solver:
            raise ValueError(f"Method {method} does not exist")

        return solver(self, self.policy)
