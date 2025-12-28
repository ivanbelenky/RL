"""RL Copyright © 2023 Iván Belenky"""

from abc import ABC, abstractmethod

import numpy as np

from rl.solvers.model_based import policy_iteration, value_iteration, vq_pi_iter_naive
from rl.utils import Policy, RewardGenerator

PROB_TOL = 1e-3
ESTIMATE_ITERS = int(1e3)


class MarkovReward(ABC):
    @abstractmethod
    def generate(self, state: int, action: int) -> float:
        raise NotImplementedError

    @abstractmethod
    def r_sas(self, next_state: int) -> float:
        """
        r(s,a,s') = E[Rt|St-1 = s, At-1 = a, St = s']
        """

        raise NotImplementedError

    def r_sa(self, p_s: np.ndarray, state: int, action: int):
        """
        r(s,a) = E[Rt|St-1 = s, At-1 = a]
        """
        p = p_s[state][action]
        r = 0
        for i, ps in enumerate(p):
            r += ps * self.mean(state=self.states[i])
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
        self.states, self.actions = r_sa.shape
        self._r_sa = r_sa

    def generate(self, state: int = 0, action: int = 0) -> float:
        return self._r_sa[state][action]

    def r_sa(self, p_s: np.ndarray, state: int, action: int):
        return self._r_sa[state][action]

    def r_sas(self, next_state: int) -> float:
        return np.mean(self._r_sa[next_state])


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
        if not pi_sa and not (s or a):
            raise ValueError("Either pi_sa or s and a must be provided")

        if pi_sa:
            self.pi_sa = pi_sa
            sa: tuple[int, int] = self.pi_sa.shape
            self.s, self.a = sa
            self._validate_attr()
        else:
            self.s = s
            self.a = a
            # equal probable policy
            self.pi_sa = np.ones((self.s, self.a)) / self.a

    def _validate_attr(self):
        if not np.allclose(self.pi_sa.sum(axis=1), 1, atol=PROB_TOL):
            raise ValueError("Each row must sum to 1")

    def update_policy(self, q_pi: np.ndarray):
        """
        Updates the policy based on the Q function: for each state s
        the action a that maximizes Q(s,a) is selected. If there are
        multiple actions that maximize Q(s,a) then the policy is
        updated to be equally probable among those actions.
        """
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

    def __call__(self, state: int) -> np.ndarray:
        """
        Collapses the policy to a single action, i.e. a sample from the
        random variable that represents the policy.
        """
        return np.random.choice(self.pi_sa[state], p=self.pi_sa[state])


class MDP:
    VQ_PI_SOLVERS = {"iter_n": vq_pi_iter_naive}

    OPTIMAL_POLICY_SOLVERS = {
        "policy_iteration": policy_iteration,
        "value_iteration": value_iteration,
    }

    def __init__(
        self,
        p_s: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
        gamma: float = 0.9,
        policy: Policy | None = None,
        reward_gen: RewardGenerator | None = None,
    ):
        self.p_s = p_s
        self.states = states
        self.actions = actions
        self.gamma = gamma
        self.reward_gen = reward_gen
        self.history = []
        self._validate_attr()

        self.S = self.states.shape[0]
        self.A = self.actions.shape[0]
        self.policy = policy if policy else MarkovPolicy(s=self.S, a=self.A)

    @property
    def cum_return(self) -> float:
        return np.sum([r for _, r in self.history])

    @property
    def discounted_return(self) -> float:
        return np.sum([r * (self.gamma**i) for i, (_, r) in enumerate(self.history)])

    def _validate_attr(self):
        S = self.states.shape[0]
        A = self.actions.shape[0]
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
        return self.policy.pi_sa(state)

    def vq_pi(
        self, policy: MarkovPolicy | None = None, method: str = "iter_n"
    ) -> np.ndarray:
        """
        Individual state value functions and action-value functions
        vpi and qpi cannot be calculated for bigger problems. That
        constraint will give rise to parametrizations via DL.
        """
        policy = policy if policy else self.policy
        solver = self.VQ_PI_SOLVERS.get(method)
        if not solver:
            raise ValueError(f"Method {method} does not exist")

        return solver(self, policy)

    def optimize_policy(
        self,
        method: str = "policy_iteration",
        policy: MarkovPolicy | None = None,
    ) -> MarkovPolicy:
        """
        Optimal policy is the policy that maximizes the expected
        discounted return. It is the policy that maximizes the
        value function for each possible state.
        """
        policy = policy if policy else self.policy
        solver = self.OPTIMAL_POLICY_SOLVERS.get(method)
        if not solver:
            raise ValueError(f"Method {method} does not exist")

        return solver(self, policy)

    def __call__(self, state: int = 0) -> tuple[int, float]:
        p = self.p_s[state][self.policy(state)]
        next_state = np.random.choice(self.states, p=p)
        self.curr_state = next_state
        reward = self.reward_gen.generate(next_state)

        self.history.append((self.curr_state, reward))

        return next_state, reward
