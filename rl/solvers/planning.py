from collections.abc import Callable, Hashable, Sequence
from typing import cast

import numpy as np
from tqdm import tqdm

from rl.model_free import ModelFree, ModelFreePolicy
from rl.solvers.model_free import (
    _set_policy,
    _set_s0_a0,
    get_sample,
)
from rl.types import SizedIterable
from rl.utils import (
    MAX_ITER,
    MAX_STEPS,
    PQueue,
    Qpi,
    Sample,
    Samples,
    Transition,
    UCTNode,
    UCTree,
    Vpi,
    VQPi,
    _get_sample_step,
    _typecheck_all,
)


def dynaq[StateT: Hashable, ActionT: Hashable](
    states: SizedIterable[StateT],
    actions: SizedIterable[ActionT],
    transition: Transition[StateT, ActionT],
    state_0: StateT | None = None,
    action_0: ActionT | None = None,
    gamma: float = 1.0,
    kappa: float = 0.01,
    n: int = 1,
    plus: bool = False,
    alpha: float = 0.05,
    n_episodes: int = MAX_ITER,
    policy: ModelFreePolicy | None = None,
    eps: float | None = None,
    samples: int = 1000,
    max_steps: int = MAX_STEPS,
) -> tuple[
    VQPi[StateT, ActionT, ModelFreePolicy],
    Samples[StateT, ActionT, ModelFreePolicy],
]:
    """
    TODO: docs
    """
    policy = _set_policy(policy, eps, actions, states)

    _typecheck_all(
        tabular_idxs=[states, actions],
        transition=transition,
        constants=[gamma, kappa, n, alpha, n_episodes, samples, max_steps],
        booleans=[plus],
        policies=[policy],
    )

    # check ranges

    sample_step = _get_sample_step(samples, n_episodes)

    model = ModelFree(states, actions, transition, gamma=gamma, policy=policy)
    v, q, final_samples = _dyna_q(
        model,
        state_0,
        action_0,
        n,
        alpha,
        kappa,
        plus,
        int(n_episodes),
        max_steps,
        sample_step,
    )

    return VQPi(v, q, policy), final_samples


def _dyna_q[StateT: Hashable, ActionT: Hashable](
    MF: ModelFree[StateT, ActionT],
    s_0: StateT | None,
    a_0: ActionT | None,
    n: int,
    alpha: float,
    kappa: float,
    plus: bool,
    n_episodes: int,
    max_steps: int,
    sample_step: int,
) -> tuple[
    Vpi[StateT],
    Qpi[tuple[StateT, ActionT]],
    Samples[StateT, ActionT, ModelFreePolicy],
]:
    π, α, γ, κ = MF.policy, alpha, MF.gamma, kappa

    v, q = MF.init_vq()

    S, A = MF.states.N, MF.actions.N
    model_sas = np.zeros((S, A), dtype=int)
    model_sar = np.zeros((S, A), dtype=float)
    times_sa = np.zeros((S, A), dtype=int)
    visited_sa = set()

    samples = []
    current_t = 0
    for n_episode in tqdm(range(n_episodes), desc="Dyna-Q", unit="episodes"):
        s_0, _ = _set_s0_a0(MF, s_0, None)

        s = MF.states.get_index(s_0)
        T = int(max_steps)

        for t in range(T):
            a = π(s)
            (s_, r), end = MF.step_transition(s, a)  # real next state
            q[s, a] = q[s, a] + α * (r + γ * np.max(q[s_]) - q[s, a])

            times_sa[s, a] = current_t

            # assuming deterministic environment
            model_sas[s, a] = s_
            model_sar[s, a] = r

            visited_sa.add((s, a))

            current_t += 1

            for _ in range(n):
                if len(visited_sa) == 0:
                    break
                rs, ra = list(visited_sa)[np.random.randint(len(visited_sa))]
                s_m = model_sas[rs, ra]  # model next state
                r_ = model_sar[rs, ra]
                R = r_
                if plus:
                    tau = current_t - times_sa[rs, ra]
                    R = R + κ * np.sqrt(tau)
                q[rs, ra] = q[rs, ra] + α * (R + γ * np.max(q[s_m]) - q[rs, ra])
                π.update_policy(q, rs)

            π.update_policy(q, s_)
            s = s_  # current state equal next state
            if end:
                break

        if n_episode % sample_step == 0:
            samples.append(Sample(*get_sample(MF, v, q, π, n_episode, True)))

    return Vpi(v, MF.states), Qpi(q, MF.stateaction), Samples(samples)


def priosweep[StateT: Hashable, ActionT: Hashable](
    states: SizedIterable[StateT],
    actions: SizedIterable[ActionT],
    transition: Transition[StateT, ActionT],
    state_0: StateT | None = None,
    action_0: ActionT | None = None,
    gamma: float = 1.0,
    theta: float = 0.01,
    n: int = 1,
    plus: bool = False,
    alpha: float = 0.05,
    n_episodes: int = MAX_ITER,
    policy: ModelFreePolicy | None = None,
    eps: float | None = None,
    samples: int = 1000,
    max_steps: int = MAX_STEPS,
) -> tuple[
    VQPi[StateT, ActionT, ModelFreePolicy],
    Samples[StateT, ActionT, ModelFreePolicy],
]:
    """
    TODO: docs
    """
    policy = _set_policy(policy, eps, actions, states)

    _typecheck_all(
        tabular_idxs=[states, actions],
        transition=transition,
        constants=[gamma, theta, n, alpha, n_episodes, samples, max_steps],
        booleans=[plus],
        policies=[policy],
    )

    # check ranges

    sample_step = _get_sample_step(samples, n_episodes)

    model = ModelFree(states, actions, transition, gamma=gamma, policy=policy)
    v, q, final_samples = _priosweep(
        model,
        state_0,
        action_0,
        n,
        alpha,
        theta,
        int(n_episodes),
        max_steps,
        sample_step,
    )

    return VQPi(v, q, policy), final_samples


def _priosweep[StateT: Hashable, ActionT: Hashable](
    MF: ModelFree[StateT, ActionT],
    s_0: StateT | None,
    a_0: ActionT | None,
    n: int,
    alpha: float,
    theta: float,
    n_episodes: int,
    max_steps: int,
    sample_step: int,
) -> tuple[
    Vpi[StateT],
    Qpi[tuple[StateT, ActionT]],
    Samples[StateT, ActionT, ModelFreePolicy],
]:
    π, α, γ = MF.policy, alpha, MF.gamma
    v, q = MF.init_vq()

    P, Pq, θ = 0, PQueue[tuple[int, int]]([]), theta

    S, A = MF.states.N, MF.actions.N
    model_sas = np.zeros((S, A), dtype=int)
    model_sar = np.zeros((S, A), dtype=float)
    times_sa = np.zeros((S, A), dtype=int)

    samples, current_t = [], 0
    for n_episode in tqdm(range(n_episodes), desc="priosweep", unit="episodes"):
        s_0, _ = _set_s0_a0(MF, s_0, None)

        s = MF.states.get_index(s_0)
        T = int(max_steps)

        for t in range(T):
            a = π(s)
            (s_, r), end = MF.step_transition(s, a)  # real next state
            times_sa[s, a] = current_t
            model_sas[s, a] = s_
            model_sar[s, a] = r

            P = np.abs(r + γ * np.max(q[s_]) - q[s, a])
            if P > θ:
                Pq.push((s, a), P)

            current_t += 1

            for _ in range(n):
                if Pq.empty():
                    break

                ps, pa = Pq.pop()
                s_m = model_sas[ps, pa]  # model next state
                r_ = model_sar[ps, pa]
                R = r_

                q[ps, pa] = q[ps, pa] + α * (R + γ * np.max(q[s_m]) - q[ps, pa])

                # grab all the index where model_sas == s
                mmask = model_sas == s
                for ss, aa in zip(*np.where(mmask)):
                    rr = model_sar[ss, aa]
                    P = np.abs(rr + γ * np.max(q[s]) - q[ss, aa])
                    if P > θ:
                        Pq.push((s, a), P)

            π.update_policy(q, s_)
            s = s_  # current state equal next state
            if end:
                break

        if n_episode % sample_step == 0:
            samples.append(Sample(*get_sample(MF, v, q, π, n_episode, True)))

    return Vpi(v, MF.states), Qpi(q, MF.stateaction), Samples(samples)


def t_sampling[StateT: Hashable, ActionT: Hashable](
    states: SizedIterable[StateT],
    actions: SizedIterable[ActionT],
    transition: Transition[StateT, ActionT],
    state_0: StateT | None = None,
    action_0: ActionT | None = None,
    gamma: float = 1.0,
    n_episodes: int = MAX_ITER,
    policy: ModelFreePolicy | None = None,
    eps: float | None = None,
    samples: int = 1000,
    optimize: bool = False,
    max_steps: int = MAX_STEPS,
) -> tuple[
    VQPi[StateT, ActionT, ModelFreePolicy],
    Samples[StateT, ActionT, ModelFreePolicy],
]:
    """
    TODO: docs
    """
    policy = _set_policy(policy, eps, actions, states)

    _typecheck_all(
        tabular_idxs=[states, actions],
        transition=transition,
        constants=[gamma, n_episodes, samples, max_steps],
        booleans=[optimize],
        policies=[policy],
    )

    # TODO: check ranges

    sample_step = _get_sample_step(samples, n_episodes)

    model = ModelFree(states, actions, transition, gamma=gamma, policy=policy)
    v, q, final_samples = _t_sampling(
        model, state_0, action_0, int(n_episodes), optimize, max_steps, sample_step
    )

    return VQPi(v, q, policy), final_samples


def _t_sampling[StateT: Hashable, ActionT: Hashable](
    MF: ModelFree[StateT, ActionT],
    s_0: StateT | None,
    a_0: ActionT | None,
    n_episodes: int,
    optimize: bool,
    max_steps: int,
    sample_step: int,
) -> tuple[
    Vpi[StateT],
    Qpi[tuple[StateT, ActionT]],
    Samples[StateT, ActionT, ModelFreePolicy],
]:
    π, γ = MF.policy, MF.gamma
    v, q = MF.init_vq()

    S, A = MF.states.N, MF.actions.N
    n_sas = np.zeros((S, A, S), dtype=int)  # p(s'|s,a)
    model_sar = np.zeros((S, A, S), dtype=float)  # r(s,a,s') deterministic reward

    samples = []
    for n_episode in tqdm(
        range(n_episodes), desc="Trajectory Sampling", unit="episodes"
    ):
        s, a = _set_s0_a0(MF, s_0, a_0)
        a_ = MF.actions.get_index(a)
        s = MF.states.get_index(s)

        for _ in range(int(max_steps)):
            (s_, r), end = MF.step_transition(s, a_)  # real next state

            n_sas[s, a_, s_] += 1
            model_sar[s, a_, s_] = r  # assumes deterministic reward

            # p_sas is the probability of transitioning from s to s'
            p_sas = n_sas[s, a_] / np.sum(n_sas[s, a_])
            next_s_mask = np.where(p_sas)[0]
            max_q = np.max(q[next_s_mask, :], axis=1)
            r_ns = model_sar[s, a_, next_s_mask]
            p_ns = p_sas[next_s_mask]

            q[s, a_] = np.dot(p_ns, r_ns + γ * max_q)

            π.update_policy(q, s)
            a_ = π(s_)
            s = s_

            if end:
                break

        if n_episode % sample_step == 0:
            samples.append(Sample(*get_sample(MF, v, q, π, n_episode, optimize)))

    return Vpi(v, MF.states), Qpi(q, MF.stateaction), Samples(samples)


def rtdp():
    raise NotImplementedError


def _best_child[StateT, ActionT: Hashable](
    v: UCTNode[StateT, ActionT], Cp: float
) -> UCTNode[StateT, ActionT]:
    actions = list(v.children)
    qs = np.array([v.children[a].q for a in actions])
    ns = np.array([v.children[a].n for a in actions])
    ucb = qs / ns + Cp * np.sqrt(np.log(v.n) / ns)
    return v.children[actions[int(np.argmax(ucb))]]


def _expand[StateT, ActionT: Hashable](
    v: UCTNode[StateT, ActionT],
    transition: Transition[StateT, ActionT],
    actions: Sequence[ActionT],
) -> UCTNode[StateT, ActionT]:
    a = actions[np.random.randint(len(actions))]
    (s_, _), end = transition(v.state, a)
    v_prime = UCTNode(s_, a, 0, 1, v, end)
    v.children[a] = v_prime
    return v_prime


def _tree_policy[StateT, ActionT: Hashable](
    tree: UCTree[StateT, ActionT],
    Cp: float,
    transition: Transition[StateT, ActionT],
    action_map: Callable[[StateT], Sequence[ActionT]],
    eps: float,
) -> UCTNode[StateT, ActionT]:
    v = tree.root
    while not v.is_terminal:
        actions = action_map(v.state)
        took_actions = v.children.keys()
        unexplored = set(actions) - set(took_actions)
        if not took_actions:
            return _expand(v, transition, actions)
        if unexplored and np.random.rand() < eps:
            return _expand(v, transition, tuple(unexplored))
        v = _best_child(v, Cp)
    return v


def _default_policy[StateT, ActionT: Hashable](
    v_leaf: UCTNode[StateT, ActionT],
    transition: Transition[StateT, ActionT],
    action_map: Callable[[StateT], Sequence[ActionT]],
    max_steps: int,
) -> float:
    step, r = 0, 0
    s = v_leaf.state

    if v_leaf.is_terminal:
        return r

    while step < max_steps:
        actions = action_map(s)
        a = actions[np.random.randint(len(actions))]
        (s, _r), end = transition(s, a)
        r += _r
        if end:
            return 1
        step += 1
    return -1


def _backup[StateT, ActionT: Hashable](
    v_leaf: UCTNode[StateT, ActionT], delta: float
) -> None:
    v = v_leaf
    while v:
        v.n += 1
        v.q += delta
        v = v.parent


def mcts[StateT, ActionT: Hashable](
    s0: StateT,
    Cp: float,
    budget: int,
    transition: Transition[StateT, ActionT],
    action_map: Callable[[StateT], Sequence[ActionT]],
    max_steps: int,
    tree: UCTree[StateT, ActionT] | None = None,
    eps: float = 1,
    verbose: bool = True,
) -> tuple[ActionT, UCTree[StateT, ActionT]]:
    """
    Effectively implementing the UCT search algorithm
    """
    s = s0
    if not tree:
        tree = UCTree(s, Cp)
    for _ in tqdm(range(budget), desc="MCTS", disable=not verbose):
        v_leaf = _tree_policy(tree, Cp, transition, action_map, eps)
        delta = _default_policy(v_leaf, transition, action_map, max_steps)
        _backup(v_leaf, delta)

    v_best = _best_child(tree.root, 0)
    return cast(ActionT, v_best.action), tree
