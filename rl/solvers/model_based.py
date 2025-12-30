from __future__ import annotations

import numpy as np
from numpy.linalg import norm as lnorm

from rl.utils import (
    MAX_ITER,
    MAX_STEPS,
    TOL,
    Qpi,
    Vpi,
    VQPi,
    _get_sample_step,
)


def get_sample(mdp, v, q, π, n_iter):
    _idx = n_iter
    # TODO: refactor, there is no states tabular index here
    # and there is not stateaction
    _v, _q = Vpi(v.copy(), mdp.states), Qpi(q.copy(), mdp.stateaction)
    _pi = None
    return (_idx, _v, _q, _pi)


# NOTE: There are in-place methods optional. That is, there is actually an
# inplace sweep of all states instantaneously, through the vectorization
# of the update equations for DP methods. Should be Faster to execute and
# slower to converge. But tests must be carried out to verify this claim.


def vq_pi_iter_naive(
    mdp: MDP,
    policy: MarkovPolicy,
    tol: float = TOL,
    inplace=False,
    max_iters: int = MAX_STEPS,
    samples: int = 1000,
) -> VQPi:
    # TODO: to be used
    _sample_step = _get_sample_step(samples, max_iters // 10)  # RULE OF THUMB

    v, q = _vq_pi_iter_naive(mdp, policy, tol, max_iters, inplace)

    return VQPi((v, q, policy))


def _inplace_step_pe(mdp: MDP, v_i, _, π_sa, r_sa, p_s, γ):
    for s in range(mdp.states.N):
        v_i[s] = np.dot(π_sa[s], r_sa[:, s])
        v_i[s] += γ * np.dot(p_s[s] @ v_i, π_sa[s])
    return v_i


def _naive_step_pe(_, v_i, v_i_1, π_sa, r_sa, p_s, γ):
    v_i = np.diag(π_sa @ r_sa)
    v_i = v_i + γ * np.diag((p_s @ v_i_1) @ π_sa.T)
    return v_i


# pe: policy evaluation
ITER_NAIVE_STEP_MAP = {"inplace": _inplace_step_pe, "naive": _naive_step_pe}


def _vq_pi_iter_naive[S: int, A: int](
    mdp: MDP[S, A],
    policy: MarkovPolicy,
    tol: float,
    max_iters: int,
    inplace: bool,
) -> tuple[Vpi, Qpi]:
    γ = mdp.gamma
    p_s = mdp.p_s

    v_i = np.ones(mdp.states.N)
    diff_norm = TOL * 2

    update_step = ITER_NAIVE_STEP_MAP["inplace" if inplace else "naive"]

    π_sa = np.array([policy.π(s) for s in range(mdp.states.N)])  # SxA
    r_sa = np.array(
        [[mdp.r_sa(s, a) for s in range(mdp.states.N)] for a in range(mdp.actions.N)]
    )  # AxS

    n_iter = 0
    while (n_iter < max_iters) and (diff_norm > tol):
        v_i_1 = v_i.copy()
        v_i = update_step(mdp, v_i, v_i_1, π_sa, r_sa, p_s, γ)
        diff_norm = lnorm(v_i - v_i_1)
        n_iter += 1

    vπ = v_i
    qπ = r_sa + (p_s @ vπ).T

    return Vpi(vπ, idx=mdp.states), Qpi(qπ, idx=mdp.stateaction)


def policy_iteration(
    mdp: MDP,
    policy: MarkovPolicy,
    tol_eval: float = TOL,
    max_iters_eval: int = MAX_ITER,
    tol_opt: float = TOL,
    max_iters_opt: int = MAX_ITER,
) -> VQPi:
    (v_i_1, q_i_1, _) = vq_pi_iter_naive(
        mdp,
        policy,
        tol_eval,
        max_iters_eval,
    )

    v_i, q_i = v_i_1.copy(), q_i_1.copy()

    diff_norm = 2 * tol_opt

    n_iter = 0

    while (n_iter < max_iters_opt) and (diff_norm > tol_opt):
        v_i_1 = v_i.copy()
        q_i_1 = q_i.copy()

        policy.update_policy(q_i_1)
        (v_i, q_i, _) = vq_pi_iter_naive(mdp, policy, tol_eval, max_iters_eval)

        n_iter += 1
        diff_norm = lnorm(v_i.v - v_i_1.v)

    return VQPi((v_i, q_i, mdp.policy))


def _inplace_step_vi(mdp, v_i, _, r_sa, p_s, γ):
    for s in range(mdp.S):
        v_i[s] = np.max(r_sa[:, s] + γ * (p_s[s] @ v_i))
    return v_i, None


def _naive_step_vi(_, v_i, v_i_1, r_sa, p_s, γ):
    q_i = r_sa + γ * (p_s @ v_i_1).T
    v_i = np.max(q_i, axis=0)
    return v_i, q_i


VALUE_ITERATION_STEP_MAP = {"inplace": _inplace_step_vi, "naive": _naive_step_vi}


def value_iteration(
    mdp: MDP,
    policy: MarkovPolicy,
    inplace: bool = False,
    tol: float = TOL,
    max_iters: int = MAX_ITER,
    samples: int = 1000,
) -> VQPi:
    # TODO: to be used
    _sample_step = _get_sample_step(samples, max_iters // 10)  # RULE OF THUMB

    v, q = _value_iteration(mdp, policy, tol, max_iters, inplace)

    return VQPi((v, q, policy))


def _value_iteration(
    mdp: MDP,
    policy: MarkovPolicy,
    tol: float,
    max_iters: int,
    inplace: bool,
) -> tuple[Vpi, Qpi]:
    policy = policy if policy else mdp.policy

    γ = mdp.gamma
    p_s = mdp.p_s

    v_i = np.ones(mdp.states.N)
    diff_norm = TOL * 2

    update_step = VALUE_ITERATION_STEP_MAP["inplace" if inplace else "naive"]

    r_sa = np.array(
        [[mdp.r_sa(s, a) for s in range(mdp.states.N)] for a in range(mdp.actions.N)]
    )  # AxS

    n_iter = 0
    while (n_iter < max_iters) and (diff_norm > tol):
        v_i_1 = v_i.copy()
        v_i, q_i = update_step(mdp, v_i, v_i_1, r_sa, p_s, γ)
        diff_norm = lnorm(v_i - v_i_1)
        n_iter += 1

    policy.update_policy(q_i)
    return Vpi(v_i, mdp.states), Qpi(q_i, mdp.stateaction)


from rl.mdp import MDP, MarkovPolicy  # noqa
