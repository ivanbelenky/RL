import numpy as np
from numpy.linalg import norm as lnorm

from rl.mdp import MDP
from rl.utils import (
    MAX_ITER,
    MAX_STEPS,
    TOL,
    Policy,
    Qpi,
    Samples,
    Vpi,
    VQPi,
    _get_sample_step,
)

# TODO: refactor, docs


def get_sample(MDP, v, q, π, n_iter):
    _idx = n_iter
    # TODO: refactor, there is no states tabular index here
    # and there is not stateaction
    _v, _q = Vpi(v.copy(), MDP.states), Qpi(q.copy(), MDP.stateaction)
    _pi = None
    return (_idx, _v, _q, _pi)


# There are in-place methods optional. That is, there is actually an
# inplace sweep of all states instantaneously, through the vectorization
# of the update equations for DP methods. Should be Faster to execute and
# slower to converge. But tests must be carried out to verify this claim.


def vq_pi_iter_naive(
    MDP,
    policy: Policy,
    tol: float = TOL,
    inplace=False,
    max_iters: int = MAX_STEPS,
    samples: int = 1000,
) -> tuple[VQPi, Samples]:
    sample_step = _get_sample_step(samples, max_iters // 10)  # RULE OF THUMB

    v, q, samples = _vq_pi_iter_naive(MDP, policy, tol, max_iters, inplace, sample_step)

    return VQPi((v, q, policy)), samples


def _inplace_step_pe(MDP, v_i, _, π_sa, r_sa, p_s, γ):
    for s in range(MDP.S):
        v_i[s] = np.dot(π_sa[s], r_sa[:, s])
        v_i[s] += γ * np.dot(p_s[s] @ v_i, π_sa[s])
    return v_i


def _naive_step_pe(_, v_i, v_i_1, π_sa, r_sa, p_s, γ):
    v_i = np.diag(π_sa @ r_sa)
    v_i = v_i + γ * np.diag((p_s @ v_i_1) @ π_sa.T)
    return v_i


# pe: policy evaluation
ITER_NAIVE_STEP_MAP = {"inplace": _inplace_step_pe, "naive": _naive_step_pe}


def _vq_pi_iter_naive(MDP, policy, tol, max_iters, inplace, sample_step):
    γ = MDP.gamma
    p_s = MDP.p_s

    v_i = np.ones(MDP.S)
    diff_norm = TOL * 2

    update_step = ITER_NAIVE_STEP_MAP["inplace" if inplace else "naive"]

    π_sa = np.array([policy.π(s) for s in range(MDP.S)])  # SxA
    r_sa = np.array(
        [[MDP.r_sa(s, a) for s in range(MDP.S)] for a in range(MDP.A)]
    )  # AxS

    n_iter, samples = 0, []
    while (n_iter < max_iters) and (diff_norm > tol):
        v_i_1 = v_i.copy()
        v_i = update_step(MDP, v_i, v_i_1, π_sa, r_sa, p_s, γ)
        diff_norm = lnorm(v_i - v_i_1)
        n_iter += 1

        # TODO: fix this shit
        # if n_iter % sample_step == 0:
        #    samples.append(get_sample(MDP, v_i, None, policy, n_iter))

    vπ = v_i
    qπ = r_sa + (p_s @ vπ).T

    return vπ, qπ  # TODO: eventually return samples, they are broken AF


def policy_iteration(
    MDP,
    policy: Policy,
    tol_eval: float = TOL,
    max_iters_eval: int = MAX_ITER,
    tol_opt: float = TOL,
    max_iters_opt: int = MAX_ITER,
    samples: int = 1000,
) -> VQPi:
    (v_i_1, q_i_1, _), final_samples = vq_pi_iter_naive(
        MDP,
        policy,
        tol_eval,
        max_iters_eval,
        samples=samples,
    )
    v_i, q_i = v_i_1.copy(), q_i_1.copy()

    diff_norm = 2 * tol_opt

    n_iter = 0

    while (n_iter < max_iters_opt) and (diff_norm > tol_opt):
        v_i_1 = v_i.copy()
        q_i_1 = q_i.copy()

        policy.update_policy(q_i_1)
        vq_i, _samples = vq_pi_iter_naive(MDP, policy, tol_eval, max_iters_eval)

        n_iter += 1
        diff_norm = lnorm(v_i.v - v_i_1.v)

    return vq_i


def _inplace_step_vi(MDP, v_i, _, r_sa, p_s, γ):
    for s in range(MDP.S):
        v_i[s] = np.max(r_sa[:, s] + γ * (p_s[s] @ v_i))
    return v_i, None


def _naive_step_vi(_, v_i, v_i_1, r_sa, p_s, γ):
    q_i = r_sa + γ * (p_s @ v_i_1).T
    v_i = np.max(q_i, axis=0)
    return v_i, q_i


VALUE_ITERATION_STEP_MAP = {"inplace": _inplace_step_vi, "naive": _naive_step_vi}


def value_iteration(
    MDP,
    policy: Policy | None = None,
    inplace: bool = False,
    tol: float = TOL,
    max_iters: int = MAX_ITER,
    samples: int = 1000,
) -> VQPi:
    sample_step = _get_sample_step(samples, max_iters // 10)  # RULE OF THUMB

    v, q = _value_iteration(MDP, policy, tol, max_iters, inplace, sample_step)

    return VQPi(v, q, policy)


def _value_iteration(mdp: MDP, policy, tol, max_iters, inplace, sample_step):
    policy = policy if policy else MDP.policy

    γ = mdp.gamma
    p_s = mdp.p_s

    v_i = np.ones(mdp.S)
    diff_norm = TOL * 2

    update_step = VALUE_ITERATION_STEP_MAP["inplace" if inplace else "naive"]

    r_sa = np.array(
        [[mdp.r_sa(s, a) for s in range(mdp.S)] for a in range(mdp.A)]
    )  # AxS

    n_iter, samples = 0, []
    while (n_iter < max_iters) and (diff_norm > tol):
        v_i_1 = v_i.copy()
        v_i, q_i = update_step(mdp, v_i, v_i_1, r_sa, p_s, γ)
        diff_norm = lnorm(v_i - v_i_1)
        n_iter += 1

        # if n_iter % sample_step == 0:
        #    samples.append(get_sample(MDP, v_i, q_i, policy, n_iter))

    policy.update_policy(q_i)
    return Vpi(v_i, mdp.states), Qpi(q_i, mdp.stateaction)
