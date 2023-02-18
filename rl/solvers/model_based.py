from typing import Tuple

import numpy as np
from numpy.linalg import norm as lnorm

from rl.utils import (
    Policy,
    _typecheck_all,
    _get_sample_step,
    _check_ranges,
    VQPi,
    Samples,
    Transition,
    Vpi,
    Qpi,
    PQueue,
    MAX_ITER,
    MAX_STEPS,
    TOL
) 

#TODO: refactor, docs

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


def vq_pi_iter_naive(MDP, policy: Policy, tol: float=TOL, inplace=False, 
    max_iters: int=MAX_STEPS) -> Tuple[VQPi, Samples]:

    sample_step = _get_sample_step(samples, max_iters//10) # RULE OF THUMB

    v, q, samples = _vq_pi_iter_naive(MDP, policy, tol, max_iters, inplace,
        sample_step)
    
    return VQPi((v, q, policy)), samples


def _inplace_step_pe(MDP, vᵢ, _, π_sa, r_sa, p_s, γ):
    for s in range(MDP.S):
        vᵢ[s] = np.dot(π_sa[s], r_sa[:,s])
        vᵢ[s] += γ * np.dot(p_s[s] @ vᵢ, π_sa[s])
    return vᵢ


def _naive_step_pe(_, vᵢ, vᵢ_1, π_sa, r_sa, p_s, γ):
    vᵢ = np.diag(π_sa @ r_sa)
    vᵢ = vᵢ + γ * np.diag((p_s @ vᵢ_1) @ π_sa.T)
    return vᵢ


# pe: policy evaluation
ITER_NAIVE_STEP_MAP = {
    'inplace': _inplace_step_pe,
    'naive': _naive_step_pe
}


def _vq_pi_iter_naive(MDP, policy, tol, max_iters, inplace, sample_step):
    γ = MDP.gamma
    p_s = MDP.p_s

    vᵢ = np.ones(MDP.S)
    diff_norm = TOL*2

    update_step = ITER_NAIVE_STEP_MAP['inplace' if inplace else 'naive']

    π_sa = np.array([policy.π(s) for s in range(MDP.S)]) #SxA
    r_sa  = np.array([[MDP.r_sa(s,a) for s in range(MDP.S)]
        for a in range(MDP.A)]) #AxS
    
    n_iter, samples = 0, []
    while (n_iter < max_iters) and (diff_norm > tol):
        vᵢ_1 = vᵢ.copy()
        vᵢ = update_step(MDP, vᵢ, vᵢ_1, π_sa, r_sa, p_s, γ)
        diff_norm = lnorm(vᵢ - vᵢ_1)
        n_iter += 1
        
        if n_iter % sample_step == 0:
            samples.append(get_sample(MDP, vᵢ, None, policy, n_iter))

    vπ = vᵢ
    qπ = r_sa + (p_s @ vπ).T

    return vπ, qπ


def policy_iteration(MDP, policy: Policy, tol_eval: float = TOL,
    max_iters_eval: int = MAX_ITER, tol_opt: float = TOL,
    max_iters_opt: int = MAX_ITER, samples: int=1000
    ) -> Tuple[VQPi, Samples]:

    vᵢ_1, q_i_1 = vq_pi_iter_naive(MDP, policy, tol_eval, max_iters_eval)
    vᵢ, q_i = vᵢ_1.copy(), q_i_1.copy()

    diff_norm = 2*tol_opt

    n_iter = 0

    while (n_iter < max_iters_opt) and (diff_norm > tol_opt):
        vᵢ_1 = vᵢ.copy()
        q_i_1 = q_i.copy()

        policy.update_policy(q_i_1)
        vᵢ, q_i = vq_pi_iter_naive(MDP, policy, tol_eval, max_iters_eval)
        
        n_iter += 1 
        diff_norm = lnorm(vᵢ - vᵢ_1)
    
    return vᵢ, q_i, samples


def _inplace_step_vi(MDP, vᵢ, _, r_sa, p_s, γ):
    for s in range(MDP.S):
        vᵢ[s] = np.max(r_sa[:,s] + γ * (p_s[s] @ vᵢ))
    return vᵢ, None


def _naive_step_vi(_, vᵢ, vᵢ_1, r_sa, p_s, γ):
    qᵢ = r_sa + γ * (p_s @ vᵢ_1).T
    vᵢ = np.max(qᵢ, axis=0)
    return vᵢ, qᵢ


VALUE_ITERATION_STEP_MAP = {
    'inplace': _inplace_step_vi,
    'naive': _naive_step_vi
}


def value_iteration(MDP, policy: Policy = None, inplace: bool=False, 
    tol: float = TOL, max_iters: int=MAX_ITER) -> Tuple[VQPi, Samples]:

    sample_step = _get_sample_step(samples, max_iters//10) # RULE OF THUMB

    v, q, samples = _value_iteration(MDP, policy, tol, max_iters, inplace,
        sample_step)

    return VQPi((v, q, policy)), samples


def _value_iteration(MDP, policy, tol, max_iters, inplace, sample_step):
    policy = policy if policy else MDP.policy

    γ = MDP.gamma
    p_s = MDP.p_s

    vᵢ = np.ones(MDP.S)
    diff_norm = TOL*2
    
    update_step = VALUE_ITERATION_STEP_MAP['inplace' if inplace else 'naive']

    r_sa  = np.array([[MDP.r_sa(s,a) for s in range(MDP.S)]   
        for a in range(MDP.A)]) #AxS

    n_iter, samples = 0, []
    while (n_iter < max_iters) and (diff_norm > tol):
        vᵢ_1 = vᵢ.copy()        
        vᵢ, qᵢ = update_step(MDP, vᵢ, vᵢ_1, r_sa, p_s, γ)
        diff_norm = lnorm(vᵢ - vᵢ_1)
        n_iter += 1

        if n_iter % sample_step == 0:
            samples.append(get_sample(MDP, vᵢ, qᵢ, policy, n_iter))

    policy.update_policy(qᵢ)