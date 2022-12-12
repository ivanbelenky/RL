"""
RL - Copyright © 2023 Iván Belenky @Leculette
"""

import numpy as np

from policy import Policy 

MAX_ITER = int(1E4)
TOL = 1E-6


def vq_π_iter_naive(
    MDP, 
    policy: Policy, 
    tol: float = TOL,
    max_iters: int = MAX_ITER) -> np.ndarray:

    γ = MDP.gamma
    p_s = MDP.p_s

    vᵢ = np.ones(MDP.S)
    diff_norm = TOL*2

    π_sa = np.array([policy.π(s) for s in range(MDP.S)]) #SxA
    r_sa  = np.array([[MDP.r_sa(s,a) for s in range(MDP.S)]   
        for a in range(MDP.A)]) #AxS

    n_iter = 0
    while (n_iter < max_iters) and (diff_norm > tol):
        vᵢ_1 = vᵢ.copy()
        
        vᵢ = np.diag(π_sa @ r_sa)
        vᵢ = vᵢ + γ * np.diag((p_s @ vᵢ_1) @ π_sa.T)

        diff_norm = np.linalg.norm(vᵢ - vᵢ_1)
        n_iter += 1
    
    vπ = vᵢ
    qπ = r_sa + (p_s @ vπ).T

    return vπ, qπ


def policy_iteration(
    MDP, 
    policy: Policy,
    tol_eval: float = TOL,
    max_iters_eval: int = MAX_ITER,
    tol_opt: float = TOL,
    max_iters_opt: int = MAX_ITER) -> np.ndarray:

    vᵢ_1, q_i_1 = vq_π_iter_naive(MDP, policy, tol_eval, max_iters_eval)
    vᵢ, q_i = vᵢ_1.copy(), q_i_1.copy()

    diff_norm = 2*tol_opt

    n_iter = 0
    while (n_iter < max_iters_opt) and (diff_norm > tol_opt):
        vᵢ_1 = vᵢ.copy()
        q_i_1 = q_i.copy()

        policy.update_policy(q_i_1)
        vᵢ, q_i = vq_π_iter_naive(MDP, policy, tol_eval, max_iters_eval)
        
        n_iter += 1 
        diff_norm = np.linalg.norm(vᵢ - vᵢ_1)
    
    return vᵢ, q_i
    

def value_iteration(
    MDP, 
    policy: Policy,
    tol: float = TOL,
    max_iters: int = MAX_ITER) -> np.ndarray:

    γ = MDP.gamma
    p_s = MDP.p_s

    vᵢ = np.ones(MDP.S)
    diff_norm = TOL*2
    
    r_sa  = np.array([[MDP.r_sa(s,a) for s in range(MDP.S)]   
        for a in range(MDP.A)]) #AxS

    n_iter = 0
    while (n_iter < max_iters) and (diff_norm > tol):
        vᵢ_1 = vᵢ.copy()
        
        qᵢ = r_sa + γ * (p_s @ vᵢ_1).T
        vᵢ = np.array([np.max(qᵢ[:,s]) for s in range(MDP.S)])

        diff_norm = np.linalg.norm(vᵢ - vᵢ_1)
        n_iter += 1

    policy.update_policy(qᵢ)
    