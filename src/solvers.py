"""
RL - Copyright © 2023 Iván Belenky @Leculette
"""

from typing import (
    Tuple, 
    Sequence, 
    Callable,
    Dict,
    List, 
    Any, 
    NewType)

import numpy as np
from numpy.linalg import norm as lnorm

from model_free import (
    ModelFree,
    ModelFreePolicy,
    EpsilonSoftPolicy)
from utils import (
    Policy,
    _typecheck_all,
    _get_sample_step,
    VQPi,
    Samples,
    Transition,
    Vpi,
    Qpi,
    MAX_ITER,
    MAX_STEPS) 


def get_sample(v, q, π, n_episode, optimize):
    _idx, _v, _q = n_episode, Vpi(v.copy()), Qpi(q.copy())
    _pi = None
    if optimize:
        _pi = π.pi.copy() 
    return (_idx, _v, _q, _pi)


def tdn(states: Sequence[Any], actions: Sequence[Any], transition: Transition,
    gamma: float=0.9, n: int=1, alpha: float=0.1, n_episodes: int=MAX_ITER,
    policy: ModelFreePolicy=None, optimize: bool=False, samples: int = 1000,
    max_steps: int=MAX_STEPS) -> Tuple[VQPi, Samples]:
    '''N-temporal differences algorithm.

    Temporal differences algorithm for estimating the value function of a
    policy, improve it and analyze it.

    Parameters
    ----------
    states : Sequence[Any]
    actions : Sequence[Any]
    transition : Callable[[Any,Any],[[Any,float], bool]]]
        transition must be a callable function that takes as arguments the
        (state, action) and returns (new_state, reward), end.
    gamma : float, optional
        Discount factor, by default 0.9
    n : int, optional
        Number of steps to look ahead, by default 1
    alpha : float, optional
        Learning rate, by default 0.1
    n_episodes : int, optional
        Number of episodes to simulate, by default 1E4
    max_steps : int, optional
        Maximum number of steps per episode, by default 1E3
    policy : ModelFreePolicy, optional
        Policy to use, by default equal probability ModelFreePolicy
    optimize : bool, optional
        Whether to optimize the policy or not, by default False
    samples : int, optional
        Number of samples to take, by default 1000
    
    Returns
    -------
    VQPi : Tuple[np.ndarray, np.ndarray, Policy, Samples]
        Value function, action-value function, policy and samples if any.

    Raises
    ------
    TypeError: If any of the arguments is not of the correct type.

    Examples
    --------
    Define state action pairs
    >>> from rl import tdn
    >>> states = [0]
    >>> actions = ['left', 'right']
    Define the transition method, taking (state, action)
    and returning (new_state, reward), end
    >>> def state_transition(state, action):
    >>>   if action == 'right':
    >>>     return (state, 0), True
    >>>   if action == 'left':
    >>>     threshold = np.random.random()
    >>>   if threshold > 0.9:
    >>>     return (state, 1), True
    >>>   else:
    >>>     return (state, 0), False
    Solve!
    >>> tdn(states, actions, state_transition, gamma=1, n=3, alpha=0.05)
    (array([0.134]), array([[0.513., 0.]]), <class 'ModelFreePolicy'>, None)
    '''    
    _typecheck_all(tabular_idxs=[states,actions], callables=[transition],
        constants=[gamma, n, alpha, n_episodes, samples, max_steps], 
        booleans=[optimize], policies=[policy])

    sample_step = _get_sample_step(samples, n_episodes)

    model = ModelFree(states, actions, transition, gamma=gamma, policy=policy)    
    v, q, samples = _tdn(model, n, alpha, n_episodes, max_steps, optimize,
        sample_step)
    
    return VQPi((v, q, model.policy.pi)), samples


def _td_step(s, a, r, t, T, n, v, q, γ, α, gammatron):
    '''td step update'''
    s_t, a_t, rr = s[t], a[t], r[t:t+n]
    G = np.dot(gammatron[:rr.shape[0]], rr)
    if t + n < T:
        G = G + γ**n * v[s[t+n]]
    v[s_t] = v[s_t] + α * (G - v[s_t])
    q_key = (s_t, a_t)
    q[q_key] = q[q_key] + α * (G - q[q_key])


def _tdn(MF, n, alpha, n_episodes, max_steps, optimize, sample_step):
    '''N-temporal differences algorithm.'''
    π = MF.policy
    α = alpha
    γ = MF.gamma

    v, q = MF.init_vq()    
    gammatron = np.array([γ**i for i in range(n)])
    
    samples = []

    n_episode = 0
    while n_episode < n_episodes:
        s_0, a_0 = MF.random_sa(value=True)
        episode = MF.generate_episode(s_0, a_0, π, max_steps)
        
        sar = np.array(episode)
        s, a, r = sar[:,0], sar[:,1], sar[:,2]
        T = s.shape[0]
        
        for t in range(T):
            _td_step(s, a, r, t, T, n, v, q, γ, α, gammatron)
            if optimize:
                π.update(q, s[t]) # inplace update
        n_episode += 1

        if sample_step and n_episode % sample_step == 0:
            samples.append(get_sample(v, q, π, n_episode))
    
    return v, q, samples






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

        diff_norm = lnorm(vᵢ - vᵢ_1)
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
        diff_norm = lnorm(vᵢ - vᵢ_1)
    
    return vᵢ, q_i
    

def value_iteration(
    MDP, 
    policy: Policy = None,
    tol: float = TOL,
    max_iters: int = MAX_ITER) -> np.ndarray:

    policy = policy if policy else MDP.policy

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

        diff_norm = lnorm(vᵢ - vᵢ_1)
        n_iter += 1

    policy.update_policy(qᵢ)
    


def first_visit_monte_carlo(MF, policy, max_episodes, max_steps, es, optimize):
    return _visit_monte_carlo(MF, policy, max_episodes, max_steps,
    first_visit=True, exploring_starts=es, optimize=optimize)


def every_visit_monte_carlo(MF, policy, max_episodes, max_steps, es, optimize):
    return _visit_monte_carlo(MF, policy, max_episodes, max_steps, 
        first_visit = False, exploring_starts=es, optimize=optimize)


def _mc_step(v, q, t, s_t, a_t, s, a, n_s, n_sa, G, first_visit):
    if s_t not in s[:-(t+1)] or not first_visit:
        n_s[s_t] = n_s[s_t] + 1
        v[s_t] = v[s_t] + (G - v[s_t])/n_s[s_t]
    
    q_key = (s_t, a_t)
    if q_key not in zip(s[:-(t+1)],a[:-(t+1)]) or not first_visit:    
        n_sa[q_key] = n_sa[q_key] + 1
        q[q_key] = q[q_key] + (G - q[q_key])/n_sa[q_key]
        return True
    
    return False


def _visit_monte_carlo(
    MF,
    policy: Policy = None,
    max_episodes: int = MAX_ITER,
    max_steps: int = MAX_STEPS,
    first_visit = True,
    exploring_starts = False,
    optimize = False) -> np.ndarray:
    '''
    '''
    v, q = np.zeros(MF.states.N), np.zeros((MF.states.N, MF.actions.N))
    n_s, n_sa = np.zeros(MF.states.N), np.zeros((MF.states.N, MF.actions.N))
    π = policy if policy else MF.policy
    γ = MF.gamma
    s_0, a_0 = MF.random_sa(value=True)

    n_episode = 0
    while n_episode < max_episodes:
        if exploring_starts:
            s_0, a_0 = MF.random_sa(value=True)

        episode = MF.generate_episode(s_0, a_0, π, max_steps)
        sar = np.array(episode)
        s, a, _ = sar.T
        
        G = 0   
        for t, (s_t, a_t, r_tt) in enumerate(sar[::-1]):
            s_t, a_t = int(s_t), int(a_t)
            G = γ*G + r_tt
            update = _mc_step(v, q, t, s_t, a_t, s, a, n_s,
                n_sa, G, first_visit)
            
            if optimize and update:
                π.update_policy(q, s_t)

        n_episode += 1
        
    return v, q


def off_policy_first_visit(MF, off_policy, policy, max_episodes, max_steps,
    ordinary, optimize):
    return _off_policy_monte_carlo(MF, off_policy, policy, max_episodes, 
        max_steps, optimize=optimize, ordinary=ordinary, first_visit=True)


def off_policy_every_visit(MF, off_policy, policy, max_episodes, max_steps,
    ordinary, optimize):
    return _off_policy_monte_carlo(MF, off_policy, policy, max_episodes, 
        max_steps, optimize=optimize, ordinary=ordinary, first_visit=False)


def _mc_step_off(q, v, t, s_t, a_t, s, a, G, w, c, c_q, 
    first_visit, ordinary):
    
    c_add = 1 if ordinary else w
    denom = w if ordinary else 1    

    if s_t not in s[:-(t+1)] or not first_visit:
        c[s_t] = c[s_t] + c_add
        v[s_t] = v[s_t] + w/c[s_t] * (G - v[s_t]/denom)

    q_key = (s_t, a_t)
    if q_key not in zip(s[:-(t+1)],a[:-(t+1)]) or not first_visit:
        c_q[q_key] = c_q[q_key] + c_add
        q[q_key] = q[q_key] + w/c_q[q_key] * (G - q[q_key]/denom)
        return True

    return False


def _off_policy_monte_carlo(
    MF,
    off_policy: Policy,
    policy: Policy = None,
    max_episodes: int = MAX_ITER,
    max_steps: int = MAX_STEPS,
    first_visit = True,
    ordinary = False,
    optimize = False
    ) -> Tuple[np.ndarray, np.ndarray]:

    γ = MF.gamma
    b = off_policy 
    π = policy if policy else MF.policy

    n_episode = 0

    v, q = np.zeros(MF.states.N), np.zeros((MF.states.N, MF.actions.N))
    c, c_q = np.zeros(MF.states.N), np.zeros((MF.states.N, MF.actions.N))

    while n_episode < max_episodes:
        G = 0
        s_0, a_0 = MF.random_sa(value=True)
        episode = MF.generate_episode(s_0, a_0, b, max_steps)
        sar = np.array(episode)
        s, a, _ = sar.T

        w = 1
        for t, (s_t, a_t, r_tt) in enumerate(sar[::-1]):
            s_t, a_t = int(s_t), int(a_t)
            rho = π.pi_as(a_t, s_t)/b.pi_as(a_t, s_t)
            if rho == 0: break

            G = γ*G + r_tt
            
            update = _mc_step_off(q, v, t, s_t, a_t, s, a, 
                G, w, c, c_q, first_visit, ordinary)
            
            w = w*rho 
            
            if update and optimize:
                π.update_policy(q, s_t) 

        n_episode += 1
    
    return v, q


# temporal difference control SARSA, QLeearning, and some others


