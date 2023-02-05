"""
RL - Copyright © 2023 Iván Belenky @Leculette
"""
from typing import (
    Tuple, 
    Sequence,  
    Any
)

import numpy as np
from numpy.linalg import norm as lnorm

from model_free import (
    ModelFree,
    ModelFreePolicy,
    EpsilonSoftPolicy
)
from utils import (
    Policy,
    _typecheck_all,
    _get_sample_step,
    _check_ranges,
    VQPi,
    Samples,
    Transition,
    Vpi,
    Qpi,
    MAX_ITER,
    MAX_STEPS,
    TOL
) 


def get_sample(v, q, π, n_episode, optimize):
    _idx, _v, _q = n_episode, Vpi(v.copy()), Qpi(q.copy())
    _pi = None
    if optimize:
        _pi = π.pi.copy() 
    return (_idx, _v, _q, _pi)


def vq_π_iter_naive(MDP, policy: Policy, tol: float = TOL,
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


def policy_iteration(MDP, policy: Policy, tol_eval: float = TOL,
    max_iters_eval: int = MAX_ITER, tol_opt: float = TOL,
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
    

def value_iteration(MDP, policy: Policy = None, tol: float = TOL,
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



def alpha_mc(states: Sequence[Any], actions: Sequence[Any], transition: Transition,
    gamma: float=0.9, alpha: float=0.05, use_N :bool=False, first_visit: bool=True,
    exploring_starts: bool=False, n_episodes: int=MAX_ITER, max_steps: int=MAX_STEPS,
    samples: int=1000, optimize: bool=False, policy: ModelFreePolicy=None, 
    eps: float=None) -> Tuple[VQPi, Samples]:
    '''α-MC state and action-value function estimation, policy optimization

    Alpha weighted Monte Carlo state and action-value function estimation, policy
    optimization. By setting use_N to True, it will use the classical weighting 
    schema, utilizing N(s) instead of a contstant α. 

    Parameters
    ----------
    states : Sequence[Any]
    actions : Sequence[Any]
    transition : Callable[[Any,Any],[[Any,float], bool]]]
        transition must be a callable function that takes as arguments the
        (state, action) and returns (new_state, reward), end.
    gamma : float, optional
        Discount factor, by default 0.9
    alpha : float, optional
        Learning rate, by default 0.1
    use_N : bool, optional
        If true, it will use 1/N(s) (number of visits) instead of α
    first_visit : bool, optional
        If true, it will only use the first visit to a state, by default True
    exploring_starts : bool, optional
        Random action at the start of each episode. 
    n_episodes : int, optional
        Number of episodes to simulate, by default 1E4
    max_steps : int, optional
        Maximum number of steps per episode, by default 1E3
    samples : int, optional
        Number of samples to take, by default 1000
    optimize : bool, optional
        Whether to optimize the policy or not, by default False
    policy : ModelFreePolicy, optional
        Policy to use, by default equal probability ModelFreePolicy
    eps : float, optional
        Epsilon for the EpsilonSoftPolicy, by default None (no exploration)

    Returns
    -------
    vqpi : Tuple[VPi, QPi, Policy]
        Value function, action-value function, policy and samples if any.
    samples : Tuple[int, List[Vpi], List[Qpi], List[np.ndarray]] 
        Samples taken during the simulation if any. The first element is the
        index of the iteration, the second is the value function, the third is
        the action-value function and the fourth is the TODO:.

    Raises
    ------
    TypeError: arguments check.
    TransitionException: transition calls function checks.
    '''
    if not policy and eps:
        _check_ranges(values=[eps], ranges=[(0,1)])
        policy = EpsilonSoftPolicy(states, actions, eps=eps)
    elif not policy:
        policy = ModelFreePolicy(states, actions)

    _typecheck_all(tabular_idxs=[states, actions],transition=transition,
        constants=[gamma, alpha, n_episodes, max_steps, samples],
        booleans=[use_N, first_visit, exploring_starts, optimize],
        policies=[policy])

    _check_ranges(values=[gamma, alpha, n_episodes, max_steps, samples],
        ranges=[(0,1), (0,1), (1,np.inf), (1,np.inf), (1,1001)])

        
    sample_step = _get_sample_step(samples, n_episodes)

    model = ModelFree(states, actions, transition, gamma=gamma, policy=policy)    
    v, q, samples = _visit_monte_carlo(model, first_visit, exploring_starts, use_N,
        alpha, n_episodes, max_steps, optimize, sample_step) 

    return VQPi((v, q, model.policy.pi)), samples


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


def _mc_step_α(v, q, t, s_t, a_t, s, a, α, G, first_visit):
    if s_t not in s[:-(t+1)] or not first_visit:
        v[s_t] = v[s_t] + α*(G - v[s_t])
    
    q_key = (s_t, a_t)
    if q_key not in zip(s[:-(t+1)],a[:-(t+1)]) or not first_visit:    
        q[q_key] = q[q_key] + α*(G - q[q_key])
        return True
    
    return False


def _visit_monte_carlo(MF, first_visit, exploring_starts, use_N, alpha, 
    n_episodes, max_steps, optimize, sample_step):
    
    π = MF.policy
    γ = MF.gamma
    α = alpha

    samples = []

    v, q = np.zeros(MF.states.N), np.zeros((MF.states.N, MF.actions.N))
    if use_N:
        n_s, n_sa = np.zeros(MF.states.N), np.zeros((MF.states.N, MF.actions.N))

    s_0, a_0 = MF.random_sa(value=True)

    n_episode = 0
    while n_episode < n_episodes:
        if exploring_starts:
            s_0, a_0 = MF.random_sa(value=True)

        episode = MF.generate_episode(s_0, a_0, π, max_steps)
        sar = np.array(episode)
        s, a, _ = sar.T
        
        G = 0   
        for t, (s_t, a_t, r_tt) in enumerate(sar[::-1]):
            s_t, a_t = int(s_t), int(a_t)
            G = γ*G + r_tt
            if use_N:
                update = _mc_step(v, q, t, s_t, a_t, s, a, n_s,
                    n_sa, G, first_visit)
            else:
                update = _mc_step_α(v, q, t, s_t, a_t, s, a,
                    α, G, first_visit)
            if optimize and update:
                π.update_policy(q, s_t)

        n_episode += 1

        if sample_step and n_episode % sample_step == 0:
            samples.append(get_sample(MF, v, q, π, n_episode))
        
    return v, q


def off_policy_mc(states: Sequence[Any], actions: Sequence[Any], transition: Transition,
    gamma: float=0.9, first_visit: bool=True, ordinary: bool=True,  
    n_episodes: int=MAX_ITER, max_steps: int=MAX_STEPS, samples: int=1000, 
    optimize: bool=False, policy: ModelFreePolicy=None, eps: float=None, 
    b :ModelFreePolicy=None) -> Tuple[VQPi, Samples]: 
    '''Off-policy Monte Carlo state and action value function estimation, policy 
    
    Off policy Monte Carlo method for estimating state and action-value functtions
    as well as optimizing policies. If no behavior policy is provided an 
    equal probability one for each (s,a) pair will be used. In order to guarantee
    convergence you must specify 

    Parameters
    ----------
    states : Sequence[Any]
    actions : Sequence[Any]
    transition : Callable[[Any,Any],[[Any,float], bool]]]
        transition must be a callable function that takes as arguments the
        (state, action) and returns (new_state, reward), end.
    gamma : float, optional
        Discount factor, by default 0.9
    first_visit : bool, optional
        If true, it will only use the first visit to a state, by default True
    ordinary : bool, optional
        ordinary sampling, beware! high variance, by default False
    n_episodes : int, optional
        Number of episodes to simulate, by default 1E4
    max_steps : int, optional
        Maximum number of steps per episode, by default 1E3
    samples : int, optional
        Number of samples to take, by default 1000
    optimize : bool, optional
        Whether to optimize the policy or not, by default False
    policy : ModelFreePolicy, optional
        Policy to use, by default equal probability ModelFreePolicy
    eps : float, optional
        Epsilon for the EpsilonSoftPolicy, by default None (no exploration)
    b : ModelFreePolicy, optional
        Behavior policy, by default None (equal probability ModelFreePolicy)

    Returns
    -------
    vqpi : Tuple[VPi, QPi, Policy]
        Value function, action-value function, policy and samples if any.
    samples : Tuple[int, List[Vpi], List[Qpi], List[np.ndarray]] 
        Samples taken during the simulation if any. The first element is the
        index of the iteration, the second is the value function, the third is
        the action-value function and the fourth is the TODO:.

    Raises
    ------
    TypeError: arguments check.
    TransitionException: transition calls function checks.
    '''
    if not policy and eps:
        _typecheck_all(constants=[eps])
        _check_ranges(values=[eps], ranges=[(0,1)])
        policy = EpsilonSoftPolicy(states, actions, eps=eps)
    elif not policy:
        policy = ModelFreePolicy(states, actions)
    elif not b:
        b = ModelFreePolicy(states, actions)

    _typecheck_all(tabular_idxs=[states, actions],transition=transition,
        constants=[gamma, n_episodes, max_steps, samples],
        booleans=[first_visit, optimize],
        policies=[policy, b])
    _check_ranges(values=[gamma, n_episodes, max_steps, samples],
        ranges=[(0,1), (1,np.inf), (1,np.inf), (1,1001)])

    sample_step = _get_sample_step(samples, n_episodes)

    model = ModelFree(states, actions, transition, gamma=gamma, policy=policy)    
    v, q, samples = _off_policy_monte_carlo(model, first_visit, n_episodes, 
        max_steps, optimize, ordinary, sample_step)

    return VQPi((v, q, policy)), samples


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


def _off_policy_monte_carlo(MF, off_policy, max_episodes, max_steps, first_visit,
    ordinary, optimize, sample_step):

    γ = MF.gamma
    b = off_policy 
    π = MF.policy

    samples = []

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

        if sample_step and n_episode % sample_step == 0:
            samples.append(get_sample(MF, v, q, π, n_episode))
    
    return v, q



def tdn(states: Sequence[Any], actions: Sequence[Any], transition: Transition,
    gamma: float=0.9, n: int=1, alpha: float=0.05, n_episodes: int=MAX_ITER,
    policy: ModelFreePolicy=None, eps: float=None, optimize: bool=False, 
    samples: int = 1000, max_steps: int=MAX_STEPS) -> Tuple[VQPi, Samples]:
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
    eps : float, optional
        Epsilon value for the epsilon-soft policy, by default None (no exploration)
    optimize : bool, optional
        Whether to optimize the policy or not, by default False
    samples : int, optional
        Number of samples to take, by default 1000
    
    Returns
    -------
    vqpi : Tuple[VPi, QPi, Policy]
        Value function, action-value function, policy and samples if any.
    samples : Tuple[int, List[Vpi], List[Qpi], List[np.ndarray]] 
        Samples taken during the simulation if any. The first element is the
        index of the iteration, the second is the value function, the third is
        the action-value function and the fourth is the TODO:.

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
    if not policy and eps:
        _typecheck_all(constants=[eps])
        _check_ranges(values=[eps], ranges=[(0,1)])
        policy = EpsilonSoftPolicy(states, actions, eps=eps)
    elif not policy:
        policy = ModelFreePolicy(states, actions)

    _typecheck_all(tabular_idxs=[states,actions], tansition=transition,
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
    gammatron = np.array([γ**i for i in range(n)])

    v, q = MF.init_vq()
    
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
            samples.append(get_sample(MF, v, q, π, n_episode))
    
    return v, q, samples


# temporal difference control SARSA, QLeearning, and some others


