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

from rl.model_free import (
    ModelFree,
    ModelFreePolicy,
    EpsilonSoftPolicy
)
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


def get_sample(MF, v, q, π, n_episode, optimize):
    _idx = n_episode
    _v, _q = Vpi(v.copy(), MF.states), Qpi(q.copy(), MF.stateaction)
    _pi = None
    if optimize:
        _pi = ModelFreePolicy(MF.actions.N, MF.states.N)
        _pi.pi = π.pi.copy()
    return (_idx, _v, _q, _pi)


def _set_s0_a0(MF, s_0, a_0):
    if not s_0:
        s_0, _ = MF.random_sa(value=True) 
    if not a_0:
        _, a_0 = MF.random_sa(value=True)

    return s_0, a_0


def _set_policy(policy, eps, actions, states):
    if not policy and eps:
        _typecheck_all(constants=[eps])
        _check_ranges(values=[eps], ranges=[(0,1)])
        policy = EpsilonSoftPolicy(actions, states, eps=eps)
    elif not policy:
        policy = ModelFreePolicy(actions, states)
    
    return policy
    

def alpha_mc(states: Sequence[Any], actions: Sequence[Any], transition: Transition,
    gamma: float=0.9, alpha: float=0.05, use_N :bool=False, first_visit: bool=True,
    exploring_starts: bool=True, n_episodes: int=MAX_ITER, max_steps: int=MAX_STEPS,
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
    TransitionException: transition calls function checks.
    '''
    policy = _set_policy(policy, eps, actions, states)

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
            samples.append(get_sample(MF, v, q, π, n_episode, optimize))

    return v, q, samples


def off_policy_mc(states: Sequence[Any], actions: Sequence[Any], transition: Transition,
    gamma: float=0.9, first_visit: bool=True, ordinary: bool=False,  
    n_episodes: int=MAX_ITER, max_steps: int=MAX_STEPS, samples: int=1000, 
    optimize: bool=False, policy: ModelFreePolicy=None, eps: float=None, 
    b: ModelFreePolicy=None) -> Tuple[VQPi, Samples]: 
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
    TransitionException: transition calls function checks.
    '''
    
    policy = _set_policy(policy, eps, actions, states)
    if not b:
        b = ModelFreePolicy(actions, states)

    _typecheck_all(tabular_idxs=[states, actions],transition=transition,
        constants=[gamma, n_episodes, max_steps, samples],
        booleans=[first_visit, optimize],
        policies=[policy, b])
    _check_ranges(values=[gamma, n_episodes, max_steps, samples],
        ranges=[(0,1), (1,np.inf), (1,np.inf), (1,1001)])

    sample_step = _get_sample_step(samples, n_episodes)

    model = ModelFree(states, actions, transition, gamma=gamma, policy=policy)    
    v, q, samples = _off_policy_monte_carlo(model, b, n_episodes, 
        max_steps, first_visit, ordinary, optimize, sample_step)

    return VQPi((v, q, policy)), samples


def _mc_step_off(q, v, t, s_t, a_t, s, a, G, w, c, c_q, 
    first_visit, ordinary):
    
    c_add = 1 if ordinary else w
    denom = w if ordinary else 1    

    if s_t not in s[:-(t+1)] or not first_visit:
        c[s_t] = c[s_t] + c_add
        if w < 1E-10:
            if ordinary:
                v[s_t] = v[s_t] - 1/c[s_t] * v[s_t]
        else:
            v[s_t] = v[s_t] + w/c[s_t] * (G - v[s_t]/denom)
        
    q_key = (s_t, a_t)
    if q_key not in zip(s[:-(t+1)],a[:-(t+1)]) or not first_visit:
        c_q[q_key] = c_q[q_key] + c_add
        if w < 1E-10:
            if ordinary:
                q[q_key] = q[q_key] - 1/c_q[q_key] * q[q_key]
        else:
            q[q_key] = q[q_key] + w/c_q[q_key] * (G - q[q_key]/denom)
        return True

    return False


def _off_policy_monte_carlo(MF, off_policy, n_episodes, max_steps, first_visit,
    ordinary, optimize, sample_step):

    γ = MF.gamma
    b = off_policy 
    π = MF.policy

    samples = []

    n_episode = 0

    v, q = np.zeros(MF.states.N), np.zeros((MF.states.N, MF.actions.N))
    c, c_q = np.zeros(MF.states.N), np.zeros((MF.states.N, MF.actions.N))

    while n_episode < n_episodes:
        G = 0.
        s_0, a_0 = MF.random_sa(value=True)
        episode = MF.generate_episode(s_0, a_0, b, max_steps)
        sar = np.array(episode)
        s, a, _ = sar.T

        w = 1.
        for t, (s_t, a_t, r_tt) in enumerate(sar[::-1]):
            if w < 1E-10:
                break

            s_t, a_t = int(s_t), int(a_t)
            
            rho = π.pi_as(a_t, s_t)/b.pi_as(a_t, s_t)
            w = w*rho 
            
            G = γ*G + r_tt
            update = _mc_step_off(q, v, t, s_t, a_t, s, a, 
                G, w, c, c_q, first_visit, ordinary)
            
            if update and optimize:
                π.update_policy(q, s_t) 
        
        n_episode += 1

        if sample_step and n_episode % sample_step == 0:
            samples.append(get_sample(MF, v, q, π, n_episode, optimize))
    
    return v, q, samples



def tdn(states: Sequence[Any], actions: Sequence[Any], transition: Transition,
    state_0: Any=None, action_0: Any=None, gamma: float=0.9, n: int=1, 
    alpha: float=0.05, n_episodes: int=MAX_ITER, policy: ModelFreePolicy=None, 
    eps: float=None, optimize: bool=False, method: str='sarsa', samples: int=1000, 
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
    state_0 : Any, optional
        Initial state, by default None (random)
    action_0 : Any, optional
        Initial action, by default None (random)
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
    policy = _set_policy(policy, eps, actions, states)

    if method not in METHODS:
        raise ValueError(
            f'Unknown method {method}\n'
            'Available methods are (sarsa, sarsa_on, qlearning, expected_sarsa'
            ', dqlearning)')

    _typecheck_all(tabular_idxs=[states,actions], transition=transition,
        constants=[gamma, n, alpha, n_episodes, samples, max_steps], 
        booleans=[optimize], policies=[policy])

    sample_step = _get_sample_step(samples, n_episodes)

    model = ModelFree(states, actions, transition, gamma=gamma, policy=policy)  
    
    _tdn = METHOD_MAP[method]

    v, q, samples = _tdn(model, state_0, action_0, n, alpha, n_episodes,
        max_steps, optimize, method, sample_step)
    
    return VQPi((v, q, policy)), samples


def _td_step(s, a, r, t, T, n, v, q, γ, α, gammatron, π=None):
    '''td step update'''
    s_t, a_t, rr = s[t], a[t], r[t:t+n]
    G = np.dot(gammatron[:rr.shape[0]], rr)
    G_v, G_q = G, G
    if t + n < T:
        G_v = G_v + (γ**n) * v[s[t+n]]
        G_q = G_q + (γ**n) * q[s[t+n], a[t+n]]

    v[s_t] = v[s_t] + α * (G_v - v[s_t])
    q_key = (s_t, a_t)
    q[q_key] = q[q_key] + α * (G_q - q[q_key])
    

def _td_qlearning(s, a, r, t, T, n, v, q, γ, α, gammatron, π=None):
    '''td qlearning update'''
    s_t, a_t, rr = s[t], a[t], r[t:t+n]
    G = np.dot(gammatron[:rr.shape[0]], rr)
    if t + n < T:
        G = G + (γ**n) * np.max(q[s[t+n]])

    v[s_t] = v[s_t] + α * (G - v[s_t])
    q_key = (s_t, a_t)
    q[q_key] = q[q_key] + α * (G - q[q_key])
    

def _td_expected_sarsa(s, a, r, t, T, n, v, q, γ, α, gammatron, π=None):
    s_t, a_t, rr = s[t], a[t], r[t:t+n]
    G = np.dot(gammatron[:rr.shape[0]], rr)
    if t + n < T:
        G = G + (γ**n) * np.dot(π.pi[s[t+n]], q[s[t+n]])
    
    v[s_t] = v[s_t] + α * (G - v[s_t])
    q_key = (s_t, a_t)
    q[q_key] = q[q_key] + α * (G - q[q_key])


STEP_MAP = {
    'sarsa': _td_step,
    'qlearning': _td_qlearning,
    'expected_sarsa': _td_expected_sarsa,  
}


def _tdn_onoff(MF, s_0, a_0, n, alpha, n_episodes, max_steps, optimize, 
    method, sample_step):
    '''N-temporal differences algorithm.
    
    This is the basic implementation of the N-temporal difference algorithm. 
    When optimizing the policy, the method for updating will be quasi-off 
    policy. That is the updates are taking place with respect to the q-values
    updated on each step, but each step corresponds to the old policy. This 
    implies that at the beginning of the updates are strictly on policy, and 
    at the end, when probably all the states have been visited, the updates 
    are off policy. 
    '''
    π = MF.policy
    α = alpha
    γ = MF.gamma
    gammatron = np.array([γ**i for i in range(n)])
    v, q = MF.init_vq()


    f_step = STEP_MAP[method]

    samples = []
    n_episode = 0
    while n_episode < n_episodes:
        if not s_0:
           s_0, _ = MF.random_sa(value=True) 
        if not a_0:
            _, a_0 = MF.random_sa(value=True)
        episode = MF.generate_episode(s_0, a_0, π, max_steps)
        
        sar = np.array(episode)
        s, a, r = sar[:,0], sar[:,1], sar[:,2]
        
        s = s.astype(int)
        a = a.astype(int)

        T = s.shape[0]
        for t in range(T):
            f_step(s, a, r, t, T, n, v, q, γ, α, gammatron, π)
            # episode is already set so next step is not generated
            # via a greedy strategy, each episode generation is greedy
            if optimize:  
                # in/out-place update for current and next episode
                # off policy without importance weighting
                π.update_policy(q, s[t]) 
        
        n_episode += 1

        if sample_step and n_episode % sample_step == 0:
            samples.append(get_sample(MF, v, q, π, n_episode, optimize))
    
    return v, q, samples


def _td_dq_step(s, a, r, t, T, n, v1, q1, v2, q2, γ, α, gammatron, π):
    '''td step update'''
    s_t, a_t, rr = s[t], a[t], r[t:t+n]
    G = np.dot(gammatron[:rr.shape[0]], rr)
    G_v, G_q = G, G
    if t + n < T:
        G_v = G_v + (γ**n) * v2[s[t+n]]
        G_q = G_q + (γ**n) * q2[s[t+n], np.argmax(q1[s[t+n]])]

    v1[s_t] = v1[s_t] + α * (G_v - v1[s_t])
    q_key = (s_t, a_t)
    q1[q_key] = q1[q_key] + α * (G_q - q1[q_key])


def _double_q(MF, s_0, a_0, n, alpha, n_episodes, max_steps, optimize, 
    method, sample_step):

    π, α, γ = MF.policy, alpha, MF.gamma
    gammatron = np.array([γ**i for i in range(n)])
    
    v1, q1 = MF.init_vq()
    v2, q2 = MF.init_vq()
    v, q = MF.init_vq()

    samples = []
    n_episode = 0
    while n_episode < n_episodes:
        s_0, a_0 = _set_s0_a0(MF, s_0, a_0)
        episode = MF.generate_episode(s_0, a_0, π, max_steps)
        
        sar = np.array(episode)
        s, a, r = sar[:,0], sar[:,1], sar[:,2]
        
        s = s.astype(int)
        a = a.astype(int)

        T = s.shape[0]
        for t in range(T):
            if np.random.rand() < 0.5:
                _td_dq_step(s, a, r, t, T, n, v1, q1, v2, q2, γ, α, gammatron, π)
            else:
                _td_dq_step(s, a, r, t, T, n, v2, q2, v1, q1, γ, α, gammatron, π)

            v = (v1 + v2)/2
            q = (q1 + q2)/2
            
            if optimize:  
                π.update_policy(q, s[t])
        
        n_episode += 1

        if sample_step and n_episode % sample_step == 0:
            samples.append(get_sample(MF, v, q, π, n_episode, optimize))
    
    return v, q, samples


def _tdn_on(MF, s_0, a_0, n, alpha, n_episodes, max_steps, optimize,
    method, sample_step):
    '''N-temporal differences algorithm for learning.
    
    Super slow and inefficient, but readable and replicated exactly
    from sutton's n-step SARSA
    '''
    π, α, γ = MF.policy, alpha, MF.gamma
    gammatron = np.array([γ**i for i in range(n)])

    v, q = MF.init_vq()

    samples = []
    n_episode = 0
    while n_episode < n_episodes:
        s_0, a_0 = _set_s0_a0(MF, s_0, a_0)

        s = MF.states.get_index(s_0)
        a = MF.actions.get_index(a_0)
        T = int(max_steps)
        R, A, S, G = [], [a], [s], 0 
        for t in range(T):
            if t < T:
                (s, r), end = MF.step_transition(s, a)
                R.append(r)
                S.append(s)
                if end:
                    T = t + 1
                else:
                    a = π(s)
                    A.append(a)
            
            tau = t - n + 1
            if tau >= 0:
                rr = np.array(R[tau:min(tau+n, T)])
                G = gammatron[:rr.shape[0]].dot(rr)
                G_v, G_q = G, G
                if tau + n < T:
                    G_v = G_v + γ**n * v[S[tau+n]]
                    G_q = G_q + γ**n * q[S[tau+n], A[tau+n]]
                
                s_t = S[tau]
                a_t = A[tau]
                v[s_t] = v[s_t] + α * (G_v - v[s_t])
                q[(s_t, a_t)] = q[(s_t, a_t)] + α * (G_q - q[(s_t, a_t)])
                
                π.update_policy(q, s_t)

            if tau == T - 1:
                break

        if n_episode % sample_step == 0:
            samples.append(get_sample(MF, v, q, π, n_episode, optimize))
        n_episode += 1

    return v, q, samples


METHOD_MAP = {
    'sarsa_on': _tdn_on,
    'sarsa': _tdn_onoff,
    'qlearning': _tdn_onoff,
    'expected_sarsa': _tdn_onoff, 
    'dqlearning': _double_q
}


METHODS = METHOD_MAP.keys()


def n_tree_backup(states: Sequence[Any], actions: Sequence[Any], transition: Transition,
    state_0: Any=None, action_0: Any=None, gamma: float=1.0, n: int=1, 
    alpha: float=0.05, n_episodes: int=MAX_ITER, policy: ModelFreePolicy=None, 
    eps: float=None, optimize: bool=False, samples: int=1000, max_steps: int=MAX_STEPS
    ) -> Tuple[VQPi, Samples]:
    '''N-temporal differences algorithm.

    Temporal differences algorithm for estimating the value function of a
    policy, improve it and analyze it.

    Parameters
    ----------
    states : Sequence[Any]
    actions : Sequence[Any]
    state_0 : Any, optional
        Initial state, by default None (random)
    action_0 : Any, optional
        Initial action, by default None (random)
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
    samples : Tuple[int, List[Vpi], List[Qpi], List[ModelFreePolicy]] 
        Samples taken during the simulation if any. The first element is the
        index of the iteration, the second is the value function, the third is
        the action-value function and the fourth is the TODO:.

    Raises
    ------
    TransitionException: Ill defined transitions.
    '''    
    policy = _set_policy(policy, eps, actions, states)

    _typecheck_all(tabular_idxs=[states,actions], transition=transition,
        constants=[gamma, n, alpha, n_episodes, samples, max_steps], 
        booleans=[optimize], policies=[policy])

    sample_step = _get_sample_step(samples, n_episodes)

    model = ModelFree(states, actions, transition, gamma=gamma, policy=policy)  
    
    v, q, samples = _n_tree_backup(model, state_0, action_0, n, alpha, n_episodes,
        max_steps, optimize, sample_step)
    
    return VQPi((v, q, policy)), samples


def _n_tree_backup(MF, s_0, a_0, n, alpha, n_episodes, max_steps, 
    optimize, sample_step):

    π, α, γ = MF.policy, alpha, MF.gamma

    v, q = MF.init_vq()

    samples = []
    n_episode = 0
    while n_episode < n_episodes:
        s_0, a_0 = _set_s0_a0(MF, s_0, a_0)

        s = MF.states.get_index(s_0)
        a = MF.actions.get_index(a_0)
        T = int(max_steps)
        R, A, S, G = [], [a], [s], 0 
        
        for t in range(T):
            if t < T:
                (s, r), end = MF.step_transition(s, a)
                R.append(r)
                S.append(s)
                if end:
                    T = t + 1
                else:
                    _, a = MF.random_sa()
                    A.append(a)

            tau = t - n + 1
            if tau >= 0:
                if t + 1 >= T:
                    G = R[-1]
                else:
                    G = R[t] + γ*np.dot(π.pi[s[t]], q[s[t]])

                for k in range(min(t, T-1), tau):
                    G = R[k-1] + γ*np.dot(π.pi[s[k-1]], q[s[k-1]]) + \
                        γ*π.pi[s[k-1],A[k-1]]*(G-q[s[k-1], A[k-1]])
                
                q[S[tau], A[tau]] = q[S[tau], A[tau]] + α[G-q[S[tau], A[tau]]] 
                
                if optimize:
                    π.update_policy(q, S[tau])

            if tau == T - 1:
                break

        if n_episode % sample_step == 0:
            samples.append(get_sample(MF, v, q, π, n_episode, optimize))
        n_episode += 1

    return v, q, samples