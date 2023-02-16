from typing import (
    Tuple, 
    Sequence,  
    Any
)

import numpy as np
from numpy.linalg import norm as lnorm

from rl.solvers.model_free import (
    get_sample, 
    _set_s0_a0,
    _set_policy,
)
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


def dynaq(states: Sequence[Any], actions: Sequence[Any], transition: Transition,
    state_0: Any=None, action_0: Any=None, gamma: float=1.0, kappa: float=0.01, 
    n: int=1, plus: bool=False, alpha: float=0.05, n_episodes: int=MAX_ITER,
    policy: ModelFreePolicy=None, eps: float=None, samples: int=1000,
    max_steps: int=MAX_STEPS) -> Tuple[VQPi, Samples]:
    '''
    TODO: docs
    '''
    policy = _set_policy(policy, eps, actions, states)

    _typecheck_all(tabular_idxs=[states,actions], transition=transition,
        constants=[gamma, kappa, n, alpha, n_episodes, samples, max_steps], 
        booleans=[plus], policies=[policy])

    # check ranges
    
    sample_step = _get_sample_step(samples, n_episodes)

    model = ModelFree(states, actions, transition, gamma=gamma, policy=policy)
    v, q, samples = _dyna_q(model, state_0, action_0, n, alpha, kappa, plus,
        n_episodes, max_steps, sample_step)

    return VQPi((v, q, policy)), samples


def _dyna_q(MF, s_0, a_0, n, alpha, kappa, plus, n_episodes, max_steps,
    sample_step):

    π, α, γ, κ = MF.policy, alpha, MF.gamma, kappa

    v, q = MF.init_vq()
    
    S, A = MF.states.N, MF.actions.N
    model_sas = np.zeros((S, A), dtype=int)
    model_sar = np.zeros((S, A), dtype=float)
    times_sa = np.zeros((S, A), dtype=int)

    samples = []
    current_t = 0
    n_episode = 0
    while n_episode < n_episodes:
        s_0, _ = _set_s0_a0(MF, s_0, None)

        s = MF.states.get_index(s_0)
        T = int(max_steps)
        
        for t in range(T):
            a = π(s)
            (s_, r), end = MF.step_transition(s, a) # real next state
            q[s, a] = q[s, a] + α*(r + γ*np.max(q[s_]) - q[s, a])
            
            times_sa[s, a] = current_t

            # assuming deterministic environment
            model_sas[s, a] = s_
            model_sar[s, a] = r
            
            current_t += 1

            for _ in range(n):
                rs, ra = MF.random_sa()
                s_m = model_sas[rs, ra] # model next state
                r_ = model_sar[rs, ra]
                R = r_
                if plus:
                    tau = current_t - times_sa[rs, ra]
                    R = R + κ*np.sqrt(tau)
                q[rs, ra] = q[rs, ra] + α*(R + γ*np.max(q[s_m]) - q[rs, ra])
            
            π.update_policy(q, s_)
            s = s_ # current state equal next state
            if end:
                break 
        
        if n_episode % sample_step == 0:
            samples.append(get_sample(MF, v, q, π, n_episode, True))
        n_episode += 1

    return v, q, samples


def priosweep(states: Sequence[Any], actions: Sequence[Any], transition: Transition,
    state_0: Any=None, action_0: Any=None, gamma: float=1.0, theta: float=0.01, 
    n: int=1, plus: bool=False, alpha: float=0.05, n_episodes: int=MAX_ITER,
    policy: ModelFreePolicy=None, eps: float=None, samples: int=1000,
    max_steps: int=MAX_STEPS) -> Tuple[VQPi, Samples]:
    '''
    TODO: docs
    '''
    policy = _set_policy(policy, eps, actions, states)

    _typecheck_all(tabular_idxs=[states,actions], transition=transition,
        constants=[gamma, theta, n, alpha, n_episodes, samples, max_steps], 
        booleans=[plus], policies=[policy])

    # check ranges
    
    sample_step = _get_sample_step(samples, n_episodes)

    model = ModelFree(states, actions, transition, gamma=gamma, policy=policy)
    v, q, samples = _priosweep(model, state_0, action_0, n, alpha, theta, 
        n_episodes, max_steps, sample_step)

    return VQPi((v, q, policy)), samples


def _priosweep(MF, s_0, a_0, n, alpha, theta, n_episodes, max_steps, 
        sample_step):

    π, α, γ = MF.policy, alpha, MF.gamma
    v, q = MF.init_vq()
    
    P, Pq, θ = 0, PQueue([]), theta 

    S, A = MF.states.N, MF.actions.N
    model_sas = np.zeros((S, A), dtype=int)
    model_sar = np.zeros((S, A), dtype=float)
    times_sa = np.zeros((S, A), dtype=int)

    samples, current_t, n_episode = [], 0, 0
    while n_episode < n_episodes:
        s_0, _ = _set_s0_a0(MF, s_0, None)

        s = MF.states.get_index(s_0)
        T = int(max_steps)
        
        for t in range(T):
            a = π(s)
            (s_, r), end = MF.step_transition(s, a) # real next state
            times_sa[s, a] = current_t
            model_sas[s, a] = s_
            model_sar[s, a] = r

            P = np.abs(r + γ*np.max(q[s_]) - q[s, a])
            if P > θ:
                Pq.push((s, a), P)
 
            current_t += 1

            for _ in range(n):
                if Pq.empty():
                    break

                ps, pa = Pq.pop()
                s_m = model_sas[ps, pa] # model next state
                r_ = model_sar[ps, pa]
                R = r_
                
                q[ps, pa] = q[ps, pa] + α*(R + γ*np.max(q[s_m]) - q[ps, pa])

                # grab all the index where model_sas == s
                mmask = (model_sas == s)
                for ss, aa in zip(*np.where(mmask)):
                    rr = model_sar[ss, aa]
                    P = np.abs(rr + γ*np.max(q[s]) - q[ss, aa])
                    if P > θ:
                        Pq.push((s, a), P)
                
            π.update_policy(q, s_)
            s = s_ # current state equal next state
            if end:
                break 
        
        if n_episode % sample_step == 0:
            samples.append(get_sample(MF, v, q, π, n_episode, True))
        n_episode += 1

    return v, q, samples


def t_sampling():
    raise NotImplementedError


def rtdp():
    raise NotImplementedError


def mcts():
    raise NotImplementedError