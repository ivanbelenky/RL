from typing import (
    Tuple, 
    Sequence,  
    Any
)

from tqdm import tqdm
import numpy as np
from numpy.linalg import norm as lnorm

from rl.solvers.model_free import (
    get_sample, 
    _set_s0_a0,
    _set_policy,
)
from rl.model_free import ModelFree, ModelFreePolicy
from rl.utils import (
    UCTree,
    UCTNode,
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
        int(n_episodes), max_steps, sample_step)

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
    for n_episode in tqdm(range(n_episodes), desc='Dyna-Q', unit='episodes'):
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

    _typecheck_all(tabular_idxs=[states, actions], transition=transition,
        constants=[gamma, theta, n, alpha, n_episodes, samples, max_steps], 
        booleans=[plus], policies=[policy])

    # check ranges
    
    sample_step = _get_sample_step(samples, n_episodes)

    model = ModelFree(states, actions, transition, gamma=gamma, policy=policy)
    v, q, samples = _priosweep(model, state_0, action_0, n, alpha, theta, 
        int(n_episodes), max_steps, sample_step)

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

    samples, current_t = [], 0
    for n_episode in tqdm(range(n_episodes), desc='priosweep', unit='episodes'):
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

    return v, q, samples


def t_sampling(states: Sequence[Any], actions: Sequence[Any], transition: Transition,
    state_0: Any=None, action_0: Any=None, gamma: float=1.0,
    n_episodes: int=MAX_ITER, policy: ModelFreePolicy=None, eps: float=None, 
    samples: int=1000, optimize: bool=False, max_steps: int=MAX_STEPS
    ) -> Tuple[VQPi, Samples]:
    '''
    TODO: docs
    '''
    policy = _set_policy(policy, eps, actions, states)

    _typecheck_all(tabular_idxs=[states,actions], transition=transition,
        constants=[gamma, n_episodes, samples, max_steps], 
        booleans=[optimize], policies=[policy])

    # TODO: check ranges
    
    sample_step = _get_sample_step(samples, n_episodes)

    model = ModelFree(states, actions, transition, gamma=gamma, policy=policy)
    v, q, samples = _t_sampling(model, state_0, action_0, int(n_episodes),
        optimize, max_steps, sample_step)

    return VQPi((v, q, policy)), samples


def _t_sampling(MF, s_0, a_0, n_episodes, optimize, 
    max_steps, sample_step):
    
    π, γ = MF.policy, MF.gamma
    v, q = MF.init_vq()
    
    S, A = MF.states.N, MF.actions.N
    n_sas = np.zeros((S, A, S), dtype=int) # p(s'|s,a) 
    model_sar = np.zeros((S, A, S), dtype=float) # r(s,a,s') deterministic reward

    samples = []
    for n_episode in tqdm(range(n_episodes), desc='Trajectory Sampling', unit='episodes'):
        s, a = _set_s0_a0(MF, s_0, a_0)
        a_ = MF.actions.get_index(a)
        s = MF.states.get_index(s)
        
        for _ in range(int(max_steps)):
            (s_, r), end = MF.step_transition(s, a_) # real next state
            
            n_sas[s, a, s_] += 1
            model_sar[s, a, s_] = r # assumes deterministic reward

            # p_sas is the probability of transitioning from s to s'
            p_sas = n_sas[s,a]/np.sum(n_sas[s, a]) 
            next_s_mask = np.where(p_sas)[0]
            max_q = np.max(q[next_s_mask, :], axis=1)
            r_ns = model_sar[s, a, next_s_mask]
            p_ns = p_sas[next_s_mask]
            
            q[s, a] = np.dot(p_ns, r_ns + γ*max_q)

            π.update_policy(q, s)
            a_ = π(s_)
            s = s_

            if end:
                break

        if n_episode % sample_step == 0:
            samples.append(get_sample(MF, v, q, π, n_episode, optimize))
    
    return v, q, samples


def rtdp():
    raise NotImplementedError



def _best_child(v, Cp):
    actions = np.array(list(v.children.keys()))
    qs = np.array([v.children[a].q for a in actions])
    ns = np.array([v.children[a].n for a in actions])
    ucb = qs/ns + Cp*np.sqrt(np.log(v.n)/ns)
    return v.children[actions[np.argmax(ucb)]]
    

def _expand(v, transition, actions):
    a = np.random.choice(list(actions))
    (s_, _), end =  transition(v.state, a)
    v_prime = UCTNode(s_, a, 0, 1, v, end)
    v.children[a] = v_prime
    return v_prime


def _tree_policy(tree, Cp, transition, action_map, eps):
    v = tree.root
    while not v.is_terminal:
        actions = action_map(v.state)
        took_actions = v.children.keys()
        unexplored = set(actions) - set(took_actions)
        if unexplored:
            return _expand(v, transition, unexplored)
        v = _best_child(v, Cp)
    return v


def _default_policy(v_leaf, transition, action_map, max_steps):
    step, r = 0, 0
    s = v_leaf.state

    if v_leaf.is_terminal:
        return r

    while step < max_steps:
        actions = action_map(s)
        a = np.random.choice(actions)
        (s, _r), end = transition(s, a)
        r += _r
        if end:
            return r
        step += 1
    return r
        

def _backup(v_leaf, delta):
    v = v_leaf
    while not v:
        v.n += 1
        v.q += delta
        v = v.parent


def mcts(s0, Cp, budget, transition, action_map, max_steps, eps=0.1, tree=None):
    '''
    Effectively implementing the UCT search algorithm
    '''
    s = s0
    if not tree:
        tree = UCTree(s, Cp)
    for _ in tqdm(range(budget)):
        v_leaf = _tree_policy(tree, Cp, transition, action_map, eps=eps)
        delta = _default_policy(v_leaf, transition, action_map, max_steps)
        _backup(v_leaf, delta)
        
    v_best = _best_child(tree.root, 0)
    return v_best.action, UCTree(v_best)
