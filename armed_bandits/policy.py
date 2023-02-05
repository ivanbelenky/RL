"""
RL - Copyright © 2023 Iván Belenky @Leculette
"""


import numpy as np
import numpy.random as rnd

from utils import Policy, RewardGenerator



GAUSSIAN = [RewardGenerator('normal', rnd.random(), rnd.random()) for _ in range(10)]
NGAMES = 1
NSTEPS = 1000


class EpsilonGreedyBanditPolicy(Policy):
    def __init__(self, k: int=10, epsilon: float=0.1, offset: float=0.0):
        self.k = k
        self.eps = epsilon
        self.offset = offset
        self.q_values = np.zeros(k) + self.offset
        self.N = np.zeros(k)

    def __call__(self) -> int:
        if rnd.random() < self.eps:
            return rnd.randint(self.k)
        
        return np.argmax(self.q_values) 

    def update_policy(self, action: int, reward: float) -> None:
        N = self.N[action] + 1
        self.N[action] = N
        
        Q = self.q_values[action]
        R = reward
        Qnew = Q + 1/N*(R-Q)

        self.q_values[action] = Qnew


class UCBPolicy(Policy):
    def __init__(self, k: int=10, c: float=2.0, offset: float=0.0):
        self.k = k
        self.c = c
        self.offset = offset
        self.q_values = np.zeros(k) + self.offset
        self.N = np.zeros(k)
        self.init_counter = 0
    
    def __call__(self):
        if self.init_counter < self.k:
            action_index = self.init_counter
            self.init_counter += 1
            return action_index
        
        return np.argmax(
            self.q_values + self.c*np.sqrt(np.log(np.sum(self.N))/self.N))

    def update_policy(self, action, reward):
        N = self.N[action] + 1
        self.N[action] = N
        
        Q = self.q_values[action]
        R = reward
        Qnew = Q + 1/N*(R-Q)

        self.q_values[action] = Qnew


class AlphaEpsilonGreedyBanditPolicy(EpsilonGreedyBanditPolicy):
    def __init__(self, k: int=10, epsilon: int=0.1, alpha: int=0.1):
        super().__init__(k, epsilon)
        self.alpha = alpha
        
    def update_policy(self, action, reward):
        Q = self.q_values[action]
        R = reward
        Qnew = Q + self.alpha*(R-Q)

        self.q_values[action] = Qnew


class GradientPolicy(Policy):
    def __init__(self, k: int=10, alpha: float=0.1):
        self.k = k
        self.alpha = alpha
        self.rewards = []
        self.H = np.zeros(k)
        self.Pr = np.zeros(k)
    
    def __call__(self) -> int:
        self.Pr = np.exp(self.H)/np.sum(np.exp(self.H))
        return np.random.choice(self.k, p=self.Pr)

    def update_policy(self, action, reward) -> None:
        self.H -= self.alpha*(reward - np.mean(self.rewards))*self.Pr
        self.H[action] += self.alpha*(reward - np.mean(self.rewards))        
        self.rewards.append(reward)


EGREEDY = EpsilonGreedyBanditPolicy()