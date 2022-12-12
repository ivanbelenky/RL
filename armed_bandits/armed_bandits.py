'''
RL

Copyright © 2023 Iván Belenky

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files, to deal without restriction, 
including without limitation the rights to  use, copy, modify, merge, publish, 
distribute, sublicense, and/or sell copies of this.
'''

from typing import List

import numpy as np
import numpy.random as rnd

from policy import Policy
from armed_bandits.policy import EGREEDY
from reward import GaussianRewardGenerator, RewardGenerator


GAUSSIAN = [GaussianRewardGenerator(rnd.random(), rnd.random()) for _ in range(10)]
NGAMES = 1
NSTEPS = 1000


class MultiArmedBandit:
    def __init__(
        self, 
        k: int = 10, 
        reward_generators: List[RewardGenerator] = GAUSSIAN, 
        n_games: int = NGAMES,
        policy: Policy = EGREEDY):
        
        self.k = k
        self.reward_generators = reward_generators
        self.N = n_games
        self.histories = []
        self.reward_history = []
        self.action_history = []
        self.policy = policy
        self.ground_truth = np.argmax([
            rg.mean() for rg in self.reward_generators])

    def step(self, action: int) -> float:    
        reward = self.reward_generators[action].generate()
        self.reward_history.append(reward)
        self.action_history.append(action)

        return reward
    
    def reset(self) -> None:
        self.action_history = []
        self.reward_history = []

    def evaluate_policy(self) -> List[float]:
        for _ in range(self.N):
            self.step(self.policy())

        return self.reward_history

    def update_policy(self) -> None:
        for _ in range(self.N):
            action = self.policy()
            reward = self.step(action)
            self.policy.update_policy(action, reward)

    def best_action_percentage(self) -> None:
        ah = np.array(self.action_history)
        n = ah[ah==self.ground_truth]
        return n.shape[0]/ah.shape[0]

    

