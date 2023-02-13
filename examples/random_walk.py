import numpy as np

from rl import tdn, alpha_mc

states = [1,2,3,4,5]
actions = ['?'] #there are no actions :D

def random_walk(state, action):
    go_right = np.random.random() > 0.5 
    if go_right:
        if 1+state <= 5:
            return (1+state, 0), False
        return (state, 1), True
    else:
        if state-1 == 0:
            return (state, 0), True
        return (state-1, 0), False


_, samples_mc_01 = alpha_mc(states, actions, random_walk, alpha=0.01,
    first_visit=True, n_episodes=200)

# ... 