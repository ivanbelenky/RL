import numpy as np
import matplotlib.pyplot as plt
plt.style.use("dark_background")

from rl import gradient_mc
from rl.approximators import LinearApproximator


states = [(i//101,i) for i in range(1001)]
actions = ['?'] #there are no actions :D


def random_walk(state, action):
    group, pos = state
    go_right = np.random.random() > 0.5 
    steps = np.random.randint(1,100)

    if go_right:
        if pos+steps <= 1000:
            new_pos = pos+steps
            new_group = new_pos//101
            return ((new_group, new_pos), 0), False
        return (state, 1), True
    else:
        if pos-steps > 0:
            new_pos = pos-steps
            new_group = new_pos//101
            return ((new_group, new_pos), 0), False
        return (state, -1), True
    

def state_aggregator(state):
    group, _ = state
    x = np.zeros(10)
    x[group] = 1
    return x


if __name__ == "__main__":
    approximator = LinearApproximator(k=10, fs_xs=10, basis_xs=state_aggregator)
    vqpi, samples  = gradient_mc(states, actions, random_walk, approximator,
        max_steps=1E5, alpha=2*10E-5)

    plt.plot(vqpi[0])