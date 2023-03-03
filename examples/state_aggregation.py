import numpy as np
import matplotlib.pyplot as plt
plt.style.use("dark_background")

from rl import gradient_mc, semigrad_tdn
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
    approximator = LinearApproximator(k=10, fs=10, basis=state_aggregator)
    vqpi, samples  = gradient_mc(states, actions, random_walk, approximator,
        max_steps=1E5, alpha=2*10E-5)
    vqpi, samples  = gradient_mc(states, actions, random_walk, approximator, n_episodes=3E4,
                            max_steps=1E5, alpha=2*10E-5)
    vqpi_td, samples_td = semigrad_tdn(states, actions, random_walk, approximator, n_episodes=3E4,
                            max_steps=1E5, alpha=2*10E-5)

    vpi_true = 2/1000*np.arange(1001) - 1
    plt.figure(figsize=(10,5))
    plt.plot(vpi_true[1:], label='True value')
    plt.plot(vqpi_td[0][1:], label='semigrad-tdn')
    plt.plot(vqpi[0][1:], label='gradient-mc')
    plt.legend(loc=4)   