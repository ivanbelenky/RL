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


def state_generator():
    pos = np.random.randint(1,1000)
    group = pos//101
    return (group, pos)


if __name__ == "__main__":
    approximator_mc = LinearApproximator(k=10, fs=10, basis=state_aggregator)
    approximator_td = LinearApproximator(k=10, fs=10, basis=state_aggregator)
    
    vqpi_mc, samples_mc = gradient_mc(random_walk, state_generator, actions, 
                                      approximator_mc, n_episodes=3E4, max_steps=1E5, 
                                      alpha=2*10E-5)
    vqpi_td, samples_td = semigrad_tdn(random_walk, state_generator, actions, 
                                       approximator_td, n_episodes=3E4, max_steps=1E5,
                                       alpha=2*10E-5)
    
    vhat_mc = vqpi_mc[0]
    vhat_td = vqpi_td[0]

    state_sample =[(pos//101, pos) for pos in np.arange(1001)]
    vpi_true = 2/1000*np.arange(1001) - 1
    vpi_mc = np.array([vhat_mc(s) for s in state_sample])
    vpi_td = np.array([vhat_td(s) for s in state_sample])

    plt.figure(figsize=(10,5))
    plt.plot(vpi_true, label='True value')
    plt.plot(vpi_td, label='semigrad-tdn')
    plt.plot(vpi_mc, label='gradient-mc')
    plt.legend(loc=4)