import numpy as np
import matplotlib.pyplot as plt
plt.style.use("dark_background")

from rl import semigrad_tdn, gradient_mc, IHT, tiles
from rl.approximators import LinearApproximator


ACTIONS = (-1, 0, 1)
X_BOUND = [-1.2, 0.5]
V_BOUND = [-0.07, 0.07]


def mountain_car(state, action):
    x, v = state
    new_v = v + 0.001*action - 0.0025*np.cos(3*x)
    new_x = x + new_v

    if new_x < X_BOUND[0]:
        new_x = X_BOUND[0]
        new_v = 0 
        return ((new_x, new_v), -1), False
    elif new_x > X_BOUND[1]:
        return (state, 10), True
    else:
        new_v = np.clip(new_v, V_BOUND[0], V_BOUND[1])
        return ((new_x, new_v), -1), False


def state_generator():
    x = np.random.uniform(X_BOUND[0], X_BOUND[1])
    v = np.random.uniform(V_BOUND[0], V_BOUND[1])
    return (x, v)


iht_s = IHT(1000)
iht_sa = IHT(4096)


def state_action_aggregator(sa):
    s, a = sa
    x, v = s
    f = np.zeros(4096)
    tile = tiles(iht_sa, 8, [8*x/(0.5+1.2), 8*v/(0.07+0.07)], [a])
    f[tile] = 1
    return f


def state_aggregator(state):
    x, v = state
    f = np.zeros(1000)
    tile = tiles(iht_s, 8, [8*x/(0.5+1.2), v/(0.07+0.07)])
    f[tile] = 1
    return f


if __name__ == "__main__":
    vhat = LinearApproximator(fs=1000, basis=state_aggregator)
    qhat = LinearApproximator(fs=4096, basis=state_action_aggregator) 

    vqpi_mc, samples_mc = gradient_mc(mountain_car, state_generator, ACTIONS,
        vhat, q_hat=qhat, state_0=(0,0), action_0=0, n_episodes=500, 
        max_steps=1E4, alpha=0.1/8, eps=0.1, optimize=True)
    