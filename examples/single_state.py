from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from rl import ModelFreePolicy, off_policy_mc

plt.style.use("dark_background")

# Define model free
states = [0]
actions = ["left", "right"]

N_EPISODES = 10000


def single_state_transition(state, action):
    if action == "right":
        return (state, 0), True
    if action == "left":
        threshold = np.random.random()
        if threshold > 0.9:
            return (state, 1), True
        else:
            return (state, 0), False


b = ModelFreePolicy(actions, states)  # by default 1 half
pi = ModelFreePolicy(actions, states)
pi.pi[0] = np.array([1, 0])


# calculate ordinary and weighted samples state value functions
_sampling_way_off_policy = partial(
    off_policy_mc,
    states=states,
    actions=actions,
    transition=single_state_transition,
    policy=pi,
    b=b,
    first_visit=True,
    gamma=1.0,
    n_episodes=N_EPISODES,
)

vqpi_ord, samples_ord = _sampling_way_off_policy(ordinary=True)

vqpi_w, samples_w = _sampling_way_off_policy(ordinary=False)


# Plot!
v_ords = [v[1].values()[0] for v in samples_ord[1:]]
v_w = [v[1].values()[0] for v in samples_w[1:]]
idxs = [v[0] for v in samples_ord[1:]]

plt.figure(figsize=(10, 5))
plt.plot(idxs, v_ords, label="Ordinary Importance Sampling")
plt.plot(idxs, v_w, label="Weighted Importance Sampling")
plt.xlabel("Number of episodes")
plt.ylabel("v(0)")
plt.xscale("log")
plt.legend(loc=1)
plt.show()
