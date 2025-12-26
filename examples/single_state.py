import numpy as np
import matplotlib.pyplot as plt

plt.style.use("dark_background")

from rl import off_policy_mc, ModelFreePolicy

# Define model free
states = [0]
actions = ["left", "right"]


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
vqpi_ord, samples_ord = off_policy_mc(
    states,
    actions,
    single_state_transition,
    policy=pi,
    b=b,
    ordinary=True,
    first_visit=True,
    gamma=1.0,
    n_episodes=1e4,
)

vqpi_w, samples_w = off_policy_mc(
    states,
    actions,
    single_state_transition,
    policy=pi,
    b=b,
    ordinary=False,
    first_visit=True,
    gamma=1.0,
    n_episodes=1e4,
)


# Plot!
vords = [v[1].values()[0] for v in samples_ord[1:]]
vw = [v[1].values()[0] for v in samples_w[1:]]
idxs = [v[0] for v in samples_ord[1:]]

plt.figure(figsize=(10, 5))
plt.plot(idxs, vords, label="Ordinary Importance Sampling")
plt.plot(idxs, vw, label="Weighted Importance Sampling")
plt.xlabel("No episodes")
plt.ylabel("v(0)")
plt.xscale("log")
plt.legend(loc=1)
plt.show()
