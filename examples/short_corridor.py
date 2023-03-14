import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

from tqdm import tqdm 

from rl import reinforce_mc
from rl.approximators import ModelFreeTL, LinearApproximator

actions = ['left', 'right']

def short_corridor(state, action):
    go_right = (action == 'right')
    if state == 1:
        if go_right:
            return (2, -1), False
        return (1, 0), False
    if state == 2:
        if go_right:
            return (1, -1), False 
        return (3, -1), False
    if state == 3:
        if go_right:
            return (state, 0), True
        return (2, -1), False


def random_state():
    return np.random.randint(1,4)


def state_action_aggregator(sa):
    _, a = sa
    right = (a == 'right')
    if right:
        return np.array([1., 0.])
    return np.array([0., 1.])


if __name__ == "__main__":
    pi_hat = LinearApproximator(fs=2, basis=state_action_aggregator)
    pi_hat.w = np.array([-5E-8,1E-8])
    pi, samples = reinforce_mc(short_corridor, random_state, pi_hat, actions, state_0=1, alpha=2E-8,
                            gamma=1, n_episodes=1000, max_steps=1000, samples=100, tol=1/np.inf)
    model = ModelFreeTL(short_corridor, random_state, pi, gamma=1)

    SMOOTH = 100
    rewards = []
    for i in tqdm(range(SMOOTH)):
        _rewards = []
        for policy in samples:
            a0 = policy(1)
            episode = model.generate_episode(1, a0, policy=policy, max_steps=1000)
            sar = np.array(episode)
            _rewards.append(sar[:,2].astype(int).sum())
        rewards.append(_rewards)
    
    plt.plot(np.array(rewards).mean(axis=0))
    plt.show()
