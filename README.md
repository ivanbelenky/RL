# Reinforcement Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/YannDubs/disentangling-vae/blob/master/LICENSE) 
[![Python 3.5+](https://img.shields.io/badge/python-3.5+-blue.svg)](https://www.python.org/downloads/release/python-360/)

## Installation 

### setup.py
```sh
$ python setup.py install
```

### pypi (incoming)

```sh
pip install mypyrl 
```


# Overview

This repository contains code that implements algorithms and models from Sutton's book on reinforcement learning. The book, titled "Reinforcement Learning: An Introduction," is a classic text on the subject and provides a comprehensive introduction to the field.

The code in this repository is organized into several modules, each of which covers differents topics.


# Methods

- [x] Multi Armed Bandits
  - [x] Epsilon Greedy
  - [x] Optimistic Initial Values
  - [x] Gradient 
  - [x] α (non stationary)
- [x] Model Based
  - [x] Policy Evaluation
  - [x] Policy Iteration
  - [x] Value Iteration
- [x] Monte Carlo estimation and control
  - [x] First-visit α-MC
  - [x] Every-visit α-MC  
  - [x] MC with Exploring Starts
  - [x] Off-policy MC, ordinary and weighted importance sampling   
- [x] Temporal Difference
  - [x] TD(n) estimation 
  - [x] n-step SARSA 
  - [x] n-step Q-learning
  - [x] n-step Expected SARSA
  - [x] double Q learning
  - [x] n-step Tree Backup 
- [x] Planning
  - [x] Dyna-Q/Dyna-Q+
  - [x] Prioritized Sweeping
  - [x] Trajectory Sampling
  - [x] MCTS
- [ ] On-policy Prediction 
  - [x] Linear SGD/semi-SGD
  - [ ] ANN
  - [ ] Least-Squares TD
  - [ ] Kernel-based
- [ ] On-policy Control 
  - [x] Episodic semi-gradient
  - [x] Semi-gradient n-step Sarsa
  - [ ] Differential Semi-gradient n-step Sarsa

<br>

All model free solvers will work just by defining `states` `actions` and a `trasition` function. Transitions are defined as a function that takes a state and an action and returns a tuple of the next state and the reward. The transition function also returns a boolean indicating whether the episode has terminated.

```python
states: Sequence[Any]
actions: Sequence[Any]
transtion: Callable[[Any, Any], Tuple[Tuple[Any, float], bool]]
```

# Examples 

**Single State Infinite Variance Example 5.5**

![](https://github.com/ivanbelenky/RL/blob/master/assets/images/single_state.png)


```python
from mypyrl import off_policy_mc, ModelFreePolicy

states = [0]
actions = ['left', 'right']

def single_state_transition(state, action):
    if action == 'right':
        return (state, 0), True
    if action == 'left':
        threshold = np.random.random()
        if threshold > 0.9:
            return (state, 1), True
        else:
            return (state, 0), False

b = ModelFreePolicy(actions, states) #by default equiprobable
pi = ModelFreePolicy(actions, states)
pi.pi[0] = np.array([1, 0])

# calculate ordinary and weighted samples state value functions
vqpi_ord, samples_ord = off_policy_mc(states, actions, single_state_transition,
    policy=pi, b=b, ordinary=True, first_visit=True, gamma=1., n_episodes=1E4)

vqpi_w, samples_w = off_policy_mc(states, actions, single_state_transition, 
    policy=pi, b=b, ordinary=False, first_visit=True, gamma=1., n_episodes=1E4)
```

![](https://github.com/ivanbelenky/RL/blob/master/assets/images/ordinary_vs_weighted.png)

<br>

**Monte Carlo Tree Search maze solving plot**

```python
s = START_XY
budget = 500
cp = 1/np.sqrt(2)
end = False
max_steps = 50
while not end:
    action, tree = mcts(s, cp, budget, obstacle_maze, action_map, max_steps, eps=1)
    (s, _), end = obstacle_maze(s, action)

tree.plot()
```

![](https://github.com/ivanbelenky/RL/blob/master/assets/images/uct.png)

<br>

# Contributing

While the code in this package provides a basic implementation of the algorithms from the book, it is not necessarily the most efficient or well-written. If you have suggestions for improving the code, please feel free to open an issue.

In addition to the code, there are useful Jupyter notebooks [here](https://www.google.com) that provide examples of how to use the implemented algorithms and models. Notebooks are usually implementations of examples present on suttons book.

Overall, this package provides a valuable resource for anyone interested in learning about reinforcement learning and implementing algorithms from scratch. By no means prod ready.
