# Reinforcement Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/YannDubs/disentangling-vae/blob/master/LICENSE) 
[![Python 3.5+](https://img.shields.io/badge/python-3.5+-blue.svg)](https://www.python.org/downloads/release/python-360/)

## Installation (near future)

`pip install mypyrl`


## Overview

This repository contains code that implements algorithms and models from Sutton's book on reinforcement learning. The book, titled "Reinforcement Learning: An Introduction," is a classic text on the subject and provides a comprehensive introduction to the field.

The code in this repository is organized into several modules, each of which covers differents topics.


# Tabular Solution Methods

## Multi Armed Bandits

- [x] Epsilon Greedy
- [x] Optimistic Initial Values
- [x] Gradient 
- [x] α (non stationary)

## Model Based methods
- [x] Policy Evaluation
- [x] Policy Iteration
- [x] Value Iteration


## Model Free methods
 
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

## Planning and Learning Methods

- [x] Dyna-Q/Dyna-Q+
- [x] Prioritized Sweeping
- [ ] Trajectory Sampling
- [ ] MCTS 
- [ ] RTDP


# Approximate Solution Methods

- [ ] On-policy Prediction
  - [x] Linear SGD/semi-SGD
  - [ ] ANN
  - [ ] Least-Squares TD
  - [ ] Memory-based
  - [ ] Kernel-based
- [ ] On-policy Control 
  - [ ] Episodic semi-gradient
  - [ ] Semi-gradient n-step Sarsa
  - [ ] Differential Semi-gradient n-step Sarsa


All solvers will work just by defining `states` `actions` and a `trasition` function. Transitions are defined as a function that takes a state and an action and returns a tuple of the next state and the reward. The transition function also returns a boolean indicating whether the episode has terminated.

```python
states: Sequence[Any]
actions: Sequence[Any]
transtion: Callable[[Any, Any], Tuple[Tuple[Any, float], bool]]
```

### Examples 

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

b = ModelFreePolicy(actions, states) #by default 1 half
pi = ModelFreePolicy(actions, states)
pi.pi[0] = np.array([1, 0])

# calculate ordinary and weighted samples state value functions
vqpi_ord, samples_ord = off_policy_mc(states, actions, single_state_transition,
    policy=pi, b=b, ordinary=True, first_visit=True, gamma=1., n_episodes=1E4)

vqpi_w, samples_w = off_policy_mc(states, actions, single_state_transition, 
    policy=pi, b=b, ordinary=False, first_visit=True, gamma=1., n_episodes=1E4)
```

![](https://github.com/ivanbelenky/RL/blob/master/assets/images/ordinary_vs_weighted.png)

# Contributing

While the code in this package provides a basic implementation of the algorithms from the book, it is not necessarily the most efficient or well-written. If you have suggestions for improving the code, please feel free to open an issue.

In addition to the code, there are useful Jupyter notebooks ![here](https://404.com) that provide examples of how to use the implemented algorithms and models. Notebooks are usually implementations of examples present on suttons book.

Overall, this package provides a valuable resource for anyone interested in learning about reinforcement learning and implementing algorithms from scratch. By no means prod ready.
