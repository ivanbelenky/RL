# **Monte Carlo Methods**

_"Monte Carlo methods utilize experience—sample sequences of states, actions, and rewards from actual or simulated interaction with an environment. Learning from actual experience is striking because it requires no prior knowledge of the environment’s dynamics, yet can still attain optimal behavior. Learning from simulated experience is also powerful. Although a model is required, the model need only generate sample transitions, not the complete probability distributions of all possible transitions that is required for dynamic programming (DP). In surprisingly many cases it is easy to generate experience sampled according to the desired probability distributions, but infeasible to obtain the distributions in explicit form."_

So basically if we have a universe in which we can sample stuff, we dont even have to bother with the model. If we want to simulate it, we must create a transition model, and therefore we leave out the complications of building the actual transition probability functions for each state action pair. It is fair to say that given this approach, it is mandatory to tae an episodic approach, otherwise there is no end to the endeavor.

## **Monte Carlo Prediction**

The goal here is to learn the state-value function. The basic idea behind monte carlo is to average the value function across every episode we have at hand. Lets say that we have to find $\color{orange}v_{\pi}(s)$. Each occurence of state $\color{orange}s$ in an episode is called a _visit_ to $\color{orange}s$. 

### **First visit Monte Carlo**

First visit just averages all episodes after the _first visit to each state_ $\color{orange}s$

Given 
- Input: $\color{orange}\pi$
- Initialize
  -  state-value: $\color{orange}v(s)$
  -  returns for each state: $\color{orange}R(s)$ 
-  While some condition is true (tolerance, amount of iterations)
   -  Generate an episode following policy $\color{orange}{\pi \rightarrow  S_{0}, A_{0}, R_{1}, \cdots , S_{T-1}, A_{T-1}, R_{T}}$
   -  $\color{orange}G \leftarrow 0$
   -  loop `episode[::-1]`, i.e. $\color{orange}T-1, T-2, \cdots, 0$
      -  $\color{orange}G \leftarrow \gamma G + R_{t+1}$
      -  if state $\color{orange}s$ not in $\color{orange}[S_0, S_1, \cdots, S_{t-1} ]$:
         -  $\color{orange} R(s) \leftarrow \text{append} \ G$
         -  $\color{orange} v_{\pi}(s) \leftarrow avg(R(s))$  


The implementation is arbitrary. If it gets complicated with the backwards iteration, and the forward iteration, check the natural implementation following the basic principles of the algorithm, isomorphic. 

If a model is not available is particualrly useful to estimate action values, because otherwise it would be difficult to asses what is the best action to take given a state, not knowing what is the space of possible next states. In DP state values is all you want to know, since you have the modl. In MC you dont have the model, so you need to estimate the action values. 

Monte Carlo methods for estimating action values are exactly the same as the above, but instead of averaging the returns for each state, you average the returns for each state action pair. 

Major drawback: you may not visit all state action pairs. Edge example, if $\color{orange}\pi$ is deterministic. Exploration vs exploitation, and/or maintaining exploration. Solutions
- `exploring starts`: make sure that all episodes start ant a specific state-action pair, and that every pair has a nonzero probability of being selected as the start
- `stochastic selection of all possible action`: select sometimes a random action, remember epsilon policies with Multi Armed Bandits. 

### **Monte Carlo Control**

We can do basically the same as before: evaluate, improve, evaluate, improve, etc... but this time we are not able to make a policy greedy just by using value functions, since the model is lacking. Therefore we have a very similar picture as in DP but this time 

$$\color{orange}{
\pi_0 \overset{\mathbb{E}}{\longrightarrow} q_{\pi_0} \overset{\mathbb{I}}{\longrightarrow} \pi_1 \overset{\mathbb{E}}{\longrightarrow} \cdots \overset{\mathbb{I}}{\longrightarrow} \pi_{\*} \overset{\mathbb{E}}{\longrightarrow} q_{\pi_{\*}}
}
$$

so we are selecting the policy $\color{orange}\pi_{k+1}$ greedy with respect to $\color{orange}q_{\pi_{k}}$.


We can replicate value iteration and policy iteration in some sense. That is we can try to estimate hardly the `q` function or we could also try to improve on an episode on episode basis. 


### **Monte Carlo with Exploring Starts**

Monte Carlo with **exploring starts** is the natural way to implement this idea

**ES**

- Input: $\color{orange}\pi$
- $\color{orange} G\leftarrow 0$
- $\color{orange} Q(s,a) \leftarrow q_0(s,a) \in \mathbb(R)$
- $\color{orange} R(s,a) \leftarrow \emptyset$ 
- Loop forever 
  - generate episode $\color{orange}S_{0}, A_{0}, R_{1}, \cdots , S_{T-1}, A_{T-1}, R_{T}$ making sure that the first state action pair is selected randomly
  - loop `episode[::-1]` : `index:T--`
    - $\color{orange}G\leftarrow \gamma G + R_{t+1}$
    - if `(s,a)` present not in episode `episode[:T]:
      - $\color{orange}R(s,a) \rightarrow  \text{append} (G)$ 
      - $\color{orange}Q(s,a) \rightarrow avg(R(s,a))$
      - $\color{orange}\pi(s) \rightarrow \text{greedy} (Q(s,a))$


### **Monte Carlo without Exploring Starts**

There is a basic separation in policy improvement algorithms, and MC w/o Exploring Starts seems a nice way to introduce as Sutton does. On/Off policy.

- On Policy uses the same policy as  the one that is optimizing
- Off Policy uses one policy to optimize and another one to _search_ or **generate the data**. 

This is like learning from someone else's experience vs our own. On the latter we must concentrate on knowing how good we are at the task, and try to navigate the behavior space in a way such that we maximize a goal, or minimize a cost. The experience to which we are going to be exposed while investigating it is going to be somewhat biased by our current one, so we are somewhat sensitive to local minima you would say. When learning from someone else (or elses) in principle we are not prisoners of our biased trajectories. But nevertheless we need to be able to say to a degree that the current actions are somewhat compatible with our history. We can basically explore but weighting the exploration on how useful it is to us.

ES is somewhat not realizable since it may be the case that some state action pairs are never visited. 

- Input: $\color{orange}\pi$
- $\color{orange} G\leftarrow 0$
- $\color{orange} Q(s,a) \leftarrow q_0(s,a) \in \mathbb(R)$
- $\color{orange} R(s,a) \leftarrow \emptyset$ 
- Loop forever 
  - generate episode $\color{orange}S_{0}, A_{0}, R_{1}, \cdots , S_{T-1}, A_{T-1}, R_{T}$ making sure that the first state action pair is selected randomly
  - loop `episode[::-1]` : `index:T--`
    - $\color{orange}G\leftarrow \gamma G + R_{t+1}$
    - if `(s,a)` present not in episode `episode[:T]:
      - $\color{orange}R(s,a) \rightarrow  \text{append} (G)$ 
      - $\color{orange}Q(s,a) \rightarrow avg(R(s,a))$
      - $\color{orange}\pi(s) \rightarrow \text{greedy} (Q(s,a))$
      - $\color{orange}\forall a \in \mathbin{A}(S_t)$
        - if $\color{orange}a=\argmax_a Q(s,a) \rightarrow \pi(a|s) = 1-\varepsilon + \frac{\varepsilon}{|\mathbin{A}(S_t)}$ else $\color{orange}\pi(a|s) = \frac{\varepsilon}{|\mathbin{A}(S_t)}$


The above algorithm optimizes over the $\color{orange}\varepsilon-soft$ policies, described as to be policies that have non-zero probability of selecting any action under all possible states. That is we are optimizing over a modified transitional operator. This is a word in which sometimes noise kicks you out of what seems to be the best policy, so in turn you learn to optimize assuming that sometimes you may be kicked out from a local optimum. 

### **Off-Policy Importance Sampling**

How to mantain exploration and at the same time explore all possible actions, to find the potential better ones. The above algorithm implies a compromise, since we are optimizing over a near-optimal policy that still explores. So an alternative can  be to have two policies
- one that explores: `behavior policy`
- one that gets optimized: `target policy`

Off policy vs On policy tradeoffs
- harder - simpler
- more data - less data
- more variance - less variance
- more time for convergence - less time for convergence
- general framework, superset that includes `on policy` - special case of `behavior = target`
- learn from whatever you choose - learn from what you do 

**Importance sampling** comes into place here. This is just a technique for estimating expected values under one distribution, given samples from another one. The following example is quite intuitive

$$
\color{orange}{
\mathbb{E}_{\sim p}(f) = \int f(x) p(x) dx = \mathbb{E}_{\sim q}(f \cdot p/q) = \int \frac{f(x)p(x)}{q(x)}q(x) dx
}
$$

So in essence what `IS` is doing doing is: weighting each point in probability space by a factor that is proportionate to how likely is to sample from $\color{orange}p$ instead of $\color{orange}q$.

In the case f episodes or state-action trajectories, we get that the probability of obtaining a trajectory $\color{orange}A_t, S_{t+1}, \cdots, S_T$ under policy $\color{orange}\pi$ is 

$$
\color{orange}{
   Pr\left\{ A_t, S_{t+1}, \cdots, S_T | S_t, A_{t:T-1} \sim \pi \right\} = \prod_{k=t}^{T-1}\pi(A_k|S_k) \cdot p(S_{k+1}|S_k, A_k)
} 
$$

The same applies to any policy. If the behavior policy is $\color{orange}b$, the probability of obtaining a particuar trajectory is

$$\color{orange}{
\prod_{k=t}^{T-1}b(A_k|S_k) \cdot p(S_{k+1}|S_k, A_k)
}
$$

even if the way the world transitions is hidden, i.e. how the world makes an update to the global state, we can 

$$\color{orange}{
   \frac{\prod_{k=t}^{T-1}\pi(A_k|S_k) \cdot p(S_{k+1}|S_k, A_k)}{\prod_{k=t}^{T-1}b(A_k|S_k) \cdot p(S_{k+1}|S_k, A_k)} = \frac{\prod_{k=t}^{T-1}\pi(A_k|S_k)}{\prod_{k=t}^{T-1}b(A_k|S_k)} = \rho_{t:T-1}
}
$$

and is $\color{orange}\rho_{t:T-1}$ the weighting factor for importance sampling. After this all that is left to underttand is that 
- the probability space from which the samples are withdrawn corresponsd to the state-action trajectories 
- the expectation operator has to be applied to the returns $\color{orange}$ 
- returns are a mapping from trajectories to real numbers
- expectation is going to be the mean of the returns 

Extra notation:
- $\color{orange}\tau(s)$: set of time steps in which state $\color{orange}s$ was visited. For every visit this is. For first time, it would be the set of all time steps that were first visits to s within their respective episodes. 
- $\color{orange} T(t)$: index of the last time step in the episode belonging to the range $\color{orange}[t, T-1]$
- $\color{orange}G(t)$: return after t up through $\color{orange}T(t)$. 

we define then `ordinary importance sampling` as

$$\color{orange}{
   V(s) = \frac{\sum_{t\in \tau(s)} \rho_{t:T(t)-1}G_t}{|\tau(s)|}
}
$$

and `weighted importance sampling` as

$$\color{orange}{
   V(s) = \frac{\sum_{t\in \tau(s)} \rho_{t:T(t)-1}G_t}{\sum_{t\in \tau(s)} \rho_{t:T(t)-1}}
}
$$

basic differences between these two
- first visit
  - `ordinary` is unbiased for first visit but can be extremely variant since the ratios are not bounded. 
  - `weighted` is biased (although it converges to zero) and its variance is bounded by the maximum return. 
- every visit
  - `ordinary` biased but converges to zero
  - `weighted` biased but converges to zero

Down below a nice example displays the convergence problems for ordinary importance samples.

![](/assets/images/infinite_variance.png)

<br/>

**First-Visit off policy evaluation naive implementation**

- Input: $\color{orange}\pi$
- $\color{orange} Q(s,a) \leftarrow q_0(s,a) \in \mathbb(R)$
- $\color{orange} R(s,a) \leftarrow \emptyset$ 
- $\color{orange} \tau(s,a) \leftarrow \emptyset$
- $\color{orange} \rho(s,a) \leftarrow \emptyset$
- Loop some number of episodes 
  - $\color{orange} b\leftarrow$ any policy with coverage of $\color{orange}\pi$
  - $\color{orange} G\leftarrow 0$
  - $\color{orange} W \leftarrow 1$ 
  - generate episode with $\color{orange}b \rightarrow S_{0}, A_{0}, R_{1}, \cdots , S_{T-1}, A_{T-1}, R_{T}$
  - loop `episode[::-1]` - `index:t => (t=T; t>0; t--)`
    - $\color{orange}G\leftarrow \gamma G + R_{t+1}$
    - $\color{orange}W\leftarrow W \cdot \frac{\pi(a_t|s_t)}{b(a_t|s_t)}$
    - if $\color{orange}W=0$ then break
    - if `(s,a)` not in episode `episode[:T]`:
      - $\color{orange}\tau(s,a) \leftarrow append(t)$
      - $\color{orange}R(s,a) \leftarrow  \text{append} (G)$ 
      - $\color{orange}\rho(s,a) \leftarrow  \text{append} (W)$
  - $\color{orange} Q(s,a) \leftarrow \frac{\sum \rho R(s,a)}{\sum \rho}$


<br/>

**Incremental implementation**

The methods described in the multi armed bandits sections can easily be applied when implementing incremental versions of the montecarlo algorithms displayed in this notes. The one difference on the update rule corresponds to `weighted` averages since this is just not only dependent on count. 

Given a set of returns $\color{orange}{G}_k$ and a respective set of weights $\color{orange}\rho_k$ the weighted average is 

$$\color{orange}{
   V_{K+1}(s) = \frac{\rho_k G_k}{\rho_{kk}} = \frac{\sum_{k=1}^K \rho_k G_k}{\sum_{k=1}^K \rho_k}
}
$$ 

therefore 

$$\color{orange}{
   V_{K+1}(s) = \frac{\sum_{k}^{K-1}{\rho_k G_k} + \rho_K G_K}{\sum_{k}^{K} \rho_k}  = V_K + \frac{\rho_K}{\sum_k^K \rho_k} (G_K - V_K )
}
$$

and this can get implemented without the need to save list of returns and weights, just the last one. 

<br/>

### **Off Policy Control**

This is the fun part. Using off-policy methods to do policy improvement. So lets enumerate the concepts and musts
- `behavior` policy $\color{orange}b$ is going to generate the episodes.
  - `coverage` must be guaranteed
  - must be soft, i.e. $\color{orange}b(a|s)\geq 0 \ \forall s \in S, \ a \in A$
- `target` policy is the one that is going to be greedy with respect to the q function. 

**Off-policy MC control incremental implementation for finding $\color{orange}\pi \approx \pi^{*}$**

- Intialize:
  - $\color{orange} Q(s,a) \leftarrow q_0(s,a) \in \mathbb(R)$
  - $\color{orange} C(s,a) \leftarrow 0$ 
  - $\color{orange} \pi(s) = \argmax_{a} Q(s,a)$
- Loop some number of episodes 
  - $\color{orange} b\leftarrow$ any soft policy with coverage of $\color{orange}\pi$
  - $\color{orange} G\leftarrow 0$
  - $\color{orange} W \leftarrow 1$ 
  - generate episode with $\color{orange}b \rightarrow S_{0}, A_{0}, R_{1}, \cdots , S_{T-1}, A_{T-1}, R_{T}$
  - loop `episode[::-1]` - `index:t => (t=T-1; t>0; t--)`
    - $\color{orange}G\leftarrow \gamma G + R_{t+1}$
    - $\color{orange}C(s,a) \leftarrow C(s,a) + W$
    - $\color{orange}Q(s,a) \leftarrow Q(s,a) + \frac{W}{C(s,a)}(Q(s,a)-G)$
    - $\color{orange}\pi(s,a) \leftarrow \argmax_{a}Q(s,a)$
    - if $\color{orange}\pi(s_t)\neq a_t \rightarrow $ continue
    - $\color{orange}W \leftarrow W\cdot \frac{1}{b(a|s)}$
    

<br/>

As a final remark is worth noting that there exists two other algorithms in the book, more specialized to reduce variance, and faster convergence. They do not provide a core conceptual understanding of the problem, and how Monte Carlo solves it. 
- `discounted aware importance sample`
- `per-decision importance sample`


<br/>

<br>
<br>

### Copyright
Copyright © 2023 Iván Belenky. The code in this repository is licensed under the MIT License.
All this notes correspond to Sutton's book, this is just a summary. 
