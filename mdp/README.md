# Markov Decision Process (MDP) Framework

This code provides a framework for defining and solving Markov Decision Processes (MDPs) in Python. It includes classes for defining MDPs, policies, and rewards, as well as functions for solving MDPs using various algorithms.

## Classes 
### `MarkovReward`

The `MarkovReward` class is an abstract base class for generating rewards in an MDP. It defines the `generate()` and `r_sas()` methods, which must be implemented by subclasses.
TabularReward

The `TabularReward` class is a concrete implementation of MarkovReward that uses a reward table to generate rewards. It has a constructor that takes a reward table `r_sa` as an input and stores it internally. The `generate()` method returns the reward for a given state and action, and the `r_sas()` method returns the mean reward for the next state. It could be considered as nonsense to create a class just to hold this up, but the idea is to be able to define arbitrarily reward generator functions for each state action pair. This suggests that we are just going to be dealing with independent $p(r,s'|s,a)$. Even continuous ones that 

### `MarkovPolicy`

The `MarkovPolicy` class extends the `Policy` abstract base class and defines a policy for an `MDP`. It has a constructor that takes a policy table `pi_sa` as an input and stores it internally. The `update_policy()` method updates the policy using the given value function `q_pi`, and the `π()` method returns the policy for a given state.
MDP.

### `MDP`

The `MDP` class represents an MDP and provides methods for solving it. It has a constructor that takes a state transition matrix `p_s`, a list of states, a list of actions, a discount factor `gamma`, and optional `policy` and `reward_gen` objects. The `value_function()` and `optimal_policy()` methods can be used to compute the value function and optimal policy for the MDP using various solvers.

### `Solvers`

The code includes a number of solver functions for computing the value function and optimize policies for an MDP, including `vq_π_iter_naive`, `policy_iteration`, and `value_iteration`. These solvers can be used with the `MDP` class's `value_function()` and `optimal_policy()` methods to solve an MDP.

<br>

# **Dynamic Programming (DP) the cool kid in town**

DP is the cool kid in town, since everybody is trying to copy him in some or other way. This does not mean that he is the coolest.

DP is a collection of algorithms that can be used to compute optimal policies
given a perfect model of the environment as MDP. Extremely limited, given 
the great computational expense. All methods can be thought as trying to 
achieve same effect. 

MDP is finite. Even given continuous examples the approach taken is always
to quantize it. 

Optimality equations are operators. The system is guaranteed to have a 
solution and converge if gamma < 1 or the runs are episodic. If the
tasks are completely known then this is just a system of linear equations.

The bellman equation itself, is an operator/mapping whose fixed point is the value function. It is usually called the **Bellman Expectation Operator** or the **Bellman Policy Operator**. It can be proven to converge under reasonable assumptions, like $\gamma < 1$. This method is noted as **iterative policy evaluation**. 

<br>

## **Iterative Evaluation**

The iterative solution to the expected policy value function, would be written as 

$$ \color{orange}
v_{k+1}(s) = { \sum_{a \in A} \pi(a|s) \sum_{s', r} p(s,r|s',a)[ r + \gamma v_k(s')] = \operatorname{B_{\pi}}[v_{k}(s)]  } 
$$

where $\operatorname{B_{\pi}}$ is the **Bellman Expectation Operator**, naturally, to considering the bellman equality equation as an operator that acts on the value function. It is easy to see that the actual value function is a fixed point of this operator.

$$\color{orange}
\operatorname{B_{\pi}}[v_{\pi}] = v_{\pi}
$$


it is easy to show that $\operatorname{B_{\pi}}$ is a contraction mapping under the $L_\infty$ norm

$$\color{orange}
\begin{aligned}
\left|\left|\operatorname{B_{\pi}}[v] - \operatorname{B_{\pi}}[u]\right|\right|_\infty  &= \\ \\
&= \gamma \left|\left| \sum_{a \in A} \pi(a|s) \sum_{s', r} p(s,r|s',a)[v(s') - u(s')]\right|\right|_\infty \\ \\ 
&\leq \gamma ||v - u||_\infty
\end{aligned}
$$

Given that there is exactly one value function we can show that 

$$\color{orange}
\lim_{k\rightarrow \infty} \operatorname{B_{\pi}}[v_0] = v_{\pi}
$$

given the fact that 

$$\color{orange}
\left|\left|v_{k} - v_{\pi} \right|\right| = \left|\left| \operatorname{B_{\pi}}[v_{k-1}] - \operatorname{B_{\pi}}v_{\pi} \right|\right| \leq  \gamma \left|\left| v_{k-1} - v_{\pi} \right|\right| \leq \cdots \leq \gamma^k \left|\left| v_{0} - v_{\pi} \right|\right| 
$$

<br>

## **Policy Improvement** 

As the title suggest, dynamic programming also encompasses methods to solve the optimal problem, that is the best policy there is given an MDP. In the same fashion we can define Optimality Equations for the value function. It can be easily proven by the absurd that the following is true for 

$$\color{orange}
v_{\*} = \max_{a \in A} q_{\pi_{\*}}(s,a) = \max_{a \in A} \sum_{s', r} p(s,r|s',a)[ r + \gamma v_{\*}(s')]
$$


Mouthful absurd. If the above is not true it is possible to:
- define $\pi'(s)$ that modifies the policy for all states with the above rule 
- new policy chooses the action that maximizes the state value function for every $s$. 
- calculate the value function with this new policy  
- when we encounter each state again the new policy is going to kick in. Since it always gives more reward, and since the value function is composed by discounted rewards, we now have a higher value function policy. Hence an absurd, since $v_{\*}$ was already the optimal. 
  
If this is still not clear was a mouthful see that 

$$\color{orange}
\begin{aligned}
v_{\pi}(s) &\leq q_{\pi}(s, \pi'(s)) \\
&= \mathbb{E_{\pi'}}[r+\gamma v_{\pi}(s')]\\
&\leq \mathbb{E_{\pi'}}[r+\gamma q_{\pi}(s', \pi'(s'))]\\
&= \mathbb{E_{\pi'}}[r+\gamma r + \gamma^2 v_{\pi}(s', \pi'(s'))]\\
& \ \ \vdots \\ 
&\leq \mathbb{E_{\pi'}}\left[\sum_k r_k \gamma^k \right]\\
&= v_{\pi'(s)}(s)
\end{aligned}
$$

### **Policy Iteration**

It is precisely the policy improvement theorem the one that guarantees  that the policy iteration techniques will converge to the optimal policy after going under a couple of iterations.

As Sutton, illustrates the Policy Iteration algorithm consists of the following 

$$\color{orange}
\pi_0 \overset{\mathbb{E}}{\longrightarrow} v_{\pi_0} \overset{\mathbb{I}}{\longrightarrow} \pi_1 \overset{\mathbb{E}}{\longrightarrow} \cdots \overset{\mathbb{I}}{\longrightarrow} \pi_{\*} \overset{\mathbb{E}}{\longrightarrow} v_{\pi_{\*}}
$$

So this particular solution is quite costly since we have to perform an evaluation step every single time the policy changes, and this is costly, mostly in iterative settings. But there are good news, that is Value Iteration.

### **Value Iteration**

$$\color{orange}
v_{\pi}(s)=\sum_{a \in A} \pi(a|s)q_{\pi}(s,a)
$$

 We define then the **Bellman Optimality Operator** as 

$$\color{orange}
\operatorname{B_{\*}}[v(s)] := \max_a[r(s,a) + \gamma v(s)]
$$

and we can show that it is a contraction mapping under the $L_\infty$ norm once again, with the help of the following property

$$\color{orange}
|\max_a f(a) - \max_a g(a) | \leq \max_a |f(a) - g(a)|
$$

then

$$\color{orange}
\begin{aligned}
\left|\left|\operatorname{B_{\*}}[v] - \operatorname{B_{\*}}[u]\right|\right|_\infty  &= \\ \\
&= \gamma \left|\left| \max_a \sum_{s', r} p(s,r|s',a)[v(s') - u(s')]\right|\right|_\infty \\ \\
&\leq \gamma ||v - u||_\infty
\end{aligned}
$$

once again the optimal value function is a fixed point of the Bellman Optimality Operator, i.e.

$$\color{orange}
v_{\*} = \operatorname{B_{\*}}[v_{\*}]
$$

implying that an iterative approach can be built such that

$$\color{orange}
v_{k+1} = \operatorname{B_{\*}}[v_{k}]
$$

Since we are guaranteed convergence, we can basically apply policy iteration but going for iterative policy evaluation just with one step.  

<br>
<br>

### Copyright
Copyright © 2023 Iván Belenky. This code is licensed under the MIT License. 
