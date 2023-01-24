# Monte Carlo Methods 

_"Monte Carlo methods utilize experience—sample sequences of states, actions, and rewards from actual or simulated interaction with an environment. Learning from actual experience is striking because it requires no prior knowledge of the environment’s dynamics, yet can still attain optimal behavior. Learning from simulated experience is also powerful. Although a model is required, the model need only generate sample transitions, not the complete probability distributions of all possible transitions that is required for dynamic programming (DP). In surprisingly many cases it is easy to generate experience sampled according to the desired probability distributions, but infeasible to obtain the distributions in explicit form."_

So basically if we have a universe in which we can sample stuff, we dont even have to bother with the model. If we want to simulate it, we must create a transition model, and therefore we leave out the complications of building the actual transition probability functions for each state action pair. It is fair to say that given this approach, it is mandatory to tae an episodic approach, otherwise there is no end to the endeavor.

## Monte Carlo Prediction

The goal here is to learn the state-value function. The basic idea behind monte carlo is to average the value function across every episode we have at hand. Lets say that we have to find $\color{orange}v_{\pi}(s)$. Each occurence of state $\color{orange}s$ in an episode is called a _visit_ to $\color{orange}s$. 

### First visit Monte Carlo

First visit just averages all episodes after the _first visit to each state_ $\color{orange}s$

Given 
- Input: $\color{orange}\pi$
- Initialize
  -  state-value: $\color{orange}v(s)$
  -  returns for each state: $\color{orange}R(s)$ 
-  While some condition is true (tolerance, amount of iterations)
   -  Generate an episode following policy $\color{orange}{\pi \rightarrow  S_{0}, A_{0}, R_{1}, \cdots , S_{T-1}, A_{T-1}, R_{T}}$
   -  $\color{orange}G \leftarrow 0$
   -  loop episode[::-1], i.e. $\color{orange}T-1, T-2, \cdots, 0$
      -  $\color{orange}G \leftarrow \gamma G + R_{t+1}$
      -  if state $\color{orange}s$ not in $\color{orange}[S_0, S_1, \cdots, S_{t-1} ]$:
         -  $\color{orange} R(s) \leftarrow \text{append} \ G$
         -  $\color{orange} v_{\pi}(s) \leftarrow avg(R(s))$  


The implementation is arbitrary. If it gets complicated with the backwards iteration, and the forward iteration, check the natural implementation following the basic principles of the algorithm, isomorphic. 



<br>
<br>

### Copyright
Copyright © 2023 Iván Belenky. This code is licensed under the MIT License. 


<!-- the interface should be somewhat  -->