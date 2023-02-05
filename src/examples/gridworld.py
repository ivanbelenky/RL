"""
RL - Copyright © 2023 Iván Belenky @Leculette
"""
import sys

import numpy as np

from src.mdp import MDP, TabularReward

GRID_SIZE = 5 # 5x5 gridworld

def main():
    optimization_method = sys.argv[1] if len(sys.argv) > 1 else None
    if optimization_method is None:
        print("No optimization method specified")
        return
        
    actions = np.arange(4) # up, right, down, left
    states = np.arange(GRID_SIZE**2)
    p_s = np.zeros((GRID_SIZE**2, 4, GRID_SIZE**2))

    #initialized the transition matrix
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            state_idx = i*GRID_SIZE+j
            p_s[state_idx][0][max(i-1,0)*GRID_SIZE+j] = 1
            p_s[state_idx][1][i*GRID_SIZE+min(j+1,GRID_SIZE-1)] = 1
            p_s[state_idx][2][min(i+1,GRID_SIZE-1)*GRID_SIZE+j] = 1
            p_s[state_idx][3][i*GRID_SIZE+max(j-1,0)] = 1

    #rewrite probs for potential positions of teleport         
    p_s[0][1] = np.zeros(GRID_SIZE**2)
    p_s[0][1][21] = 1

    p_s[2][3] = np.zeros(GRID_SIZE**2) 
    p_s[2][3][21] = 1

    p_s[6][0] = np.zeros(GRID_SIZE**2)
    p_s[6][0][21] = 1 

    p_s[2][1] = np.zeros(GRID_SIZE**2)
    p_s[2][1][13] = 1

    p_s[4][3] = np.zeros(GRID_SIZE**2)
    p_s[4][3][13] = 1

    p_s[8][0] = np.zeros(GRID_SIZE**2)
    p_s[8][0][13] = 1


    #by not specifying the policy we get a equal prob one
    #it is fair to notice that this init process is tedious for tabular MDPs
    
    #initializing reward, if we land on target 
    r_sa = np.zeros((GRID_SIZE**2, 4))
    
    #Border.
    #If it try to go out of the grid, it gets -1 reward
    for i in range(GRID_SIZE):
        r_sa[i][0] = -1
        r_sa[i*GRID_SIZE+GRID_SIZE-1][1] = -1
        r_sa[i*GRID_SIZE][3] = -1
        r_sa[GRID_SIZE*(GRID_SIZE-1)+i][2] = -1

    #A
    #If lands on (0,1) position it gets a reward of +10
    r_sa[0][1] += 10
    r_sa[2][3] += 10
    r_sa[6][0] += 10

    #B
    #If lands on (0,3) position it gets a reward of +5
    r_sa[2][1] += 5
    r_sa[4][3] += 5
    r_sa[8][0] += 5

    print("Reward matrix going up")
    print(r_sa[:,0].reshape(GRID_SIZE,GRID_SIZE))
    print("Reward matrix going right")
    print(r_sa[:,1].reshape(GRID_SIZE,GRID_SIZE))
    print("Reward matrix going down")
    print(r_sa[:,2].reshape(GRID_SIZE,GRID_SIZE))
    print("Reward matrix going left")
    print(r_sa[:,3].reshape(GRID_SIZE,GRID_SIZE))
    
    #Define the Markov Decision Process
    mdp = MDP(p_s, states, actions, gamma = 0.9, 
        reward_gen=TabularReward(r_sa))

    #calculate beforehand
    v, q = mdp.vq_pi()
    print("Value Function before optimizing")
    print(v.reshape(GRID_SIZE,GRID_SIZE))
    print('-'*50)
    print("Q Function before optimizing")
    print(q.reshape(GRID_SIZE,GRID_SIZE,4))
    print('\n')

    mdp.optimize_policy(method=optimization_method)
    v, q = mdp.vq_pi()
    print("Value Function after optimizing")
    print(v.reshape(GRID_SIZE,GRID_SIZE))
    print('-'*50)
    print("Q Function after optimizing")
    print(q.reshape(GRID_SIZE,GRID_SIZE,4))
    print('\n')
 
    print("Optimal policy up action")
    print(mdp.policy.pi_sa[:,0].reshape(GRID_SIZE, GRID_SIZE))
    print("Optimal policy right action")
    print(mdp.policy.pi_sa[:,1].reshape(GRID_SIZE, GRID_SIZE))
    print("Optimal policy down action")
    print(mdp.policy.pi_sa[:,2].reshape(GRID_SIZE, GRID_SIZE))
    print("Optimal policy left action")
    print(mdp.policy.pi_sa[:,3].reshape(GRID_SIZE, GRID_SIZE))
    

if __name__ == '__main__':
    main()