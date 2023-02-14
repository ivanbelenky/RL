import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

from rl import dynaq, ModelFree

GRID_HEIGHT, GRID_WIDTH = 6, 9
START_XY, GOAL_XY = (0,3), (8,5)
OBSTACLES = [(2,2), (2,3), (2,4), (5,1), (7,3), (7,4), (7,5)]

states = [(x,y) for x in range(GRID_WIDTH) for y in range(GRID_HEIGHT) 
    if (x,y) not in OBSTACLES]
actions = ['left', 'right', 'up', 'down'] 


def obstacle_maze(state, action):
    x,y = state
    x_n, y_n = x, y

    if (x,y) == GOAL_XY:
        return (state, 1), True

    if action == 'left':
        x_n -= 1
    if action == 'right':
        x_n += 1
    if action == 'up':
        y_n += 1
    if action == 'down':
        y_n -= 1
    
    if x_n < 0 or x_n >= GRID_WIDTH:
        x_n = x
    if y_n < 0 or y_n >= GRID_HEIGHT:
        y_n = y
    if (x_n, y_n) in OBSTACLES:
        x_n, y_n = x, y

    state_n = (x_n, y_n)
    return (state_n, 0), False


vqpi_0, samples_0 = dynaq(states, actions, obstacle_maze, START_XY,
    n_episodes=50, gamma=0.95, alpha=0.5, eps=0.1, n=0, max_steps=2E3)

# plot found policy
final_policy = samples_0[-1][-1]
mf = ModelFree(states, actions, obstacle_maze, gamma=0.95, policy=final_policy)

lrud = ['<', '>', '^', 'v']
pi = vqpi_0[2].pi

plt.figure(figsize=(6,6))
for s, p in zip(states, pi):
    marker = lrud[np.argmax(p)]
    plt.scatter(s[0], s[1], c='red', marker=marker)

for x,y in OBSTACLES:
    plt.scatter(x, y, c='white', marker='s')

plt.xticks([])
plt.yticks([])
plt.show()


# steps per episode function of N planning steps

NS = [0, 5, 50]
SMOOTH = 30

model = ModelFree(states, actions, obstacle_maze,gamma=0.95)
init_state = model.states.get_index(START_XY)

all_steps_per_episode = []
for i in range(SMOOTH):
    vqpi_0, samples_0 = dynaq(states, actions, obstacle_maze, START_XY, 
        n_episodes=50, gamma=0.95, alpha=0.1, eps=0.1, n=0, max_steps=1E4)
    vqpi_5, samples_5 = dynaq(states, actions, obstacle_maze, START_XY, 
        n_episodes=50, gamma=0.95, alpha=0.1, eps=0.1, n=5, max_steps=1E4)
    vqpi_50, samples_50 = dynaq(states, actions, obstacle_maze, START_XY,
        n_episodes=50, gamma=0.95, alpha=0.1, eps=0.1, n=50, max_steps=1E4)
    
    steps_per_episode = []
    for s0, s5, s50 in zip(samples_0, samples_5, samples_50):
        pi0, pi5, pi50 = s0[3], s5[3], s50[3]
        a0, a5, a50 = pi0(init_state), pi5(init_state), pi50(init_state)

        ep0 = model.generate_episode(START_XY, actions[a0] ,policy=pi0)
        ep5 = model.generate_episode(START_XY, actions[a5] ,policy=pi5)
        ep50 = model.generate_episode(START_XY, actions[a50] ,policy=pi50)

        steps_per_episode.append([len(ep0), len(ep5), len(ep50)])
    all_steps_per_episode.append(steps_per_episode)


steps_per_episode = np.mean(all_steps_per_episode, axis=0)

mean_ep_steps = np.mean(all_steps_per_episode, axis=0)

plt.figure(figsize=(6,6))
for i, n in enumerate(NS):
    plt.plot(mean_ep_steps[1:,i], linewidth=2, label='n={}'.format(n))
plt.legend(loc=1)
plt.xlabel('Episode')
plt.ylabel('Steps per episode')

plt.show()