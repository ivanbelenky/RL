import matplotlib.pyplot as plt

plt.style.use("dark_background")

from rl import mcts

GRID_HEIGHT, GRID_WIDTH = 6, 9
START_XY, GOAL_XY = (0, 3), (8, 5)
OBSTACLES = [(2, 2), (2, 3), (2, 4), (5, 1), (7, 3), (7, 4), (7, 5)]

states = [
    (x, y)
    for x in range(GRID_WIDTH)
    for y in range(GRID_HEIGHT)
    if (x, y) not in OBSTACLES
]
actions = ["left", "right", "up", "down"]


def obstacle_maze(state, action):
    x, y = state
    x_n, y_n = x, y

    reward = -0.05
    if action == "left":
        x_n -= 1
    if action == "right":
        x_n += 1
    if action == "up":
        y_n += 1
    if action == "down":
        y_n -= 1

    if x_n < 0 or x_n >= GRID_WIDTH:
        x_n = x
    if y_n < 0 or y_n >= GRID_HEIGHT:
        y_n = y
    if (x_n, y_n) in OBSTACLES:
        x_n, y_n = x, y

    state_n = (x_n, y_n)
    if state_n == GOAL_XY:
        return (state_n, 1), True
    return (state_n, reward), False


def action_map(state):
    possible_actions = []
    for a in actions:
        (s, _), _ = obstacle_maze(state, a)
        if s != state:
            possible_actions.append(a)
    return possible_actions


if __name__ == "__main__":
    s = START_XY
    end = False
    tree = None
    while not end:
        action, _ = mcts(s, 0.0, 500, obstacle_maze, action_map, 25, eps=1)
        print(s, action)
        (s, _), end = obstacle_maze(s, action)

    tree.plot()
