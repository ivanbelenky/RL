from rl import tdn

GRID_HEIGHT = 7
GRID_WIDTH = 10
WIND_WEIGHT_X = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
GOAL_XY = (7, 3)

states = [(j, i) for i in range(GRID_HEIGHT) for j in range(GRID_WIDTH)]
actions = ["left", "right", "up", "down"]


def windy_grid_world(state, action):
    x, y = state
    if state == GOAL_XY:
        return (state, 1), True

    reward = 0
    if action == "left":
        x = x - 1
        y = y + WIND_WEIGHT_X[max(x, 0)]
    if action == "right":
        x = x + 1
        y = y + WIND_WEIGHT_X[min(x, GRID_WIDTH - 1)]
    if action == "up":
        y = y + 1 + WIND_WEIGHT_X[x]
    if action == "down":
        y = y - 1 + WIND_WEIGHT_X[x]

    if x < 0:
        x = 0
        reward -= 1
    if x >= GRID_WIDTH:
        x = GRID_WIDTH - 1
        reward -= 1
    if y < 0:
        y = 0
        reward -= 1
    if y >= GRID_HEIGHT:
        y = GRID_HEIGHT - 1
        reward -= 1

    return ((x, y), reward), False


vqpi, samples = tdn(
    states,
    actions,
    windy_grid_world,
    (0, 3),
    "right",
    gamma=1,
    n=1,
    alpha=0.5,
    eps=0.1,
    n_episodes=175,
    max_steps=3000,
    optimize=True,
)
