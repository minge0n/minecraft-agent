from enum import IntEnum


class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class GridWorld:
    width = 4
    height = 4
    start = (0, 0)
    goal = (3, 3)
    obstacles = frozenset({(1, 1), (2, 1), (1, 2)})
    max_steps = 20

    def __init__(self) -> None:
        self.position = self.start
        self.steps = 0

    @property
    def state(self) -> int:
        x, y = self.position
        return y * self.width + x

    def reset(self) -> int:
        self.position = self.start
        self.steps = 0
        return self.state

    def step(self, action: Action) -> tuple[int, float, bool, bool]:
        if self.position == self.goal or self.steps >= self.max_steps:
            raise RuntimeError("Reset before stepping after the episode ends")

        x, y = self.position
        if action == Action.UP:
            y = max(0, y - 1)
        elif action == Action.RIGHT:
            x = min(self.width - 1, x + 1)
        elif action == Action.DOWN:
            y = min(self.height - 1, y + 1)
        elif action == Action.LEFT:
            x = max(0, x - 1)
        else:
            raise ValueError(f"Unknown action: {action}")

        if (x, y) not in self.obstacles:
            self.position = (x, y)
        self.steps += 1
        terminated = self.position == self.goal
        truncated = not terminated and self.steps >= self.max_steps
        return self.state, float(terminated), terminated, truncated
