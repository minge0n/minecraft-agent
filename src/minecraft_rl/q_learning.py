import random

from minecraft_rl.gridworld import Action


class QTable:
    def __init__(self, state_count: int, learning_rate: float, discount: float) -> None:
        self.values = [[0.0 for _ in Action] for _ in range(state_count)]
        self.learning_rate = learning_rate
        self.discount = discount

    def greedy(self, state: int) -> Action:
        values = self.values[state]
        return max(Action, key=lambda action: values[action])

    def choose(self, state: int, epsilon: float, rng: random.Random) -> Action:
        if rng.random() < epsilon:
            return rng.choice(list(Action))
        values = self.values[state]
        best = max(values)
        return rng.choice([action for action in Action if values[action] == best])

    def update(
        self,
        state: int,
        action: Action,
        reward: float,
        next_state: int,
        terminated: bool,
        truncated: bool,
    ) -> None:
        future = 0.0 if terminated or truncated else max(self.values[next_state])
        old = self.values[state][action]
        target = reward + self.discount * future
        self.values[state][action] = old + self.learning_rate * (target - old)
