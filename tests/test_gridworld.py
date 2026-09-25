import random

import pytest

from minecraft_rl.gridworld import Action, GridWorld
from minecraft_rl.q_learning import QTable
from minecraft_rl.train import run


def test_reset_and_state_encoding() -> None:
    world = GridWorld()
    assert world.reset() == 0
    assert world.step(Action.RIGHT) == (1, 0.0, False, False)
    assert world.step(Action.DOWN) == (1, 0.0, False, False)
    assert world.step(Action.RIGHT) == (2, 0.0, False, False)
    assert world.step(Action.DOWN) == (2, 0.0, False, False)
    assert world.reset() == 0
    assert world.steps == 0
    assert world.position == world.start


def test_obstacles_block_from_each_reachable_side() -> None:
    world = GridWorld()
    world.step(Action.DOWN)
    assert world.step(Action.RIGHT) == (4, 0.0, False, False)
    world.step(Action.DOWN)
    assert world.step(Action.RIGHT) == (8, 0.0, False, False)
    world.reset()
    world.step(Action.RIGHT)
    world.step(Action.RIGHT)
    assert world.step(Action.DOWN) == (2, 0.0, False, False)


def test_walls_goal_and_terminal_reset() -> None:
    world = GridWorld()
    assert world.step(Action.UP) == (0, 0.0, False, False)
    for action in (Action.RIGHT,) * 3 + (Action.DOWN,) * 3:
        result = world.step(action)
    assert result == (15, 1.0, True, False)
    with pytest.raises(RuntimeError):
        world.step(Action.LEFT)


def test_truncation_and_reset() -> None:
    world = GridWorld()
    for _ in range(world.max_steps):
        result = world.step(Action.LEFT)
    assert result == (0, 0.0, False, True)
    with pytest.raises(RuntimeError):
        world.step(Action.LEFT)
    assert world.reset() == 0


def test_q_update_bootstraps_only_when_episode_continues() -> None:
    agent = QTable(16, learning_rate=0.5, discount=0.9)
    agent.values[1][Action.UP] = 2.0
    agent.update(0, Action.RIGHT, 1.0, 1, False, False)
    assert agent.values[0][Action.RIGHT] == pytest.approx(1.4)
    agent.update(0, Action.RIGHT, 1.0, 1, True, False)
    assert agent.values[0][Action.RIGHT] == pytest.approx(1.2)
    agent.update(0, Action.RIGHT, 1.0, 1, False, True)
    assert agent.values[0][Action.RIGHT] == pytest.approx(1.1)


def test_seed_controls_exploration_and_greedy_ties() -> None:
    agent = QTable(16, learning_rate=0.5, discount=0.9)
    assert agent.greedy(0) == Action.UP
    agent.values[0][Action.RIGHT] = 1.0
    agent.values[0][Action.DOWN] = 1.0
    assert agent.greedy(0) == Action.RIGHT
    first = [agent.choose(0, 0.0, random.Random(seed)) for seed in range(10)]
    second = [agent.choose(0, 0.0, random.Random(seed)) for seed in range(10)]
    assert first == second
    assert set(first) == {Action.RIGHT, Action.DOWN}
    first = [agent.choose(0, 1.0, random.Random(seed)) for seed in range(10)]
    second = [agent.choose(0, 1.0, random.Random(seed)) for seed in range(10)]
    assert first == second


def test_seed_reproduces_q_table_and_metrics(tmp_path) -> None:
    first = run(50, 7, tmp_path / "first.json")
    second = run(50, 7, tmp_path / "second.json")
    for key in (
        "q_table",
        "greedy_policy",
        "training_steps",
        "training_successes",
        "evaluation_successes",
    ):
        assert first[key] == second[key]


def test_run_writes_metrics_and_evaluation_does_not_update(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    updates = 0
    original_update = QTable.update

    def track_update(self, *args, **kwargs):
        nonlocal updates
        updates += 1
        return original_update(self, *args, **kwargs)

    monkeypatch.setattr(QTable, "update", track_update)
    output = tmp_path / "metrics.json"
    result = run(10, 7, output)
    assert result["training_steps"] > 0
    assert updates == result["training_steps"]
    assert result["agent"] == {
        "name": "tabular Q-learning",
        "parameters": 64,
        "action_order": ["UP", "RIGHT", "DOWN", "LEFT"],
    }
    assert result["environment"]["obstacles"] == [[1, 1], [1, 2], [2, 1]]
    assert len(result["q_table"]) == 16
    assert all(len(row) == 4 for row in result["q_table"])
    assert result["training_reward"] == result["training_successes"]
    assert '"evaluation_successes"' in output.read_text(encoding="utf-8")
