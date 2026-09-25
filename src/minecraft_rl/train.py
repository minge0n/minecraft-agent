import argparse
import json
import random
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from minecraft_rl.gridworld import Action, GridWorld
from minecraft_rl.q_learning import QTable


def run(episodes: int, seed: int, output: Path) -> dict[str, object]:
    rng = random.Random(seed)
    env = GridWorld()
    agent = QTable(env.width * env.height, learning_rate=0.5, discount=0.95)
    started = time.monotonic()
    successes = 0
    recent_successes = 0
    steps = 0

    for episode in range(episodes):
        state = env.reset()
        while True:
            action = agent.choose(state, epsilon=0.2, rng=rng)
            next_state, reward, terminated, truncated = env.step(action)
            agent.update(state, action, reward, next_state, terminated, truncated)
            steps += 1
            successes += int(terminated)
            if episode >= episodes - min(100, episodes):
                recent_successes += int(terminated)
            state = next_state
            if terminated or truncated:
                break

    evaluation_successes = 0
    evaluation_steps = 0
    evaluation_episodes = 100
    for _ in range(evaluation_episodes):
        state = env.reset()
        while True:
            action = agent.greedy(state)
            state, _, terminated, truncated = env.step(action)
            evaluation_steps += 1
            if terminated or truncated:
                evaluation_successes += int(terminated)
                break

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    result: dict[str, object] = {
        "stage": "tabular-gridworld",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit.stdout.strip() if git_commit.returncode == 0 else None,
        "seed_python": seed,
        "config": {"episodes": episodes, "epsilon": 0.2, "evaluation_episodes": 100},
        "environment": {
            "name": "GridWorld",
            "size": [env.width, env.height],
            "start": list(env.start),
            "goal": list(env.goal),
            "obstacles": [list(cell) for cell in sorted(env.obstacles)],
            "max_steps": env.max_steps,
        },
        "agent": {
            "name": "tabular Q-learning",
            "parameters": len(agent.values) * len(Action),
            "action_order": [action.name for action in Action],
        },
        "hyperparameters": {"learning_rate": 0.5, "discount": 0.95},
        "training_steps": steps,
        "training_successes": successes,
        "training_reward": successes,
        "last_100_training_successes": recent_successes,
        "evaluation_steps": evaluation_steps,
        "evaluation_successes": evaluation_successes,
        "evaluation_success_rate": evaluation_successes / evaluation_episodes,
        "q_table": agent.values,
        "greedy_policy": [
            agent.greedy(state).name for state in range(len(agent.values))
        ],
        "duration_seconds": time.monotonic() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train tabular Q-learning on GridWorld"
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    output = (
        args.output
        or Path("runs")
        / datetime.now(UTC).strftime("gridworld-%Y%m%dT%H%M%S%fZ")
        / "metrics.json"
    )
    result = run(args.episodes, args.seed, output)
    print(
        f"Training: {result['training_successes']}/{args.episodes} successes, "
        f"{result['training_steps']} steps; "
        f"last {min(args.episodes, 100)}: "
        f"{result['last_100_training_successes']} successes"
    )
    print(
        f"Evaluation: {result['evaluation_successes']}/100 successes "
        f"({result['evaluation_success_rate']:.0%}), "
        f"{result['evaluation_steps']} steps"
    )
    print("Q-table (state (x,y): UP RIGHT DOWN LEFT):")
    for state, values in enumerate(result["q_table"]):
        x, y = state % GridWorld.width, state // GridWorld.width
        print(f"  {state:2} ({x},{y}): " + " ".join(f"{value:.3f}" for value in values))
    print("Greedy policy (S=start, G=goal, #=obstacle; arrows=actions):")
    arrows = {"UP": "^", "RIGHT": ">", "DOWN": "v", "LEFT": "<"}
    for y in range(GridWorld.height):
        cells = []
        for x in range(GridWorld.width):
            position = (x, y)
            if position == GridWorld.start:
                cells.append("S")
            elif position == GridWorld.goal:
                cells.append("G")
            elif position in GridWorld.obstacles:
                cells.append("#")
            else:
                cells.append(arrows[result["greedy_policy"][y * GridWorld.width + x]])
        print(" ".join(cells))
    print(f"Saved metrics and full-precision Q-table to {output}")


if __name__ == "__main__":
    main()
