"""Train the reward head and the actor-critic of the first Minecraft actor
(docs/stage4-minecraft-actor.md).

Stages, each resumable from `--output`:

1. Reward head: fits r(s_t) on posterior RSSM states of the reward dataset.
   Held-out episodes report the error on zero and positive reward steps
   against the zero and mean baselines.
2. Actor-critic in imagination, from posterior states of real steps, saved
   after every `--save-at` update count.

The world model is the frozen canonical Stage 3 checkpoint; it never changes.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from minecraft_rl import macos_qos, runtime
from minecraft_rl.minecraft_actor import (
    ActorConfig,
    Agent,
    RewardHead,
    load_world_model,
    posterior_states,
    reward_metrics,
    train_actor,
)
from minecraft_rl.minecraft_dataset import load_episodes
from minecraft_rl.minecraft_replay import CompactVocabulary, SequenceReplay
from minecraft_rl.provenance import git_commit

LENGTH = 32


def reward_batches(replay: SequenceReplay, episodes, length: int) -> torch.Tensor:
    """Reward per transition of the replay windows, aligned with s_1..s_L."""
    rewards = [torch.as_tensor(e.arrays["reward"]).float() for e in replay.episodes]
    return rewards


def window_rewards(rewards, index, length):
    return torch.stack([rewards[e][s : s + length] for e, s in index])


def train_reward_head(model, train, test, args, out: dict[str, Any]) -> RewardHead:
    generator = torch.Generator().manual_seed(1)
    head = RewardHead(model.features)
    head.scale = args.reward_scale
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    rewards = reward_batches(train, None, LENGTH)
    history = []
    for update in range(args.reward_updates):
        batch = train.sample(32, generator)
        with torch.no_grad():
            filtered = model.filter(batch.observations, batch.actions, generator)
            s = torch.cat([filtered["h"], filtered["z"]], -1)[:, 1:]
        index = list(zip(batch.episodes.tolist(), batch.starts.tolist(), strict=True))
        target = window_rewards(rewards, index, LENGTH)
        # Rewards are rare and small: learn them in units of `reward_scale` and
        # weight the rare nonzero steps up, so the head does not settle on the mean.
        weight = 1.0 + (args.positive_weight - 1.0) * (target != 0).float()
        error = head(s) - target * args.reward_scale
        loss = (weight * error**2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if update % 100 == 0:
            history.append({"update": update, "loss": float(loss)})
    out["reward_head_history"] = history
    out["reward_head"] = {
        "train": evaluate_reward(model, head, train),
        "test": evaluate_reward(model, head, test) if test else None,
    }
    return head


@torch.no_grad()
def evaluate_reward(model, head, replay: SequenceReplay) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(2)
    rewards = reward_batches(replay, None, LENGTH)
    index = replay.evaluation_windows(LENGTH)
    predicted, target = [], []
    for start in range(0, len(index), 64):
        chunk = index[start : start + 64]
        batch = replay.batch(chunk)
        filtered = model.filter(batch.observations, batch.actions, generator)
        s = torch.cat([filtered["h"], filtered["z"]], -1)[:, 1:]
        predicted.append(head(s).flatten() / head.scale)
        target.append(window_rewards(rewards, chunk, LENGTH).flatten())
    return reward_metrics(torch.cat(predicted), torch.cat(target))


def all_start_states(model, replay: SequenceReplay):
    generator = torch.Generator().manual_seed(3)
    hs, zs = [], []
    index = replay.evaluation_windows(LENGTH)
    for start in range(0, len(index), 64):
        batch = replay.batch(index[start : start + 64])
        h, z = posterior_states(model, batch, generator)
        hs.append(h)
        zs.append(z)
    return torch.cat(hs), torch.cat(zs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--world-model",
        type=Path,
        default=Path("runs/stage3/tradeoff-P128/seed0/world_model.pt"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-episodes", type=int, default=3)
    parser.add_argument(
        "--extra",
        type=Path,
        action="append",
        default=[],
        help="more datasets for training only, such as the actor's own episodes",
    )
    parser.add_argument("--reward-updates", type=int, default=2000)
    parser.add_argument("--reward-scale", type=float, default=100.0)
    parser.add_argument("--positive-weight", type=float, default=20.0)
    parser.add_argument("--actor-updates", type=int, default=1000)
    parser.add_argument("--save-at", default="0,200,1000")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    macos_qos.run_in_background()
    runtime.configure(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    model, extra = load_world_model(args.world_model)
    vocabulary = CompactVocabulary.from_json(extra["vocabulary"])
    episodes = sorted(
        load_episodes(args.dataset / "train"), key=lambda e: e.meta.episode
    )
    if any("reward" not in e.arrays for e in episodes):
        raise SystemExit("the dataset has episodes without reward (format v1)")
    test_eps = episodes[-args.test_episodes :] if args.test_episodes else []
    train_eps = episodes[: len(episodes) - len(test_eps)]
    for extra_dataset in args.extra:
        train_eps += load_episodes(extra_dataset / "train")
    train = SequenceReplay(train_eps, vocabulary, LENGTH)
    test = SequenceReplay(test_eps, vocabulary, LENGTH) if test_eps else None
    out: dict[str, Any] = {
        "experiment": "minecraft_actor",
        "git_commit": git_commit(),
        "world_model": str(args.world_model),
        "objective": extra["objective"],
        "dataset": str(args.dataset),
        "train_episodes": len(train_eps),
        "extra_datasets": [str(p) for p in args.extra],
        "test_episodes": [e.meta.episode for e in test_eps],
        "runtime": runtime.metadata(torch.device("cpu")),
    }
    started = time.perf_counter()
    head_path = args.output / "reward_head.pt"
    if head_path.exists():
        head = RewardHead(model.features)
        saved = torch.load(head_path, weights_only=True)
        head.load_state_dict(saved["reward_head"])
        head.scale = saved["scale"]
        out |= json.loads((args.output / "reward_head.json").read_text())
    else:
        head = train_reward_head(model, train, test, args, out)
        torch.save({"reward_head": head.state_dict(), "scale": head.scale}, head_path)
        (args.output / "reward_head.json").write_text(
            json.dumps(
                {k: out[k] for k in ("reward_head", "reward_head_history")}, indent=2
            )
        )
    print(json.dumps(out["reward_head"], indent=1), flush=True)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)

    config = ActorConfig(seed=args.seed)
    torch.manual_seed(args.seed)
    agent = Agent(model.features)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate)
    starts = all_start_states(model, train)
    out["start_states"] = int(starts[0].shape[0])
    generator = torch.Generator().manual_seed(args.seed + 10)
    save_at = sorted(int(x) for x in args.save_at.split(","))
    history: list[dict[str, Any]] = []
    done = 0
    for target in save_at:
        if target > done:
            history += train_actor(
                model, head, agent, optimizer, starts, config, generator, target - done
            )
            done = target
        torch.save(
            {"agent": agent.state_dict(), "updates": done, "config": config.to_json()},
            args.output / f"agent-{done}.pt",
        )
        recent = history[-50:] if history else []
        print(
            json.dumps(
                {
                    "updates": done,
                    "imagined_return": sum(h["imagined_return"] for h in recent)
                    / max(1, len(recent)),
                    "entropy": sum(h["entropy"] for h in recent) / max(1, len(recent)),
                }
            ),
            flush=True,
        )
    out["actor_config"] = config.to_json()
    out["actor_history"] = history[::10]
    out["seconds"] = time.perf_counter() - started
    (args.output / "metrics.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
