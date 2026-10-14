"""The movement control test: reward head and actor on `reward-control-v1`
(docs/stage4-control-test.md).

Stages, each written to `--output` and skipped when its result exists:

1. `data`: counts of positive, negative and zero-progress steps, turning
   steps, per split.
2. `reward_head`: the existing reward head on whole-episode states of the
   frozen canonical RSSM, trained on the training episodes, evaluated on the
   held-out ones against the zero and the mean baselines, and on the old
   camera-spinning actor episodes (`--spin`) where they have positions.
3. `actor`: the existing actor-critic in imagination with this head,
   checkpoints at 0 and `--actor-updates`.

Every step of this script reads the frozen world model and never changes it.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy
import torch

from minecraft_rl import macos_qos, runtime
from minecraft_rl.minecraft_actor import (
    ActorConfig,
    Agent,
    RewardHead,
    load_world_model,
    train_actor,
)
from minecraft_rl.minecraft_dataset import load_episodes
from minecraft_rl.minecraft_replay import CompactVocabulary
from minecraft_rl.minecraft_reward_control import CONTROL_V1, POSITION_KEY, episode_task
from minecraft_rl.minecraft_reward_model import (
    RewardTraining,
    chronological_states,
    episode_data,
    train_frozen_head,
)
from minecraft_rl.provenance import git_commit

WORLD_MODEL = Path("runs/stage3/tradeoff-P128/seed0/world_model.pt")
# Steps that move less than this many blocks toward or away from the target
# count as "no progress" in the data and error tables.
STILL = 0.01


def control_data(model, episodes, compact):
    """`EpisodeData` whose rewards are the control rewards."""
    data = episode_data(model, episodes, compact)
    for d, episode in zip(data, episodes, strict=True):
        task = episode_task(episode.arrays)
        d.rewards = torch.as_tensor(task["rewards"]).float()
        yaw = numpy.abs(episode.arrays["action_camera"][:, 0])
        d.labels = {
            "toward": d.rewards > STILL,
            "away": d.rewards < -STILL,
            "still": d.rewards.abs() <= STILL,
            "turning_still": (d.rewards.abs() <= STILL) & torch.as_tensor(yaw >= 10.0),
        }
    return data


def counts(data) -> dict[str, Any]:
    out = {"episodes": len(data), "transitions": sum(d.steps for d in data)}
    for name in ("toward", "away", "still", "turning_still"):
        out[name] = int(sum(int(d.labels[name].sum()) for d in data))
    out["reward_per_step"] = float(torch.cat([d.rewards for d in data]).mean())
    return out


def report(predicted: torch.Tensor, target: torch.Tensor, labels, mean: float) -> dict:
    out: dict[str, Any] = {
        "transitions": int(target.numel()),
        "model_mae": float((predicted - target).abs().mean()),
        "zero_baseline_mae": float(target.abs().mean()),
        "mean_baseline_mae": float((target - mean).abs().mean()),
        "correlation": float(torch.corrcoef(torch.stack([predicted, target]))[0, 1]),
        "predicted_per_step": float(predicted.mean()),
        "actual_per_step": float(target.mean()),
        "sign_agreement_on_moving_steps": None,
        "by_kind": {},
    }
    moving = labels["toward"] | labels["away"]
    if moving.any():
        out["sign_agreement_on_moving_steps"] = float(
            ((predicted[moving] > 0) == (target[moving] > 0)).float().mean()
        )
    for name, mask in labels.items():
        n = int(mask.sum())
        out["by_kind"][name] = {
            "count": n,
            "predicted_mean": float(predicted[mask].mean()) if n else None,
            "actual_mean": float(target[mask].mean()) if n else None,
            "model_mae": float((predicted - target)[mask].abs().mean()) if n else None,
        }
    return out


def evaluate(model, head, data, mean: float) -> dict:
    states = chronological_states(model, data, seed=7)
    with torch.no_grad():
        predicted = torch.cat([head.reward(s[1:]) for s in states])
    target = torch.cat([d.rewards for d in data])
    labels = {k: torch.cat([d.labels[k] for d in data]) for k in data[0].labels}
    out = report(predicted, target, labels, mean)
    out["episode_returns"] = {
        "predicted": [float(head.reward(s[1:]).sum().detach()) for s in states],
        "actual": [float(d.rewards.sum()) for d in data],
    }
    return out


def ridge_r2(train_x, train_y, test_x, test_y, penalty: float = 10.0) -> list[float]:
    """Held-out R^2 of a ridge regression per target column."""
    mean, std = train_x.mean(0), train_x.std(0).clamp(min=1e-6)
    a = torch.cat([(train_x - mean) / std, torch.ones(len(train_x), 1)], 1).double()
    b = torch.cat([(test_x - mean) / std, torch.ones(len(test_x), 1)], 1).double()
    eye = torch.eye(a.shape[1], dtype=torch.float64) * penalty
    weights = torch.linalg.solve(a.T @ a + eye, a.T @ train_y.double())
    residual = ((b @ weights - test_y.double()) ** 2).sum(0)
    total = ((test_y.double() - test_y.double().mean(0)) ** 2).sum(0)
    return (1 - residual / total).tolist()


def heading_probe(model, train, test, train_eps, test_eps) -> dict:
    """Can the state tell the heading (sin, cos of yaw) and the distance to
    the target? The reward needs both. Probes on h, z and the current
    observation embedding e, fitted on training, scored on held-out."""

    def targets(episodes):
        rows = []
        for e in episodes:
            yaw = numpy.radians(e.arrays["privileged_yaw"])
            d = numpy.linalg.norm(
                e.arrays[POSITION_KEY] - episode_task(e.arrays)["target"], axis=-1
            )
            rows.append(numpy.stack([numpy.sin(yaw), numpy.cos(yaw), d], 1))
        return torch.as_tensor(numpy.concatenate(rows)).float()

    s_train = torch.cat(chronological_states(model, train, seed=7))
    s_test = torch.cat(chronological_states(model, test, seed=7))
    hidden = model.hidden
    inputs = {
        "h": (s_train[:, :hidden], s_test[:, :hidden]),
        "z": (s_train[:, hidden:], s_test[:, hidden:]),
        "e": (
            torch.cat([d.embedded for d in train]),
            torch.cat([d.embedded for d in test]),
        ),
    }
    y_train, y_test = targets(train_eps), targets(test_eps)
    names = ("sin_yaw", "cos_yaw", "distance")
    out: dict[str, Any] = {}
    for name, (x_train, x_test) in inputs.items():
        scores = ridge_r2(x_train, y_train, x_test, y_test)
        out[name] = dict(zip(names, scores, strict=True))
    out["held_out_states"] = int(len(y_test))
    return out


def write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("runs/stage4/control-random")
    )
    parser.add_argument("--spin", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-episodes", type=int, default=3)
    parser.add_argument("--reward-updates", type=int, default=3000)
    parser.add_argument("--actor-updates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    macos_qos.run_in_background()
    runtime.configure(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    model, extra = load_world_model(WORLD_MODEL)
    compact = CompactVocabulary.from_json(extra["vocabulary"])
    episodes = sorted(
        load_episodes(args.dataset / "train"), key=lambda e: e.meta.episode
    )
    test_eps, train_eps = (
        episodes[-args.test_episodes :],
        episodes[: -args.test_episodes],
    )
    train = control_data(model, train_eps, compact)
    test = control_data(model, test_eps, compact)
    spin_eps = [
        e
        for root in args.spin
        for e in load_episodes(root / "train")
        if POSITION_KEY in e.arrays
    ]
    spin = control_data(model, spin_eps, compact) if spin_eps else []
    # Movement rewards are dense and signed: no positive weighting and no
    # positive-centered chunks; scale 10 brings a sprint step (0.28) near 3.
    config = RewardTraining(scale=10.0, positive_weight=1.0, positive_chunk_share=0.0)
    meta = {
        "git_commit": git_commit(),
        "task": CONTROL_V1.to_json(),
        "world_model": str(WORLD_MODEL),
        "training": config.to_json(),
        "train": counts(train),
        "held_out": counts(test),
        "spin": counts(spin) if spin else None,
    }
    write(args.output / "data.json", meta)
    print(json.dumps({k: meta[k] for k in ("train", "held_out", "spin")}), flush=True)

    head_path = args.output / "reward_head.pt"
    if not head_path.exists():
        model.eval()
        states = chronological_states(model, train, seed=7)
        head, history = train_frozen_head(
            states, train, model.features, config, args.reward_updates, args.seed
        )
        mean = float(torch.cat([d.rewards for d in train]).mean())
        out = {
            "train": evaluate(model, head, train, mean),
            "held_out": evaluate(model, head, test, mean),
            "spin": evaluate(model, head, spin, mean) if spin else None,
            "history": history,
        }
        write(args.output / "reward_head.json", out)
        torch.save({"reward_head": head.state_dict(), "scale": head.scale}, head_path)
        print(
            json.dumps(
                {
                    k: {
                        m: out[k][m]
                        for m in ("model_mae", "zero_baseline_mae", "correlation")
                    }
                    for k in ("train", "held_out", "spin")
                    if out[k]
                }
            ),
            flush=True,
        )
    saved = torch.load(head_path, weights_only=True)
    head = RewardHead(model.features)
    head.load_state_dict(saved["reward_head"])
    head.scale = saved["scale"]
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)

    path = args.output / "learning_curve.json"
    if not path.exists():
        # Held-out error of heads trained on the first 3, 6 and 9 training
        # episodes: does more data of the same kind help?
        model.eval()
        all_states = chronological_states(model, train, seed=7)
        curve = []
        for n in (3, 6, len(train)):
            h, _ = train_frozen_head(
                all_states[:n],
                train[:n],
                model.features,
                config,
                args.reward_updates,
                args.seed,
            )
            mean = float(torch.cat([d.rewards for d in train[:n]]).mean())
            r = evaluate(model, h, test, mean)
            curve.append(
                {
                    "train_episodes": n,
                    "held_out_mae": r["model_mae"],
                    "zero_baseline_mae": r["zero_baseline_mae"],
                    "correlation": r["correlation"],
                    "toward_predicted_mean": r["by_kind"]["toward"]["predicted_mean"],
                }
            )
        write(path, {"curve": curve})
        print(json.dumps(curve), flush=True)

    path = args.output / "heading_probe.json"
    if not path.exists():
        write(path, heading_probe(model, train, test, train_eps, test_eps))
        print(path.read_text(), flush=True)

    path = args.output / "spin_without_positions.json"
    if args.spin and not path.exists():
        # The first actor episodes have no positions, so no control reward:
        # report the head's predictions on them, split by camera turning.
        old = [
            e
            for root in args.spin
            for e in load_episodes(root / "train")
            if POSITION_KEY not in e.arrays
        ]
        if old:
            data = episode_data(model, old, compact)
            states = chronological_states(model, data, seed=7)
            with torch.no_grad():
                predicted = torch.cat([head.reward(s[1:]) for s in states])
            yaw = torch.as_tensor(
                numpy.concatenate(
                    [numpy.abs(e.arrays["action_camera"][:, 0]) for e in old]
                )
            )
            write(
                path,
                {
                    "episodes": len(old),
                    "predicted_per_step": float(predicted.mean()),
                    "predicted_per_step_turning": float(predicted[yaw >= 10].mean()),
                    "predicted_per_step_not_turning": float(predicted[yaw < 10].mean()),
                    "predicted_return_per_episode": [
                        float(head.reward(s[1:]).sum()) for s in states
                    ],
                },
            )

    if (
        args.actor_updates <= 0
        or (args.output / f"agent-{args.actor_updates}.pt").exists()
    ):
        return
    actor_config = ActorConfig(seed=args.seed)
    torch.manual_seed(args.seed)
    agent = Agent(model.features)
    optimizer = torch.optim.Adam(agent.parameters(), lr=actor_config.learning_rate)
    states = chronological_states(model, train, seed=3)
    starts = (
        torch.cat([s[:, : model.hidden] for s in states]),
        torch.cat([s[:, model.hidden :] for s in states]),
    )
    torch.save(
        {"agent": agent.state_dict(), "updates": 0, "config": actor_config.to_json()},
        args.output / "agent-0.pt",
    )
    generator = torch.Generator().manual_seed(args.seed + 10)
    history = train_actor(
        model,
        head,
        agent,
        optimizer,
        starts,
        actor_config,
        generator,
        args.actor_updates,
    )
    torch.save(
        {
            "agent": agent.state_dict(),
            "updates": args.actor_updates,
            "config": actor_config.to_json(),
        },
        args.output / f"agent-{args.actor_updates}.pt",
    )
    recent = history[-50:]
    write(
        args.output / "actor.json",
        {
            "config": actor_config.to_json(),
            "start_states": int(starts[0].shape[0]),
            "imagined_return_last_50": sum(h["imagined_return"] for h in recent)
            / len(recent),
            "imagined_reward_per_step_last_50": sum(
                h["imagined_reward_per_step"] for h in recent
            )
            / len(recent),
            "entropy_last": history[-1]["entropy_per_factor"],
            "history": history[::10],
        },
    )
    print(json.dumps({"imagined_return": recent[-1]["imagined_return"]}), flush=True)


if __name__ == "__main__":
    main()
