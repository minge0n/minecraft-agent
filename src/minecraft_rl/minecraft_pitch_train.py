"""The pitch-control diagnostic: reward head and pitch-only actor on
`reward-pitch-v1` (docs/stage4-pitch-control.md).

Existing episodes are relabeled: the reward of transition t is computed from
the recorded `obs_pitch[t + 1]`. The original reward arrays are not changed.

Stages, each written to `--output` and skipped when its result exists:

1. `reward_head`: the existing reward head on whole-episode states of the
   frozen canonical RSSM, trained on the training episodes, evaluated on the
   held-out random episodes and the held-out actor episodes.
2. `actor`: the existing actor-critic in imagination with a pitch-only
   actor, checkpoints at 0 and `--actor-updates`.
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
    RewardHead,
    load_world_model,
    train_actor,
)
from minecraft_rl.minecraft_dataset import load_episodes
from minecraft_rl.minecraft_replay import CompactVocabulary
from minecraft_rl.minecraft_reward_model import (
    RewardTraining,
    chronological_states,
    episode_data,
    train_frozen_head,
)
from minecraft_rl.minecraft_reward_pitch import PITCH_V1, PitchOnlyAgent, pitch_rewards
from minecraft_rl.provenance import git_commit

WORLD_MODEL = Path("runs/stage3/tradeoff-P128/seed0/world_model.pt")
RANDOM = ("random", "control-random")
ACTOR = (
    "eval-actor0",
    "eval-actor200",
    "eval-actor1000",
    "eval-onpolicy1000",
    "eval-actor-r3",
)
# Pitch ranges of the next observation for the error table.
RANGES = ((-90, -10), (-10, 10), (10, 25), (25, 35), (35, 50), (50, 90))


def pitch_data(model, episodes, compact):
    data = episode_data(model, episodes, compact)
    for d, e in zip(data, episodes, strict=True):
        d.rewards = torch.as_tensor(pitch_rewards(e.arrays["obs_pitch"])).float()
        d.labels = {"next_pitch": torch.as_tensor(e.arrays["obs_pitch"][1:]).float()}
    return data


def split(root: Path, name: str):
    episodes = sorted(
        load_episodes(root / name / "train"), key=lambda e: e.meta.episode
    )
    held = episodes[-1:] if name in ACTOR else episodes[-3:]
    return episodes[: len(episodes) - len(held)], held


def evaluate(model, head, data, mean: float) -> dict[str, Any]:
    states = chronological_states(model, data, seed=7)
    with torch.no_grad():
        predicted = torch.cat([head.reward(s[1:]) for s in states])
    target = torch.cat([d.rewards for d in data])
    pitch = torch.cat([d.labels["next_pitch"] for d in data])
    out: dict[str, Any] = {
        "transitions": int(target.numel()),
        "model_mae": float((predicted - target).abs().mean()),
        "zero_baseline_mae": float(target.abs().mean()),
        "mean_baseline_mae": float((target - mean).abs().mean()),
        # The exact reward from the observed pitch is the reference: error 0.
        "exact_reference_mae": 0.0,
        "correlation": float(torch.corrcoef(torch.stack([predicted, target]))[0, 1]),
        "predicted_per_step": float(predicted.mean()),
        "actual_per_step": float(target.mean()),
        "by_pitch": {},
    }
    for low, high in RANGES:
        mask = (pitch >= low) & (pitch < high)
        n = int(mask.sum())
        out["by_pitch"][f"{low}..{high}"] = {
            "count": n,
            "predicted_mean": float(predicted[mask].mean()) if n else None,
            "actual_mean": float(target[mask].mean()) if n else None,
            "model_mae": float((predicted - target)[mask].abs().mean()) if n else None,
        }
    # Spurious reward: at the same pitch, does the prediction depend on what
    # else happens? The spread of the prediction within 2-degree pitch bins,
    # against the spread of the target there (near 0 by construction).
    bins = torch.floor(pitch / 2.0)
    spreads = []
    for b in torch.unique(bins):
        mask = bins == b
        if mask.sum() >= 20:
            spreads.append(float(predicted[mask].std()))
    out["prediction_sd_within_2_degree_bins"] = (
        float(numpy.median(spreads)) if spreads else None
    )
    out["target_sd_within_2_degree_bins"] = 2.0 / PITCH_V1.scale / numpy.sqrt(12)
    return out


def write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/stage4"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reward-updates", type=int, default=3000)
    parser.add_argument("--actor-updates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    macos_qos.run_in_background()
    runtime.configure(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    model, extra = load_world_model(WORLD_MODEL)
    compact = CompactVocabulary.from_json(extra["vocabulary"])
    train_eps, held = [], {"random": [], "actor": []}
    for name in RANDOM + ACTOR:
        fit, test = split(args.root, name)
        train_eps += fit
        held["actor" if name in ACTOR else "random"] += test
    train = pitch_data(model, train_eps, compact)
    tests = {k: pitch_data(model, v, compact) for k, v in held.items()}
    # Dense rewards in [-1.33, 0]: no positive weighting, uniform chunks;
    # scale 10 brings them to about unit size.
    config = RewardTraining(scale=10.0, positive_weight=1.0, positive_chunk_share=0.0)
    head_path = args.output / "reward_head.pt"
    if not head_path.exists():
        model.eval()
        states = chronological_states(model, train, seed=7)
        head, history = train_frozen_head(
            states, train, model.features, config, args.reward_updates, args.seed
        )
        mean = float(torch.cat([d.rewards for d in train]).mean())
        out = {
            "git_commit": git_commit(),
            "task": PITCH_V1.to_json(),
            "training": config.to_json(),
            "train_episodes": len(train),
            "train_transitions": sum(d.steps for d in train),
            "held_out_episodes": {k: len(v) for k, v in tests.items()},
            "train": evaluate(model, head, train, mean),
            **{
                f"held_out_{k}": evaluate(model, head, v, mean)
                for k, v in tests.items()
            },
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
                    for k in ("train", "held_out_random", "held_out_actor")
                }
            ),
            flush=True,
        )
    if (
        args.actor_updates <= 0
        or (args.output / f"agent-{args.actor_updates}.pt").exists()
    ):
        return
    saved = torch.load(head_path, weights_only=True)
    head = RewardHead(model.features)
    head.load_state_dict(saved["reward_head"])
    head.scale = saved["scale"]
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    config_actor = ActorConfig(seed=args.seed)
    torch.manual_seed(args.seed)
    agent = PitchOnlyAgent(model.features)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config_actor.learning_rate)
    states = chronological_states(model, train, seed=3)
    starts = (
        torch.cat([s[:, : model.hidden] for s in states]),
        torch.cat([s[:, model.hidden :] for s in states]),
    )
    torch.save(
        {"agent": agent.state_dict(), "updates": 0, "config": config_actor.to_json()},
        args.output / "agent-0.pt",
    )
    generator = torch.Generator().manual_seed(args.seed + 10)
    history = train_actor(
        model,
        head,
        agent,
        optimizer,
        starts,
        config_actor,
        generator,
        args.actor_updates,
    )
    torch.save(
        {
            "agent": agent.state_dict(),
            "updates": args.actor_updates,
            "config": config_actor.to_json(),
        },
        args.output / f"agent-{args.actor_updates}.pt",
    )
    recent = history[-50:]
    write(
        args.output / "actor.json",
        {
            "config": config_actor.to_json(),
            "action_space": "pitch factor only, every other factor at its no-op value",
            "start_states": int(starts[0].shape[0]),
            "imagined_return_first_10": sum(h["imagined_return"] for h in history[:10])
            / 10,
            "imagined_return_last_50": sum(h["imagined_return"] for h in recent)
            / len(recent),
            "imagined_reward_per_step_last_50": sum(
                h["imagined_reward_per_step"] for h in recent
            )
            / len(recent),
            "pitch_entropy_last": history[-1]["entropy_per_factor"]["pitch"],
            "history": history[::10],
        },
    )
    print(json.dumps({"imagined_return": recent[-1]["imagined_return"]}), flush=True)


if __name__ == "__main__":
    main()
