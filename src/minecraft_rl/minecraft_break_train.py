"""The block-breaking diagnostic: reward head and pitch-and-attack actor on
`reward-break-v1` (docs/stage4-block-breaking.md).

The reward of transition t is the number of blocks broken in step t
(`privileged_breaks`), never a policy input.

Stages, each written to `--output` and skipped when its result exists:

1. `reward_head`: the existing reward head on whole-episode states of the
   frozen canonical RSSM, trained on the training episodes and evaluated on
   held-out episodes: break recall, false positives and ranking, against the
   zero baseline, and the prediction by length of the current attack run.
2. `actor`: the existing actor-critic in imagination with `BreakAgent`
   (pitch and attack free), checkpoints at 0 and `--actor-updates`. Only
   runs when the reward head passes `--min-held-out-ap`.
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
from minecraft_rl.minecraft_dataset import ACTION_BUTTONS, load_episodes
from minecraft_rl.minecraft_replay import CompactVocabulary
from minecraft_rl.minecraft_reward_break import BREAK_V1, BREAKS_KEY, BreakAgent
from minecraft_rl.minecraft_reward_model import (
    RewardTraining,
    average_precision,
    chronological_states,
    episode_data,
    train_frozen_head,
)
from minecraft_rl.provenance import git_commit

WORLD_MODEL = Path("runs/stage3/tradeoff-P128/seed0/world_model.pt")
ATTACK = ACTION_BUTTONS.index("attack")
THRESHOLDS = (0.1, 0.3, 0.5)


def attack_run_lengths(attack: numpy.ndarray) -> numpy.ndarray:
    """For each step, the number of consecutive held-attack steps ending at it."""
    out = numpy.zeros(len(attack), dtype=numpy.int64)
    run = 0
    for t, held in enumerate(attack):
        run = run + 1 if held else 0
        out[t] = run
    return out


def break_data(model, episodes, compact):
    data = episode_data(model, episodes, compact)
    for d, e in zip(data, episodes, strict=True):
        d.rewards = torch.as_tensor(e.arrays[BREAKS_KEY]).float() * BREAK_V1.per_break
        run = attack_run_lengths(e.arrays["action_buttons"][:, ATTACK])
        d.labels = {"positive": d.rewards > 0, "attack_run": torch.as_tensor(run)}
    return data


def evaluate(model, head, data) -> dict[str, Any]:
    states = chronological_states(model, data, seed=7)
    with torch.no_grad():
        predicted = torch.cat([head.reward(s[1:]) for s in states])
    target = torch.cat([d.rewards for d in data])
    positive = torch.cat([d.labels["positive"] for d in data])
    run = torch.cat([d.labels["attack_run"] for d in data])
    out: dict[str, Any] = {
        "transitions": int(target.numel()),
        "breaks": int(positive.sum()),
        "model_mae": float((predicted - target).abs().mean()),
        "zero_baseline_mae": float(target.abs().mean()),
        "positive_mae": float((predicted - target)[positive].abs().mean())
        if positive.any()
        else None,
        "positive_predicted_mean": float(predicted[positive].mean())
        if positive.any()
        else None,
        "zero_predicted_mean": float(predicted[~positive].mean()),
        "average_precision": average_precision(predicted, positive),
        "average_precision_chance": float(positive.float().mean()),
        "predicted_per_step": float(predicted.mean()),
        "actual_per_step": float(target.mean()),
    }
    for threshold in THRESHOLDS:
        called = predicted >= threshold
        hits = int((called & positive).sum())
        out[f"at_{threshold}"] = {
            "called": int(called.sum()),
            "precision": hits / int(called.sum()) if called.any() else None,
            "recall": hits / int(positive.sum()) if positive.any() else None,
            "false_positive_rate": float(called[~positive].float().mean()),
        }
    # Does the prediction change with attack history? Zero-reward steps by
    # the length of the current attack run.
    out["zero_steps_by_attack_run"] = {}
    for low, high in ((0, 1), (1, 10), (10, 15), (15, 30), (30, 10**6)):
        mask = (~positive) & (run >= low) & (run < high)
        n = int(mask.sum())
        out["zero_steps_by_attack_run"][f"{low}..{high}"] = {
            "count": n,
            "predicted_mean": float(predicted[mask].mean()) if n else None,
        }
    return out


def continue_world_model(model, extra, compact, train_eps, stage3, args, out_path):
    """Ordinary continued training of the canonical RSSM: same architecture,
    same objective (G, pitch 128), the Stage 3 optimizer settings. Each batch
    is half new training windows (block-breaking data, held-out episodes
    excluded) and half Stage 3 training windows, so the old dynamics keep
    their data."""
    from minecraft_rl.minecraft_actor import CANONICAL_OBJECTIVE
    from minecraft_rl.minecraft_replay import SequenceReplay
    from minecraft_rl.minecraft_world_model import save_checkpoint

    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.train()
    new = SequenceReplay(train_eps, compact, 32)
    old = SequenceReplay(load_episodes(stage3 / "train"), compact, 32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-5)
    generator = torch.Generator().manual_seed(args.seed + 20)
    history = []
    for update in range(1, args.continue_updates + 1):
        total = 0.0
        optimizer.zero_grad()
        for replay in (new, old):
            batch = replay.sample(8, generator)
            terms = model.losses(
                batch.observations,
                batch.actions,
                batch.continues,
                generator,
                CANONICAL_OBJECTIVE,
            )
            (terms["total"] / 2).backward()
            total += float(terms["total"].detach()) / 2
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1000.0)
        optimizer.step()
        if update % 50 == 0:
            history.append({"update": update, "total": total, "grad_norm": float(norm)})
            print(json.dumps(history[-1]), flush=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    save_checkpoint(
        out_path,
        model,
        {
            **extra,
            "continued": {
                "from": str(WORLD_MODEL),
                "updates": args.continue_updates,
                "new_episodes": len(train_eps),
                "batch": "8 new + 8 Stage 3 windows of 32",
            },
        },
    )
    return history


def write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("runs/stage4/break-random2")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-episodes", type=int, default=6)
    parser.add_argument("--reward-updates", type=int, default=3000)
    parser.add_argument("--actor-updates", type=int, default=200)
    parser.add_argument("--min-ap-ratio", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--world-model",
        type=Path,
        default=WORLD_MODEL,
        help="the canonical checkpoint, or a continued one",
    )
    parser.add_argument("--continue-updates", type=int, default=0)
    parser.add_argument("--stage3-data", type=Path, default=Path("runs/stage3/data"))
    args = parser.parse_args()
    macos_qos.run_in_background()
    runtime.configure(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    model, extra = load_world_model(args.world_model)
    compact = CompactVocabulary.from_json(extra["vocabulary"])
    episodes = sorted(
        load_episodes(args.dataset / "train"), key=lambda e: e.meta.episode
    )
    test_eps = episodes[-args.test_episodes :]
    train_eps = episodes[: -args.test_episodes]
    if args.continue_updates:
        out_path = args.output / "world_model.pt"
        if not out_path.exists():
            history = continue_world_model(
                model, extra, compact, train_eps, args.stage3_data, args, out_path
            )
            write(args.output / "continue.json", {"history": history})
        model, extra = load_world_model(out_path)
    train = break_data(model, train_eps, compact)
    test = break_data(model, test_eps, compact)
    # Rare positive events: the reward loss of the first actor experiment
    # (positive weight 20, half the chunks centered on a break).
    config = RewardTraining(scale=10.0)
    head_path = args.output / "reward_head.pt"
    if not head_path.exists():
        model.eval()
        states = chronological_states(model, train, seed=7)
        head, history = train_frozen_head(
            states, train, model.features, config, args.reward_updates, args.seed
        )
        out = {
            "git_commit": git_commit(),
            "task": BREAK_V1.to_json(),
            "training": config.to_json(),
            "train_episodes": len(train),
            "held_out_episodes": len(test),
            "train": evaluate(model, head, train),
            "held_out": evaluate(model, head, test),
            "history": history,
        }
        write(args.output / "reward_head.json", out)
        torch.save({"reward_head": head.state_dict(), "scale": head.scale}, head_path)
        print(
            json.dumps(
                {
                    k: {
                        m: out[k][m]
                        for m in (
                            "breaks",
                            "average_precision",
                            "positive_predicted_mean",
                        )
                    }
                    for k in ("train", "held_out")
                }
            ),
            flush=True,
        )
    path = args.output / "cross_validation.json"
    if not path.exists():
        # Every episode held out once: 5 folds of whole episodes, so all
        # breaks are scored by a head that never trained on their episode.
        model.eval()
        everything = train + test
        states = chronological_states(model, everything, seed=7)
        folds = 5
        predicted, target, positive, run = [], [], [], []
        for k in range(folds):
            held = [i for i in range(len(everything)) if i % folds == k]
            fit = [i for i in range(len(everything)) if i % folds != k]
            h, _ = train_frozen_head(
                [states[i] for i in fit],
                [everything[i] for i in fit],
                model.features,
                config,
                args.reward_updates,
                args.seed,
            )
            with torch.no_grad():
                for i in held:
                    predicted.append(h.reward(states[i][1:]))
                    target.append(everything[i].rewards)
                    positive.append(everything[i].labels["positive"])
                    run.append(everything[i].labels["attack_run"])
        p_all, t_all = torch.cat(predicted), torch.cat(target)
        pos, r_all = torch.cat(positive), torch.cat(run)
        out = {
            "folds": folds,
            "episodes": len(everything),
            "breaks": int(pos.sum()),
            "average_precision": average_precision(p_all, pos),
            "average_precision_chance": float(pos.float().mean()),
            "positive_predicted_mean": float(p_all[pos].mean()),
            "positive_predicted": [round(float(x), 3) for x in p_all[pos]],
            "zero_predicted_mean": float(p_all[~pos].mean()),
            "zero_baseline_mae": float(t_all.abs().mean()),
            "model_mae": float((p_all - t_all).abs().mean()),
        }
        for threshold in THRESHOLDS:
            called = p_all >= threshold
            hits = int((called & pos).sum())
            out[f"at_{threshold}"] = {
                "called": int(called.sum()),
                "precision": hits / int(called.sum()) if called.any() else None,
                "recall": hits / int(pos.sum()),
                "false_positive_rate": float(called[~pos].float().mean()),
            }
        out["zero_steps_by_attack_run"] = {
            f"{low}..{high}": float(
                p_all[(~pos) & (r_all >= low) & (r_all < high)].mean()
            )
            for low, high in ((0, 1), (1, 15), (15, 30), (30, 10**6))
        }
        write(path, out)
        print(
            json.dumps(
                {
                    k: out[k]
                    for k in (
                        "breaks",
                        "average_precision",
                        "positive_predicted_mean",
                        "at_0.3",
                    )
                }
            ),
            flush=True,
        )

    # The gate: cross-validated average precision over every break, at least
    # `--min-ap-ratio` times chance. The fixed held-out split has 2 breaks,
    # too few for a gate (docs/stage4-block-breaking.md).
    cv = json.loads((args.output / "cross_validation.json").read_text())
    ratio = cv["average_precision"] / cv["average_precision_chance"]
    if ratio < args.min_ap_ratio:
        print(f"reward head fails: cross-validated AP {ratio:.1f} times chance")
        return
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
    agent = BreakAgent(model.features)
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
            "action_space": "pitch and attack only, every other factor at no-op",
            "start_states": int(starts[0].shape[0]),
            "imagined_return_first_10": sum(h["imagined_return"] for h in history[:10])
            / 10,
            "imagined_return_last_50": sum(h["imagined_return"] for h in recent)
            / len(recent),
            "imagined_reward_per_step_last_50": sum(
                h["imagined_reward_per_step"] for h in recent
            )
            / len(recent),
            "entropy_last": {
                k: history[-1]["entropy_per_factor"][k] for k in ("pitch", "attack")
            },
            "history": history[::10],
        },
    )
    print(json.dumps({"imagined_return": recent[-1]["imagined_return"]}), flush=True)


if __name__ == "__main__":
    main()
