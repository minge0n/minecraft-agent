"""One bounded correction after the temporal attack diagnostic
(docs/stage4-temporal-diagnostic.md).

The diagnostic found that the continued world model imagines sandstone
mining wrongly: from real posterior states it predicts the center ray class
of sandstone in 1-8% of ticks and about 3 times the reward that the reward
head gives on the matching real states. This script:

1. Continues the continued block-breaking RSSM (`runs/stage4/break-continued`)
   for at most `--updates` updates: the same architecture, the canonical
   objective (G, pitch 128), Adam 1e-3, gradient-norm clip 1000. Each batch
   is 8 windows of the training lives and 8 windows of the Stage 3 training
   data. The training lives are the trained temporal actor's lives 0 and 1
   and the untrained temporal actor's lives 0 and 1. Lives 2 and 3 of both
   are held out and never trained on.
2. Trains one reward head on the new latent states with the existing
   `reward-break-v1` loss: the training lives above plus the random
   block-breaking training episodes. The old head is not reused.
3. Compares the old and the new model and head on the held-out lives with
   the diagnostic of `minecraft_temporal_diagnostic`, and runs the Stage 3
   one-step and imagination evaluation on `eval_episode` and `eval_seed`.

The new checkpoint is an experimental artifact. Nothing here changes the
canonical model or any earlier run.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from minecraft_rl import macos_qos, runtime
from minecraft_rl import minecraft_world_model_eval as evaluation
from minecraft_rl.minecraft_actor import (
    CANONICAL_OBJECTIVE,
    RewardHead,
    load_world_model,
)
from minecraft_rl.minecraft_break_train import break_data
from minecraft_rl.minecraft_dataset import join_segments, load_episodes
from minecraft_rl.minecraft_replay import CompactVocabulary, SequenceReplay
from minecraft_rl.minecraft_reward_model import (
    RewardTraining,
    chronological_states,
    train_frozen_head,
)
from minecraft_rl.minecraft_temporal_attack import TemporalBreakAgent
from minecraft_rl.minecraft_temporal_diagnostic import (
    AGENT,
    EVAL,
    RUN,
    load_life,
    open_loop_report,
    reward_head_report,
)
from minecraft_rl.minecraft_world_model import save_checkpoint
from minecraft_rl.provenance import git_commit

TRAIN_LIVES = (0, 1)
HELD_OUT_LIVES = (2, 3)
POLICIES = ("trained", "untrained")
STAGE3 = Path("runs/stage3/data")
RANDOM_BREAK = Path("runs/stage4/break-random2")
LENGTH, CONTEXT, HORIZON, EVAL_WINDOWS = 32, 12, 20, 128


def lives(policy: str, indices) -> list:
    joined = join_segments(load_episodes(EVAL / policy / "eval"))
    return sorted(
        (e for e in joined if e.meta.episode in indices), key=lambda e: e.meta.episode
    )


def continue_model(model, extra, compact, train_lives, args, path) -> list[dict]:
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.train()
    new = SequenceReplay(train_lives, compact, LENGTH)
    old = SequenceReplay(load_episodes(STAGE3 / "train"), compact, LENGTH)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-5)
    generator = torch.Generator().manual_seed(args.seed + 30)
    history = []
    for update in range(1, args.updates + 1):
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
        path,
        model,
        {
            **extra,
            "continued": {
                "from": str(RUN / "world_model.pt"),
                "updates": args.updates,
                "train_lives": {p: list(TRAIN_LIVES) for p in POLICIES},
                "held_out_lives": {p: list(HELD_OUT_LIVES) for p in POLICIES},
                "batch": "8 temporal-actor life windows + 8 Stage 3 windows of 32",
                "experimental": True,
            },
        },
    )
    return history


def train_head(model, compact, train_lives, args) -> tuple[RewardHead, dict]:
    random_eps = sorted(
        load_episodes(RANDOM_BREAK / "train"), key=lambda e: e.meta.episode
    )[:-6]
    data = break_data(model, train_lives + random_eps, compact)
    states = chronological_states(model, data, seed=7)
    config = RewardTraining(scale=10.0)
    head, history = train_frozen_head(
        states, data, model.features, config, args.head_updates, args.seed
    )
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head, {
        "training": config.to_json(),
        "episodes": len(data),
        "breaks": int(sum(int(d.rewards.gt(0).sum()) for d in data)),
        "history": history[-1:] if history else [],
    }


def stage3_evaluation(model, compact) -> dict[str, Any]:
    out = {}
    train_replay = SequenceReplay(load_episodes(STAGE3 / "train"), compact, LENGTH)
    unknown = evaluation.unknown_classes(compact, model)
    frequency_rays = evaluation.ray_frequency_baseline(train_replay, model, unknown)
    frequency_slot = evaluation.slot_frequency_baseline(train_replay)
    for split in ("eval_episode", "eval_seed"):
        episodes = load_episodes(STAGE3 / split)
        generator = torch.Generator().manual_seed(1234)
        one = SequenceReplay(episodes, compact, LENGTH)
        imagine = SequenceReplay(episodes, compact, CONTEXT + HORIZON)
        windows = one.evaluation_windows(LENGTH)
        step = max(1, len(windows) // EVAL_WINDOWS)
        result_one = evaluation.one_step(
            model,
            one,
            windows[::step][:EVAL_WINDOWS],
            CONTEXT,
            unknown,
            frequency_rays,
            frequency_slot,
            generator,
            events=False,
        )
        windows = imagine.evaluation_windows(CONTEXT + HORIZON)
        step = max(1, len(windows) // EVAL_WINDOWS)
        result_imagine = evaluation.imagination(
            model,
            imagine,
            windows[::step][:EVAL_WINDOWS],
            CONTEXT,
            unknown,
            generator,
        )
        model_metrics = result_one["all"]["model"]
        out[split] = {
            "one_step": {
                key: model_metrics[key]["mean"]
                for key in (
                    "ray_accuracy",
                    "ray_accuracy_changed",
                    "ray_distance_mae_blocks",
                    "pitch_mae",
                )
            },
            "latent": {
                key: result_one["latent"][key]
                for key in ("kl_mean", "active_variables", "classes_used")
            },
            "imagination": {
                f"horizon_{k}": {
                    key: result_imagine[f"horizon_{k}"]["model"][key]["mean"]
                    for key in (
                        "ray_accuracy",
                        "ray_accuracy_changed",
                        "ray_distance_mae_blocks",
                        "pitch_mae",
                    )
                }
                for k in evaluation.HORIZONS
            },
        }
    return out


@torch.no_grad()
def held_out_report(model, compact, head, agent, args) -> dict[str, Any]:
    out = {}
    generator = torch.Generator().manual_seed(args.seed + 1)
    for policy in POLICIES:
        recorded = [
            load_life(e, compact, model, generator)
            for e in lives(policy, HELD_OUT_LIVES)
        ]
        out[policy] = {
            "reward_head": reward_head_report(recorded, head),
            "open_loop": open_loop_report(
                model, head, agent, recorded, args.stride, generator
            ),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/stage4/temporal-correction")
    )
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--head-updates", type=int, default=3000)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.updates > 1000:
        parser.error("the correction budget is at most 1,000 updates")
    macos_qos.run_in_background()
    runtime.configure(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    original, extra = load_world_model(RUN / "world_model.pt")
    compact = CompactVocabulary.from_json(extra["vocabulary"])
    saved = torch.load(RUN / "reward_head.pt", weights_only=True)
    original_head = RewardHead(original.features)
    original_head.load_state_dict(saved["reward_head"])
    original_head.scale = saved["scale"]
    original_head.eval()
    agent = TemporalBreakAgent(original.features)
    agent.load_state_dict(torch.load(AGENT, weights_only=True)["agent"])
    agent.eval()
    train_lives = [e for p in POLICIES for e in lives(p, TRAIN_LIVES)]

    model_path = args.output / "world_model.pt"
    history = None
    if not model_path.exists():
        model, _ = load_world_model(RUN / "world_model.pt")
        history = continue_model(model, extra, compact, train_lives, args, model_path)
    corrected, _ = load_world_model(model_path)
    head_path = args.output / "reward_head.pt"
    if head_path.exists():
        saved = torch.load(head_path, weights_only=True)
        head = RewardHead(corrected.features)
        head.load_state_dict(saved["reward_head"])
        head.scale = saved["scale"]
        head.eval()
        head_info = saved.get("info", {})
    else:
        head, head_info = train_head(corrected, compact, train_lives, args)
        torch.save(
            {"reward_head": head.state_dict(), "scale": head.scale, "info": head_info},
            head_path,
        )
    report = {
        "git_commit": git_commit(),
        "updates": args.updates,
        "train_lives": list(TRAIN_LIVES),
        "held_out_lives": list(HELD_OUT_LIVES),
        "continue_history": history,
        "reward_head": head_info,
        "held_out": {
            "original": held_out_report(original, compact, original_head, agent, args),
            "corrected": held_out_report(corrected, compact, head, None, args),
        },
        "stage3": {
            "original": stage3_evaluation(original, compact),
            "corrected": stage3_evaluation(corrected, compact),
        },
    }
    (args.output / "correction.json").write_text(json.dumps(report, indent=1))
    print("written", args.output / "correction.json", flush=True)


if __name__ == "__main__":
    main()
