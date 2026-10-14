"""Reward-prediction experiment on the Minecraft RSSM
(docs/stage4-reward-prediction.md).

Stages, each written to `--output` and skipped when its result exists:

1. `alignment`: the head of the first actor experiment on held-out and
   exploit episodes, with h restarted every 32 steps (as it was trained and
   evaluated) and with h carried over the whole episode (as the actor acts).
2. `tiny`: a reward head on chronological frozen states of one episode must
   fit its own positive steps.
3. `frozen`: a reward head on chronological frozen states of the training
   episodes, evaluated on held-out and exploit episodes.
4. `recurrent`: the reward-aware partial unfreeze (`RecurrentTrainer`), then
   the same evaluation.
5. `world_model`: one-step and imagination metrics of the canonical and the
   partially unfrozen model on the Stage 3 held-out splits.

Exploit episodes (`--exploit`) are evaluation only: no stage trains on them.
"""

import argparse
import copy
import json
import time
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
from minecraft_rl.minecraft_dataset import load_episodes
from minecraft_rl.minecraft_replay import CompactVocabulary, SequenceReplay
from minecraft_rl.minecraft_reward_model import (
    TRAINABLE_MODULES,
    RecurrentTrainer,
    RewardTraining,
    chronological_states,
    episode_data,
    history_labels,
    latent_ambiguity,
    probe_auc,
    report_on,
    train_frozen_head,
    window_states,
)
from minecraft_rl.minecraft_world_model import save_checkpoint
from minecraft_rl.provenance import git_commit

WORLD_MODEL = Path("runs/stage3/tradeoff-P128/seed0/world_model.pt")
OLD_HEAD = Path("runs/stage4/actor-r1/reward_head.pt")
EXPLOIT = ("eval-actor0", "eval-actor200", "eval-actor1000", "eval-onpolicy1000")


def write(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=1))


def load_head(path: Path, features: int) -> RewardHead:
    saved = torch.load(path, weights_only=True)
    head = RewardHead(features)
    head.load_state_dict(saved["reward_head"])
    head.scale = saved["scale"]
    head.eval()
    return head


def evaluate_head(model, head, sets) -> dict[str, Any]:
    out = {}
    for name, data in sets.items():
        model.eval()
        states = chronological_states(model, data, seed=7)
        out[name] = report_on(head, states, data)
    return out


def world_model_metrics(model, compact: CompactVocabulary, dataset: Path) -> dict:
    """The Stage 3 evaluation: 128 windows per split, context 12, horizon 20."""
    train = SequenceReplay(load_episodes(dataset / "train"), compact, 32)
    unknown = evaluation.unknown_classes(compact, model)
    rays = evaluation.ray_frequency_baseline(train, model, unknown)
    slot = evaluation.slot_frequency_baseline(train)
    out = {}
    for split in ("eval_episode", "eval_seed"):
        episodes = load_episodes(dataset / split)
        one = SequenceReplay(episodes, compact, 32)
        imagine = SequenceReplay(episodes, compact, 32)
        windows = {}
        for kind, replay in (("one_step", one), ("imagination", imagine)):
            index = replay.evaluation_windows(replay.length)
            step = len(index) / 128
            windows[kind] = [index[int(i * step)] for i in range(128)]
        generator = torch.Generator().manual_seed(1234)
        step_result = evaluation.one_step(
            model, one, windows["one_step"], 12, unknown, rays, slot, generator
        )
        imagined = evaluation.imagination(
            model, imagine, windows["imagination"], 12, unknown, generator
        )
        keep = (
            "ray_accuracy",
            "ray_accuracy_changed",
            "ray_distance_mae_blocks",
            "pitch_mae",
        )
        out[split] = {
            "one_step": {k: step_result["all"]["model"][k]["mean"] for k in keep},
            "persistence": {
                k: step_result["all"]["persistence"][k]["mean"] for k in keep
            },
            "latent": {
                k: step_result["latent"][k]
                for k in ("kl_mean", "active_variables", "classes_used")
            },
            "imagination": {
                f"horizon_{h}": {
                    k: imagined[f"horizon_{h}"]["model"][k]["mean"] for k in keep
                }
                for h in (1, 5, 10, 20)
            },
        }
    return out


def history_probe(model, train_eps, test_eps, train, test) -> dict:
    """Does h remember a block type that is out of view now? Probes on h, and
    on the current-observation embedding e as the no-memory control."""
    model.eval()
    train_states = chronological_states(model, train, seed=7)
    test_states = chronological_states(model, test, seed=7)
    counts = {}
    for episode in train_eps:
        kinds, types = episode.arrays["obs_ray_kind"], episode.arrays["obs_ray_type"]
        for k in set(types[kinds == 1].tolist()):
            counts[k] = counts.get(k, 0) + 1
    common = sorted(k for k, n in counts.items() if n >= 5)
    labels = {
        "train": torch.cat([history_labels(e, common) for e in train_eps]),
        "test": torch.cat([history_labels(e, common) for e in test_eps]),
    }
    inputs = {
        "h": (
            torch.cat([s[:, : model.hidden] for s in train_states]),
            torch.cat([s[:, : model.hidden] for s in test_states]),
        ),
        "embedding": (
            torch.cat([d.embedded for d in train]),
            torch.cat([d.embedded for d in test]),
        ),
    }
    out = {"block_types": common, "per_type": {}}
    for k, block in enumerate(common):
        y_train, y_test = labels["train"][:, k], labels["test"][:, k]
        out["per_type"][str(block)] = {
            "test_positive_states": int(y_test.sum()),
            **{
                name: probe_auc(x_train, y_train, x_test, y_test)
                for name, (x_train, x_test) in inputs.items()
            },
        }
    for name in inputs:
        values = [v[name] for v in out["per_type"].values() if v[name] is not None]
        out[f"{name}_mean_auc"] = sum(values) / len(values) if values else None
        out[f"{name}_types"] = len(values)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("runs/stage4/random"))
    parser.add_argument("--exploit-root", type=Path, default=Path("runs/stage4"))
    parser.add_argument("--stage3-data", type=Path, default=Path("runs/stage3/data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-episodes", type=int, default=3)
    parser.add_argument("--frozen-updates", type=int, default=3000)
    parser.add_argument("--tiny-updates", type=int, default=1000)
    parser.add_argument("--recurrent-updates", type=int, default=1000)
    parser.add_argument("--core-learning-rate", type=float, default=1e-4)
    parser.add_argument("--refresh-every", type=int, default=100)
    parser.add_argument("--actor-held-out", default="3")
    parser.add_argument(
        "--stages", default="alignment,tiny,frozen,recurrent,world_model"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    macos_qos.run_in_background()
    runtime.configure(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    stages = set(args.stages.split(","))
    model, extra = load_world_model(WORLD_MODEL)
    compact = CompactVocabulary.from_json(extra["vocabulary"])
    episodes = sorted(
        load_episodes(args.dataset / "train"), key=lambda e: e.meta.episode
    )
    test_eps = episodes[-args.test_episodes :]
    train_eps = episodes[: -args.test_episodes]
    exploit_eps = [
        e for name in EXPLOIT for e in load_episodes(args.exploit_root / name / "train")
    ]
    started = time.perf_counter()
    train = episode_data(model, train_eps, compact)
    sets = {
        "held_out": episode_data(model, test_eps, compact),
        "exploit": episode_data(model, exploit_eps, compact),
    }
    config = RewardTraining()
    meta = {
        "git_commit": git_commit(),
        "world_model": str(WORLD_MODEL),
        "dataset": str(args.dataset),
        "train_episodes": [d.episode for d in train],
        "held_out_episodes": [d.episode for d in sets["held_out"]],
        "exploit_datasets": list(EXPLOIT),
        "train_positive_steps": int(
            sum(int(d.labels["positive"].sum()) for d in train)
        ),
        "training": config.to_json(),
        "runtime": runtime.metadata(torch.device("cpu")),
        "prepare_seconds": time.perf_counter() - started,
    }
    write(args.output / "meta.json", meta)
    print(json.dumps({k: meta[k] for k in ("train_positive_steps", "prepare_seconds")}))

    path = args.output / "alignment.json"
    if "alignment" in stages and not path.exists():
        head = load_head(OLD_HEAD, model.features)
        out = {}
        for name, data in sets.items():
            out[name] = {
                "chronological": report_on(
                    head, chronological_states(model, data, seed=7), data
                ),
            }
            # Window states cover the first 992 steps of each episode only.
            trimmed = [copy.copy(d) for d in data]
            windowed = []
            for d in trimmed:
                states = window_states(model, d, 32, seed=7)
                n = states.shape[0]
                d.rewards = d.rewards[:n]
                d.labels = {k: v[:n] for k, v in d.labels.items()}
                windowed.append(torch.cat([states[:1], states]))
            out[name]["windows_32"] = report_on(head, windowed, trimmed)
        write(path, out)
        print(
            "alignment",
            json.dumps(
                {
                    k: {m: v[m]["predicted_reward_per_step"] for m in v}
                    for k, v in out.items()
                }
            ),
        )

    path = args.output / "tiny.json"
    if "tiny" in stages and not path.exists():
        model.eval()
        one = train[:1]
        states = chronological_states(model, one, seed=7)
        head, history = train_frozen_head(
            states, one, model.features, config, args.tiny_updates, args.seed
        )
        write(
            path,
            {
                "episode": one[0].episode,
                "fit": report_on(head, states, one),
                "history": history,
            },
        )
        print("tiny", json.dumps(report_on(head, states, one)["at_0.005"]))

    path = args.output / "frozen.json"
    if "frozen" in stages and not path.exists():
        model.eval()
        states = chronological_states(model, train, seed=7)
        head, history = train_frozen_head(
            states, train, model.features, config, args.frozen_updates, args.seed
        )
        torch.save(
            {"reward_head": head.state_dict(), "scale": head.scale},
            args.output / "frozen_head.pt",
        )
        out = {
            "train": report_on(head, states, train),
            **evaluate_head(model, head, sets),
            "history": history,
        }
        write(path, out)
        print(
            "frozen",
            json.dumps(
                {
                    k: out[k]["average_precision"]
                    for k in ("train", "held_out", "exploit")
                }
            ),
        )

    path = args.output / "recurrent.json"
    model_path = args.output / "world_model_recurrent.pt"
    if "recurrent" in stages and not path.exists():
        trained = copy.deepcopy(model)
        trainer = RecurrentTrainer(
            trained,
            CANONICAL_OBJECTIVE,
            train,
            config,
            args.core_learning_rate,
            args.refresh_every,
            args.seed,
        )
        state_path = args.output / "recurrent_state.pt"
        if state_path.exists():
            trainer.load(torch.load(state_path, weights_only=False))
        begun = time.perf_counter()
        while trainer.update < args.recurrent_updates:
            record = trainer.step()
            if trainer.update % 100 == 0:
                torch.save(trainer.state(), state_path)
                print(json.dumps(record), flush=True)
        trained.eval()
        for parameter in trained.parameters():
            parameter.requires_grad_(False)
        save_checkpoint(
            model_path,
            trained,
            {
                **extra,
                "reward_aware": {
                    "trainable_modules": list(TRAINABLE_MODULES),
                    "updates": trainer.update,
                    "core_learning_rate": args.core_learning_rate,
                },
            },
        )
        torch.save(
            {"reward_head": trainer.head.state_dict(), "scale": trainer.head.scale},
            args.output / "recurrent_head.pt",
        )
        trainer.head.eval()
        out = {
            "trainable_modules": list(TRAINABLE_MODULES),
            "trainable_parameters": sum(p.numel() for p in trainer.trainable),
            "updates": trainer.update,
            "seconds": time.perf_counter() - begun,
            "train": report_on(
                trainer.head, chronological_states(trained, train, seed=7), train
            ),
            **evaluate_head(trained, trainer.head, sets),
            "history": trainer.history,
        }
        write(path, out)
        print(
            "recurrent",
            json.dumps(
                {
                    k: out[k]["average_precision"]
                    for k in ("train", "held_out", "exploit")
                }
            ),
        )

    path = args.output / "actor_data.json"
    if "actor_data" in stages and not path.exists():
        # Whole actor episodes, split before any training: one episode of each
        # actor policy is held out (--actor-held-out), the rest train.
        held = int(args.actor_held_out)
        actor_train, actor_test = [], []
        for name in EXPLOIT:
            for e in load_episodes(args.exploit_root / name / "train"):
                (actor_test if e.meta.episode == held else actor_train).append(e)
        a_train = episode_data(model, actor_train, compact)
        evaluation_sets = {
            "random_held_out": sets["held_out"],
            "actor_held_out": episode_data(model, actor_test, compact),
        }
        model.eval()
        mixed = train + a_train
        states = chronological_states(model, mixed, seed=7)
        head, history = train_frozen_head(
            states, mixed, model.features, config, args.frozen_updates, args.seed
        )
        torch.save(
            {"reward_head": head.state_dict(), "scale": head.scale},
            args.output / "actor_data_head.pt",
        )
        heads = {
            "first_actor_head": load_head(OLD_HEAD, model.features),
            "A_random_only": load_head(args.output / "frozen_head.pt", model.features),
            "B_random_and_actor": head,
        }
        out = {
            "split": {
                "random_train_episodes": len(train),
                "actor_train_episodes": len(a_train),
                "actor_train_positive_steps": int(
                    sum(int(d.labels["positive"].sum()) for d in a_train)
                ),
                "random_held_out_episodes": len(sets["held_out"]),
                "actor_held_out_episodes": len(actor_test),
                "actor_held_out": [f"{name}/{held}" for name in EXPLOIT],
            },
            "updates": args.frozen_updates,
            "history": history,
            "reports": {
                name: evaluate_head(model, h, evaluation_sets)
                for name, h in heads.items()
            },
            "latent_ambiguity": {
                name: latent_ambiguity(chronological_states(model, data, seed=7), data)
                for name, data in evaluation_sets.items()
            },
        }
        write(path, out)
        print("actor_data", json.dumps(out["split"]))

    path = args.output / "history_probe.json"
    if "history_probe" in stages and not path.exists():
        out = {
            "canonical": history_probe(
                model, train_eps, test_eps, train, sets["held_out"]
            )
        }
        if model_path.exists():
            recurrent, _ = load_world_model(model_path)
            rtrain = episode_data(recurrent, train_eps, compact)
            rtest = episode_data(recurrent, test_eps, compact)
            out["recurrent"] = history_probe(
                recurrent, train_eps, test_eps, rtrain, rtest
            )
        write(path, out)
        print(
            "history_probe",
            json.dumps(
                {
                    m: {k: v for k, v in r.items() if k.endswith(("auc", "types"))}
                    for m, r in out.items()
                }
            ),
        )

    path = args.output / "world_model.json"
    if "world_model" in stages and not path.exists() and model_path.exists():
        recurrent, _ = load_world_model(model_path)
        out = {
            "canonical": world_model_metrics(model, compact, args.stage3_data),
            "recurrent": world_model_metrics(recurrent, compact, args.stage3_data),
        }
        write(path, out)
        print("world_model done")


if __name__ == "__main__":
    main()
