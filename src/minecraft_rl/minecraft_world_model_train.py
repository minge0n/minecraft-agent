"""Train the Stage 3 Minecraft RSSM on collected episodes.

The run trains on the `train` split and evaluates at fixed update counts
(`--eval-at`) on fixed windows of the `eval_episode` and `eval_seed`
splits, and on a fixed subset of training windows. Evaluation never updates
parameters. With `--tiny-windows N`, the run trains and evaluates on N fixed
training windows only: the tiny-overfit test.

The run uses the canonical runtime and is resumable (`--state`,
`--stop-after`): at every evaluation boundary and every `--boundary-every`
updates it can save the model, optimizer and generator states and exit with
code 75. The same command continues with the same results. Run it through
`minecraft_rl.sweep` for the duty cycle and the thermostat on a Mac.

Example:

    .venv/bin/python -m minecraft_rl.sweep minecraft_world_model_train --seeds 0 \\
        --output-root runs/stage3/train --duty-cycle 1.0 -- --updates 1000 \\
        --eval-at 0,100,300,1000
"""

import argparse
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from minecraft_rl import macos_energy, resumable, runtime
from minecraft_rl import minecraft_world_model_eval as evaluation
from minecraft_rl.minecraft_dataset import load_episodes
from minecraft_rl.minecraft_interface import OBSERVATION_SCHEMA
from minecraft_rl.minecraft_replay import (
    CompactVocabulary,
    SequenceReplay,
    load_schema,
    raw_sizes,
    unknown_fractions,
)
from minecraft_rl.minecraft_world_model import (
    LOSS_TERMS,
    MinecraftRSSM,
    ModelConfig,
    parameter_count,
    save_checkpoint,
)
from minecraft_rl.provenance import git_commit

SPLITS = ("train", "eval_episode", "eval_seed")


def parse_counts(text: str) -> list[int]:
    return sorted({int(part) for part in text.split(",") if part})


def evaluation_windows(replay: SequenceReplay, limit: int | None) -> list:
    index = replay.evaluation_windows(replay.length)
    if limit is not None and len(index) > limit:
        step = len(index) / limit
        index = [index[int(i * step)] for i in range(limit)]
    return index


def evaluate(
    model: MinecraftRSSM,
    replays: dict[str, SequenceReplay],
    windows: dict[str, dict[str, list]],
    unknown: torch.Tensor,
    frequency_rays: torch.Tensor,
    frequency_slot: int,
    args,
    imagine: bool,
) -> dict:
    out = {}
    for split, replay in replays.items():
        started = time.perf_counter()
        generator = torch.Generator().manual_seed(1234)
        result = {
            "one_step": evaluation.one_step(
                model,
                replay["one_step"],
                windows[split]["one_step"],
                args.context,
                unknown,
                frequency_rays,
                frequency_slot,
                generator,
                events=split != "train",
            )
        }
        if imagine:
            result["imagination"] = evaluation.imagination(
                model,
                replay["imagination"],
                windows[split]["imagination"],
                args.context,
                unknown,
                generator,
            )
        result["seconds"] = time.perf_counter() - started
        out[split] = result
    model.train()
    return out


def summary_line(update: int, losses: dict, results: dict) -> str:
    parts = [f"update {update}"]
    for split, result in results.items():
        model = result["one_step"]["all"]["model"]
        persistence = result["one_step"]["all"]["persistence"]
        parts.append(
            f"{split}: ray changed acc {model['ray_accuracy_changed']['mean']:.3f} "
            f"(persist {persistence['ray_accuracy_changed']['mean']:.3f}), "
            f"ray acc {model['ray_accuracy']['mean']:.3f} "
            f"(persist {persistence['ray_accuracy']['mean']:.3f})"
        )
    if losses:
        parts.append(f"train loss {losses['total']:.2f}")
    return "; ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", type=Path, default=Path("runs/stage3/data"))
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--eval-at", default="0,100,300,1000")
    parser.add_argument("--imagine-at", help="updates with imagination (default: last)")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--context", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--clip", type=float, default=1000.0)
    parser.add_argument("--eval-windows", type=int, default=128)
    parser.add_argument("--tiny-windows", type=int)
    parser.add_argument("--boundary-every", type=int, default=50)
    parser.add_argument("--ray-layer", default=ModelConfig.ray_layer)
    parser.add_argument(
        "--encoder-channels", type=int, default=ModelConfig.encoder_channels
    )
    parser.add_argument(
        "--decoder-channels", type=int, default=ModelConfig.decoder_channels
    )
    parser.add_argument("--output", type=Path)
    resumable.add_arguments(parser)
    args = parser.parse_args()
    runtime.configure(args.seed)
    eval_at = [u for u in parse_counts(args.eval_at) if u <= args.updates]
    if args.updates not in eval_at:
        eval_at.append(args.updates)
    imagine_at = (
        parse_counts(args.imagine_at) if args.imagine_at is not None else [args.updates]
    )
    output = (
        args.output
        or Path("runs")
        / datetime.now(UTC).strftime("stage3-%Y%m%dT%H%M%S%fZ")
        / "metrics.json"
    )
    config = ModelConfig(
        ray_layer=args.ray_layer,
        encoder_channels=args.encoder_channels,
        decoder_channels=args.decoder_channels,
    )

    schema = load_schema(args.dataset)
    episodes = {split: load_episodes(args.dataset / split) for split in SPLITS}
    compact = CompactVocabulary.from_episodes(episodes["train"], raw_sizes(schema))
    vocabulary = compact.model_vocabulary(
        schema["rows"], schema["columns"], schema["max_distance"]
    )
    imagine_length = args.context + args.horizon
    replays = {
        split: {
            "one_step": SequenceReplay(eps, compact, args.length),
            "imagination": SequenceReplay(eps, compact, imagine_length),
        }
        for split, eps in episodes.items()
    }
    train_replay = replays["train"]["one_step"]
    windows = {
        split: {
            "one_step": evaluation_windows(replay["one_step"], args.eval_windows),
            "imagination": evaluation_windows(replay["imagination"], args.eval_windows),
        }
        for split, replay in replays.items()
    }
    tiny = None
    if args.tiny_windows is not None:
        tiny = windows["train"]["one_step"][: args.tiny_windows]
        windows = {"train": {"one_step": tiny, "imagination": []}}
        replays = {"train": replays["train"]}
        imagine_at = []

    torch.manual_seed(args.seed)
    model = MinecraftRSSM(vocabulary, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    generator = torch.Generator().manual_seed(args.seed)
    unknown = evaluation.unknown_classes(compact, model)
    frequency_rays = evaluation.ray_frequency_baseline(train_replay, model, unknown)
    frequency_slot = evaluation.slot_frequency_baseline(train_replay)

    identity = {
        "experiment": "minecraft_world_model",
        "seed": args.seed,
        "config": config.to_json(),
        "arguments": {
            k: str(v) for k, v in vars(args).items() if k not in ("state", "stop_after")
        },
        "vocabulary": compact.identifier(),
    }
    session = resumable.session_from(args, identity)
    saved = session.load()
    update = 0
    history: list[dict] = []
    evaluations: dict[str, dict] = {}
    train_seconds = 0.0
    energy_before = 0.0
    session_counters = macos_energy.read()
    if saved is not None:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        generator = resumable.restore_generator(saved["generator"])
        update = saved["update"]
        history = json.loads(saved["history"])
        evaluations = json.loads(saved["evaluations"])
        train_seconds = saved["train_seconds"]
        energy_before = saved["energy_joules"]
        session.restore_global_random_state()

    def energy_joules() -> float | None:
        """CPU energy of this run over all of its processes."""
        now = macos_energy.read()
        if now is None or session_counters is None:
            return None
        return energy_before + (now - session_counters).energy_joules

    def state() -> dict:
        return {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "generator": resumable.generator_state(generator),
            "update": update,
            "history": json.dumps(history),
            "evaluations": json.dumps(evaluations),
            "train_seconds": train_seconds,
            "energy_joules": energy_joules() or 0.0,
        }

    try:
        while True:
            if update in eval_at and str(update) not in evaluations:
                results = evaluate(
                    model,
                    replays,
                    windows,
                    unknown,
                    frequency_rays,
                    frequency_slot,
                    args,
                    imagine=update in imagine_at,
                )
                evaluations[str(update)] = results
                print(
                    summary_line(update, history[-1] if history else {}, results),
                    flush=True,
                )
                session.boundary(state)
            if update >= args.updates:
                break
            started = time.perf_counter()
            if tiny is None:
                batch = train_replay.sample(args.batch, generator)
            else:
                picks = torch.randint(len(tiny), (args.batch,), generator=generator)
                batch = train_replay.batch([tiny[i] for i in picks.tolist()])
            model.train()
            terms = model.losses(
                batch.observations, batch.actions, batch.continues, generator
            )
            optimizer.zero_grad()
            terms["total"].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            train_seconds += time.perf_counter() - started
            update += 1
            if update % 10 == 0 or update == 1:
                record = {name: float(value.detach()) for name, value in terms.items()}
                record["update"] = update
                record["grad_norm"] = float(norm)
                history.append(record)
            if update % args.boundary_every == 0:
                session.boundary(state)
    except resumable.Incomplete as stopped:
        print(stopped, flush=True)
        raise SystemExit(resumable.INCOMPLETE_EXIT_CODE) from None

    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = output.parent / "world_model.pt"
    save_checkpoint(
        checkpoint, model, {"vocabulary": compact.to_json(), "updates": update}
    )
    result = {
        "experiment": "minecraft_world_model",
        "git_commit": git_commit(),
        "runtime": runtime.metadata(torch.device("cpu")),
        "seed": args.seed,
        "dataset": {
            "path": str(args.dataset),
            "observation_schema": OBSERVATION_SCHEMA,
            "schema": schema,
            "splits": {
                split: {
                    "episodes": len(eps),
                    "transitions": sum(e.steps for e in eps),
                    "world_seeds": sorted({e.meta.world_seed for e in eps}),
                    "policy_seeds": sorted({e.meta.policy_seed for e in eps}),
                    "unknown_fraction": unknown_fractions(eps, compact),
                }
                for split, eps in episodes.items()
            },
        },
        "vocabulary": {
            "identifier": compact.identifier(),
            "sizes": {family: compact.size(family) for family in compact.ids},
            "ids": compact.ids,
        },
        "model": {
            "config": config.to_json(),
            "parameters": parameter_count(model),
            "parameters_encoder": parameter_count(model.encoder),
            "parameters_decoder": parameter_count(model.decoder),
        },
        "training": {
            "updates": update,
            "batch": args.batch,
            "length": args.length,
            "optimizer": {
                "name": "Adam",
                "learning_rate": args.learning_rate,
                "eps": 1e-5,
            },
            "gradient_clip": args.clip,
            "tiny_windows": args.tiny_windows,
            "tiny_window_index": tiny,
            "loss_terms": list(LOSS_TERMS),
            "train_seconds": train_seconds,
            "seconds_per_update": train_seconds / update if update else None,
            "sequence_steps_per_second": (
                update * args.batch * (args.length + 1) / train_seconds
                if train_seconds
                else None
            ),
        },
        "evaluation": {
            "context": args.context,
            "horizon": args.horizon,
            "windows": {
                split: {kind: len(index) for kind, index in w.items()}
                for split, w in windows.items()
            },
            "at_updates": evaluations,
        },
        "history": history,
        "energy": {
            "cpu_energy_joules": energy_joules(),
            "joules_per_update": (energy_joules() or 0.0) / update if update else None,
            "source": "proc_pid_rusage RUSAGE_INFO_V6 ri_energy_nj (macOS)",
        },
        "checkpoint": str(checkpoint),
        "duration_seconds": session.elapsed_seconds,
        "sessions": session.sessions,
        "arguments": {k: str(v) for k, v in vars(args).items()},
        "config_dataclass": asdict(config),
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    session.finish()
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
