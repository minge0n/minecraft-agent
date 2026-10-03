"""Stage 2D imagination: open-loop rollouts of the Stage 2C world model.

See docs/stage2d.md. From every real start point, the world model reads the real
history up to that step and then rolls forward on its own with the real episode's
next actions (`open_loop` of the model). The imagined observations, rewards and
continuations are compared with what actually happened at horizons 1, 5, 10 and
20, during training and under a behavior policy different from the training data.
"""

import argparse
import json
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from minecraft_rl import resumable, runtime, world_model
from minecraft_rl.devices import DEVICES, select_device
from minecraft_rl.provenance import git_commit
from minecraft_rl.tmaze import Action, Observation
from minecraft_rl.world_model import DynamicsModel, Episodes

EVALUATION_DATA_SEED_OFFSET = 1_000_000
SHIFTED_DATA_SEED_OFFSET = 2_000_000
SHIFTED_TRAINING_DATA_SEED_OFFSET = 3_000_000
LATENT_SEED_OFFSET = 6_000_000


@dataclass(frozen=True)
class Config:
    model: world_model.Config = field(default_factory=world_model.Config)
    horizons: tuple[int, ...] = (1, 5, 10, 20)
    shifted_policy_weights: tuple[float, ...] = (0.8, 0.1, 0.1)
    rollout_steps: tuple[int, ...] = (0, 200, 1000, 2000, 3000, 5000)


def rollout_errors(
    model: DynamicsModel,
    episodes: Episodes,
    horizons: tuple[int, ...],
    generator: torch.Generator | None = None,
) -> dict[str, dict[str, float]]:
    """Imagined versus real outcome at each horizon k, pooled over every start s
    for which the real episode still runs at transition s + k - 1.

    - observation_accuracy: imagined o_{s+k} equals the real one.
    - trajectory_accuracy: all imagined o_{s+1} .. o_{s+k} equal the real ones.
    - reward_absolute_error: |imagined r_{s+k} - real r_{s+k}|.
    - return_absolute_error: |sum of imagined minus real rewards over s+1 .. s+k|.
    - continuation_accuracy: imagined continuation flag of step s+k is right.
    - junction_turn_*: the same reward errors restricted to steps s+k that are a
      real junction turn, the only nonzero rewards.
    - non_turn_reward_absolute_error: reward error on all other steps, whose real
      reward is 0.
    """
    longest = max(horizons)
    totals = {
        k: {
            "pairs": 0,
            "observation_correct": 0,
            "trajectory_correct": 0,
            "reward_error": 0.0,
            "return_error": 0.0,
            "continuation_correct": 0,
            "turn_pairs": 0,
            "turn_reward_error": 0.0,
            "turn_sign_correct": 0,
            "non_turn_reward_error": 0.0,
        }
        for k in horizons
    }
    turns = (episodes.observations == Observation.JUNCTION) & (
        episodes.actions != Action.FORWARD
    )
    with torch.no_grad():
        for start in range(episodes.actions.shape[1]):
            valid_start = episodes.mask[:, start]
            if not valid_start.any():
                break
            predictions = model.open_loop(episodes, start, longest, generator)
            length = predictions.reward.shape[1]
            window = slice(start, start + length)
            real_mask = episodes.mask[:, window]
            observation_correct = (
                predictions.observation_logits.argmax(-1)
                == episodes.next_observations[:, window]
            )
            trajectory_correct = observation_correct.cumprod(dim=1).bool()
            reward_error = predictions.reward - episodes.rewards[:, window]
            return_error = (reward_error * real_mask).cumsum(dim=1).abs()
            continuation_correct = (predictions.continue_logits > 0) == (
                episodes.continues[:, window] > 0.5
            )
            turn = turns[:, window]
            sign_correct = (predictions.reward > 0) == (episodes.rewards[:, window] > 0)
            for k in horizons:
                if k > length:
                    continue
                column = k - 1
                valid = real_mask[:, column]
                total = totals[k]
                total["pairs"] += int(valid.sum())
                total["observation_correct"] += int(
                    observation_correct[valid, column].sum()
                )
                total["trajectory_correct"] += int(
                    trajectory_correct[valid, column].sum()
                )
                total["reward_error"] += reward_error[valid, column].abs().sum().item()
                total["return_error"] += return_error[valid, column].sum().item()
                total["continuation_correct"] += int(
                    continuation_correct[valid, column].sum()
                )
                turn_valid = valid & turn[:, column]
                total["turn_pairs"] += int(turn_valid.sum())
                total["turn_reward_error"] += (
                    reward_error[turn_valid, column].abs().sum().item()
                )
                total["turn_sign_correct"] += int(
                    sign_correct[turn_valid, column].sum()
                )
                total["non_turn_reward_error"] += (
                    reward_error[valid & ~turn[:, column], column].abs().sum().item()
                )

    def ratio(numerator: float, denominator: int) -> float:
        return numerator / denominator if denominator else float("nan")

    return {
        str(k): {
            "pairs": t["pairs"],
            "observation_accuracy": ratio(t["observation_correct"], t["pairs"]),
            "trajectory_accuracy": ratio(t["trajectory_correct"], t["pairs"]),
            "reward_absolute_error": ratio(t["reward_error"], t["pairs"]),
            "return_absolute_error": ratio(t["return_error"], t["pairs"]),
            "continuation_accuracy": ratio(t["continuation_correct"], t["pairs"]),
            "junction_turn_pairs": t["turn_pairs"],
            "junction_turn_reward_absolute_error": ratio(
                t["turn_reward_error"], t["turn_pairs"]
            ),
            "junction_turn_sign_accuracy": ratio(
                t["turn_sign_correct"], t["turn_pairs"]
            ),
            "non_turn_reward_absolute_error": ratio(
                t["non_turn_reward_error"], t["pairs"] - t["turn_pairs"]
            ),
        }
        for k, t in totals.items()
    }


def train_and_measure(
    memory: bool,
    config: Config,
    training: Episodes,
    evaluations: dict[str, Episodes],
    device: torch.device,
    model_config: world_model.Config | None = None,
) -> tuple[DynamicsModel, torch.optim.Adam, dict[str, Any]]:
    """Train for the world-model step budget, measuring one-step and imagined error
    on every evaluation set at each rollout step. Latent samples are drawn from a
    generator reseeded identically at every rollout step."""
    model_config = model_config or config.model
    model, optimizer = world_model.build(model_config, device, memory)
    generator = torch.Generator().manual_seed(model_config.seed)
    evaluations = {name: data.to(device) for name, data in evaluations.items()}
    trained = 0
    curve: dict[str, Any] = {}
    for step in sorted({*config.rollout_steps, model_config.steps}):
        if step > model_config.steps:
            break
        world_model.train(
            model,
            optimizer,
            training,
            model_config,
            device,
            generator,
            step - trained,
        )
        trained = step
        latent_seed = model_config.seed + LATENT_SEED_OFFSET
        curve[str(step)] = {
            name: {
                "one_step": world_model.evaluate(
                    model, data, device, torch.Generator().manual_seed(latent_seed)
                ),
                "imagined": rollout_errors(
                    model,
                    data,
                    config.horizons,
                    torch.Generator().manual_seed(latent_seed),
                ),
                "latent": model.diagnostics(
                    data, torch.Generator().manual_seed(latent_seed)
                ),
            }
            for name, data in evaluations.items()
        }
    return model, optimizer, curve


def run(
    config: Config,
    device: torch.device,
    output: Path,
    session: resumable.Session | None = None,
) -> dict[str, Any]:
    """Three world models on the same budget: the Stage 2C GRU and its no-memory
    control trained on uniform-policy data, and a GRU trained on a half-uniform,
    half-shifted-policy mixture of the same size (data coverage). With a
    `session`, the run can stop after any trained model and continue in a later
    process with identical results. Each model starts from its own seeded
    generators, so the finished models are the only state."""
    session = session or resumable.Session(None, None, {})
    saved = session.load()
    model_config = config.model
    seeds = {
        "torch_initialization": model_config.seed,
        "training_data": model_config.seed,
        "shifted_training_data": model_config.seed + SHIFTED_TRAINING_DATA_SEED_OFFSET,
        "minibatch_sampling": model_config.seed,
        "evaluation_data": model_config.seed + EVALUATION_DATA_SEED_OFFSET,
        "shifted_policy_data": model_config.seed + SHIFTED_DATA_SEED_OFFSET,
    }
    uniform_training = world_model.collect_episodes(
        model_config,
        model_config.training_episodes,
        torch.Generator().manual_seed(seeds["training_data"]),
    )
    half = model_config.training_episodes // 2
    shifted_training = world_model.collect_episodes(
        model_config,
        model_config.training_episodes - half,
        torch.Generator().manual_seed(seeds["shifted_training_data"]),
        config.shifted_policy_weights,
    )
    mixed_training = Episodes(
        *(
            torch.cat(
                [getattr(uniform_training, f)[:half], getattr(shifted_training, f)]
            )
            for f in Episodes.__dataclass_fields__
        )
    )
    evaluations = {
        "uniform_policy": world_model.collect_episodes(
            model_config,
            model_config.evaluation_episodes,
            torch.Generator().manual_seed(seeds["evaluation_data"]),
        ),
        "shifted_policy": world_model.collect_episodes(
            model_config,
            model_config.evaluation_episodes,
            torch.Generator().manual_seed(seeds["shifted_policy_data"]),
            config.shifted_policy_weights,
        ),
    }
    variants = {
        "gru_uniform_data": (True, uniform_training, "gru"),
        "no_memory_uniform_data": (False, uniform_training, "gru"),
        "gru_mixed_data": (True, mixed_training, "gru"),
    }
    if model_config.kind == "rssm":
        variants |= {
            "rssm_uniform_data": (True, uniform_training, "rssm"),
            "rssm_mixed_data": (True, mixed_training, "rssm"),
        }
    models: dict[str, Any] = {} if saved is None else saved["models"]
    if saved is not None:
        session.restore_global_random_state()
    for index, (name, (memory, training, kind)) in enumerate(variants.items()):
        if name in models:
            continue
        variant_config = replace(model_config, kind=kind)
        model, optimizer, curve = train_and_measure(
            memory, config, training, evaluations, device, variant_config
        )
        checkpoint = output.parent / f"{name}.pt"
        world_model.save_checkpoint(
            checkpoint, variant_config, model_config.steps, model, optimizer
        )
        models[name] = {
            "memory": memory,
            "training_data": world_model.dataset_summary(training),
            "parameters": world_model.parameter_count(model),
            "checkpoint": str(checkpoint),
            "curve": curve,
            "final": curve[str(model_config.steps)],
        }
        if index + 1 < len(variants):
            session.boundary(lambda: {"models": models})

    result: dict[str, Any] = {
        "stage": "2d-imagination-tmaze",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "config": asdict(config),
        "seeds": seeds,
        "device": str(device),
        "runtime": runtime.metadata(device),
        "environment": {
            "name": "TMaze",
            "observations": {o.name: int(o) for o in Observation},
            "actions": {a.name: int(a) for a in Action},
            "corridor_length": model_config.corridor_length,
            "max_steps": model_config.max_steps,
        },
        "evaluation_data": {
            name: world_model.dataset_summary(data)
            for name, data in evaluations.items()
        },
        "imagination": "open loop from every real start s: real history up to s, "
        "then the most likely predicted observation is fed back with the real "
        "next action",
        "models": models,
        "checkpoint_format": world_model.CHECKPOINT_FORMAT,
        "duration_seconds": session.elapsed_seconds,
        "sessions": session.sessions,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    session.finish()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=world_model.Config.seed)
    parser.add_argument("--steps", type=int, default=world_model.Config.steps)
    world_model.add_world_model_arguments(parser)
    resumable.add_arguments(parser)
    parser.add_argument("--device", default="cpu", choices=DEVICES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runtime.configure(args.seed)
    if args.steps < 1:
        parser.error("--steps must be positive")
    config = Config(
        model=world_model.Config(
            seed=args.seed, steps=args.steps, **world_model.world_model_options(args)
        )
    )
    output = (
        args.output
        or Path("runs")
        / datetime.now(UTC).strftime("stage2d-%Y%m%dT%H%M%S%fZ")
        / "metrics.json"
    )
    session = resumable.session_from(
        args, {"experiment": "imagination", "config": asdict(config)}
    )
    try:
        result = run(config, select_device(args.device), output, session)
    except resumable.Incomplete as stopped:
        print(stopped)
        raise SystemExit(resumable.INCOMPLETE_EXIT_CODE) from None

    runtime_info = result["runtime"]
    print(
        f"Device {result['device']}, torch {runtime_info['versions']['torch']}, "
        f"threads {runtime_info['intra_op_threads']}+{runtime_info['inter_op_threads']}"
    )
    for name, data in result["evaluation_data"].items():
        print(
            f"Evaluation {name}: {data['episodes']} episodes, mean length "
            f"{data['mean_length']:.1f}, junction steps mean "
            f"{data['mean_junction_steps']:.1f} max {data['max_junction_steps']}"
        )
    for name, model in result["models"].items():
        data = model["training_data"]
        print(
            f"Model {name}: {model['parameters']} parameters, trained on "
            f"{data['episodes']} episodes, junction steps max "
            f"{data['max_junction_steps']}"
        )
    print(
        "Imagined vs real after training, per horizon k: observation accuracy, "
        "return abs error, non-turn reward abs error, turn reward sign accuracy"
    )
    for policy in result["evaluation_data"]:
        print(f"{policy}:")
        for name, model in result["models"].items():
            cells = [
                f"k={k}: {e['observation_accuracy']:.2f} "
                f"{e['return_absolute_error']:.3f} "
                f"{e['non_turn_reward_absolute_error']:.3f} "
                f"{e['junction_turn_sign_accuracy']:.2f}"
                for k, e in model["final"][policy]["imagined"].items()
            ]
            print(f"  {name:<23}" + " | ".join(cells))
    longest = str(max(config.horizons))
    print(f"Return abs error at k={longest} by training step (uniform | shifted):")
    for name, model in result["models"].items():
        cells = []
        for step, at in model["curve"].items():
            uniform = at["uniform_policy"]["imagined"][longest]
            shifted = at["shifted_policy"]["imagined"][longest]
            cells.append(
                f"{step}: {uniform['return_absolute_error']:.2f}"
                f"|{shifted['return_absolute_error']:.2f}"
            )
        print(f"  {name:<23}" + "  ".join(cells))
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
