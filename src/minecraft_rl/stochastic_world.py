"""Stage 2H: does a stochastic latent model represent uncertain futures?

See docs/stage2h.md. In the signal T-maze (`signal_tmaze.py`), the signal is
drawn when the agent first steps forward: CUE_LEFT with probability p. A turn
at the junction pays +1 toward the signalled side and -1 otherwise. The same
observable start can therefore lead to two different futures.

The deterministic GRU world model and the RSSM train on the same random-policy
data. Both are then asked, from the start state with the actions
FORWARD, then FORWARD to the junction, then LEFT:

1. One step ahead: what is P(next observation = CUE_LEFT)? The true value is p.
2. Open loop to the turn: what is the distribution of the turn reward? The
   true reward is +1 with probability p and -1 with probability 1 - p.

The GRU imagines one trajectory, so it can only commit to its most likely
signal. The RSSM samples the signal from its prior, and each sampled future
must stay consistent with its own signal.
"""

import argparse
import json
import math
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from minecraft_rl import runtime, world_model
from minecraft_rl.devices import DEVICES, select_device
from minecraft_rl.provenance import git_commit
from minecraft_rl.rssm import RSSM
from minecraft_rl.tmaze import Action, Observation
from minecraft_rl.world_model import DynamicsModel, Episodes

EVALUATION_DATA_SEED_OFFSET = 1_000_000
LATENT_SEED_OFFSET = 6_000_000


def probe_episodes(config: world_model.Config, count: int) -> Episodes:
    """`count` identical action sequences from the start state: FORWARD until
    the junction, then LEFT. Observations after the start are placeholders. A
    world model reads only o_0 = start, so its predictions never use them."""
    forward_steps = config.corridor_length + 2
    length = forward_steps + 1
    shape = (count, config.max_steps)
    observations = torch.full(shape, Observation.CORRIDOR, dtype=torch.long)
    actions = torch.zeros(shape, dtype=torch.long)
    actions[:, forward_steps] = Action.LEFT
    mask = torch.zeros(shape, dtype=torch.bool)
    mask[:, :length] = True
    return Episodes(
        observations,
        actions,
        torch.full(shape, Observation.CORRIDOR, dtype=torch.long),
        torch.zeros(shape),
        torch.ones(shape),
        mask,
        torch.zeros(count, dtype=torch.long),
    )


def signal_distribution(
    model: DynamicsModel,
    config: world_model.Config,
    rollouts: int,
    generator: torch.Generator,
) -> dict[str, Any]:
    """The predicted distribution of the signal and of the later turn reward.

    One open-loop rollout per probe episode: each imagines its own signal at
    the first step, then the turn reward at the end. For a model that keeps
    each imagined future consistent, the reward is near +1 after an imagined
    left signal and near -1 after a right one.

    `given_observed_signal` repeats the rollout after the model reads the real
    signal observation: the reward then depends on a signal that the model saw,
    not on one that it sampled. The true turn-left reward is +1 after a left
    signal and -1 after a right signal."""
    probe = probe_episodes(config, rollouts)
    turn_index = config.corridor_length + 2
    with torch.no_grad():
        one_step = model.one_step(probe, generator)
        imagined = model.open_loop(probe, 0, turn_index + 1, generator)
    signal_probability = torch.softmax(one_step.observation_logits[:, 0], -1)
    imagined_signal = imagined.observation_logits[:, 0].argmax(-1)
    turn_reward = imagined.reward[:, turn_index]
    left_signal = imagined_signal == Observation.CUE_LEFT
    right_signal = imagined_signal == Observation.CUE_RIGHT
    expected_sign = torch.where(left_signal, 1.0, -1.0)
    consistent = (torch.sign(turn_reward) == expected_sign) & (
        left_signal | right_signal
    )
    p = config.signal_left_probability
    probability_left = signal_probability[:, Observation.CUE_LEFT].mean().item()
    result: dict[str, Any] = {
        "rollouts": rollouts,
        "one_step_probability_cue_left": probability_left,
        "one_step_probability_cue_right": signal_probability[:, Observation.CUE_RIGHT]
        .mean()
        .item(),
        "one_step_probability_error": abs(probability_left - p),
        "imagined_fraction_cue_left": left_signal.float().mean().item(),
        "imagined_fraction_cue_right": right_signal.float().mean().item(),
        "turn_reward_mean": turn_reward.mean().item(),
        "turn_reward_mean_error": turn_reward.mean().item() - (2 * p - 1),
        "turn_reward_fraction_positive": (turn_reward > 0).float().mean().item(),
        "turn_reward_mean_given_cue_left": masked_mean(turn_reward, left_signal),
        "turn_reward_mean_given_cue_right": masked_mean(turn_reward, right_signal),
        "turn_reward_positive_given_cue_left": masked_mean(
            (turn_reward > 0).float(), left_signal
        ),
        "turn_reward_positive_given_cue_right": masked_mean(
            (turn_reward > 0).float(), right_signal
        ),
        "signal_consistent_fraction": consistent.float().mean().item(),
        "given_observed_signal": observed_signal_rollouts(
            model, config, rollouts, generator
        ),
    }
    if isinstance(model, RSSM):
        result["prior_signal_codes"] = prior_signal_codes(
            model, config, rollouts, generator
        )
    return result


def observed_signal_probe(
    config: world_model.Config, count: int, signal: Observation
) -> Episodes:
    probe = probe_episodes(config, count)
    probe.observations[:, 1] = signal
    return probe


def observed_signal_rollouts(
    model: DynamicsModel,
    config: world_model.Config,
    rollouts: int,
    generator: torch.Generator,
) -> dict[str, float]:
    """Mean imagined turn-left reward and its share of positive values after
    the model reads the real signal o_1, then rolls forward open loop."""
    turn_offset = config.corridor_length + 1
    result = {}
    for signal in (Observation.CUE_LEFT, Observation.CUE_RIGHT):
        probe = observed_signal_probe(config, rollouts, signal)
        with torch.no_grad():
            imagined = model.open_loop(probe, 1, turn_offset + 1, generator)
        reward = imagined.reward[:, turn_offset]
        name = signal.name.lower()
        result[f"turn_reward_mean_{name}"] = reward.mean().item()
        result[f"turn_reward_positive_{name}"] = (reward > 0).float().mean().item()
    return result


def prior_signal_codes(
    model: RSSM,
    config: world_model.Config,
    rollouts: int,
    generator: torch.Generator,
) -> dict[str, Any]:
    """Do prior samples at the signal step look like a real signal code?

    The posterior code of a signal is the most likely class of each latent
    variable after the model reads that signal. A variable is cue-coding when
    the two codes differ there. The prior samples each variable independently.
    If the cue is spread over several variables, a prior sample can mix the
    left code in one variable with the right code in another: a code that no
    real signal produces. This measures how often that happens."""
    codes = {}
    with torch.no_grad():
        for signal in (Observation.CUE_LEFT, Observation.CUE_RIGHT):
            probe = observed_signal_probe(config, 1, signal)
            filtered = model.filter(
                probe.observations[:, :2], probe.actions[:, :1], generator
            )
            codes[signal] = filtered["posterior"][0, 1].argmax(-1)
        probe = probe_episodes(config, rollouts)
        filtered = model.filter(
            probe.observations[:, :1], probe.actions[:, :0], generator
        )
        h = model.transition(
            filtered["h"][:, 0], filtered["z"][:, 0], probe.actions[:, 0]
        )
        prior = model.prior_logits(h)
        samples = model.latent(prior, generator).unflatten(
            -1, (model.variables, model.classes)
        )
    sampled = samples.argmax(-1)
    left, right = codes[Observation.CUE_LEFT], codes[Observation.CUE_RIGHT]
    coding = left != right
    if not coding.any():
        return {"cue_coding_variables": 0}
    matches_left = (sampled[:, coding] == left[coding]).all(-1)
    matches_right = (sampled[:, coding] == right[coding]).all(-1)
    return {
        "cue_coding_variables": int(coding.sum()),
        "fraction_left_code": matches_left.float().mean().item(),
        "fraction_right_code": matches_right.float().mean().item(),
        "fraction_mixed_code": (~matches_left & ~matches_right).float().mean().item(),
    }


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    return values[mask].mean().item() if mask.any() else float("nan")


def signal_negative_log_likelihood(
    model: DynamicsModel,
    episodes: Episodes,
    generator: torch.Generator,
) -> dict[str, float]:
    """Mean -log P(real o_1) over held-out episodes whose first action was
    FORWARD, the step where the signal is drawn. A model that always predicts
    the majority signal with certainty gets an infinite NLL on the minority
    episodes, so the probability is clipped at 1e-6."""
    with torch.no_grad():
        predictions = model.one_step(episodes, generator)
    reached = episodes.mask[:, 0] & (episodes.actions[:, 0] == Action.FORWARD)
    probabilities = torch.softmax(predictions.observation_logits[reached, 0], -1)
    real = episodes.next_observations[reached, 0]
    chosen = probabilities.gather(-1, real.unsqueeze(-1)).squeeze(-1)
    return {
        "episodes": int(reached.sum()),
        "negative_log_likelihood": -chosen.clamp(min=1e-6).log().mean().item(),
        "empirical_fraction_cue_left": (real == Observation.CUE_LEFT)
        .float()
        .mean()
        .item(),
    }


def entropy_bound(p: float) -> float:
    """The lowest achievable NLL: the entropy of the true signal distribution."""
    return -sum(q * math.log(q) for q in (p, 1 - p) if q > 0)


def measure(
    model: DynamicsModel,
    config: world_model.Config,
    evaluation: Episodes,
    rollouts: int,
    device: torch.device,
) -> dict[str, Any]:
    latent_seed = config.seed + LATENT_SEED_OFFSET
    return {
        "distribution": signal_distribution(
            model, config, rollouts, torch.Generator().manual_seed(latent_seed)
        ),
        "held_out_signal": signal_negative_log_likelihood(
            model, evaluation.to(device), torch.Generator().manual_seed(latent_seed)
        ),
        "one_step": world_model.evaluate(
            model, evaluation, device, torch.Generator().manual_seed(latent_seed)
        ),
        "latent": model.diagnostics(
            evaluation.to(device), torch.Generator().manual_seed(latent_seed)
        ),
    }


def run(
    config: world_model.Config,
    rollouts: int,
    device: torch.device,
    output: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    training = world_model.collect_episodes(
        config, config.training_episodes, torch.Generator().manual_seed(config.seed)
    )
    evaluation = world_model.collect_episodes(
        config,
        config.evaluation_episodes,
        torch.Generator().manual_seed(config.seed + EVALUATION_DATA_SEED_OFFSET),
    )
    models: dict[str, Any] = {}
    for kind in ("gru", "rssm"):
        model_config = replace(config, kind=kind)
        model, optimizer = world_model.build(model_config, device)
        generator = torch.Generator().manual_seed(config.seed)
        world_model.train(
            model, optimizer, training, model_config, device, generator, config.steps
        )
        models[kind] = {
            "parameters": world_model.parameter_count(model),
            "result": measure(model, model_config, evaluation, rollouts, device),
        }
    p = config.signal_left_probability
    result: dict[str, Any] = {
        "stage": "2h-signal-tmaze-distribution",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "config": asdict(config),
        "rollouts": rollouts,
        "seeds": {
            "torch_initialization": config.seed,
            "training_data": config.seed,
            "evaluation_data": config.seed + EVALUATION_DATA_SEED_OFFSET,
            "latent_sampling": config.seed + LATENT_SEED_OFFSET,
        },
        "device": str(device),
        "runtime": runtime.metadata(device),
        "truth": {
            "probability_cue_left": p,
            "turn_left_reward_mean": 2 * p - 1,
            "turn_left_reward_fraction_positive": p,
            "minimum_signal_negative_log_likelihood": entropy_bound(p),
        },
        "training_data": world_model.dataset_summary(training),
        "evaluation_data": world_model.dataset_summary(evaluation),
        "models": models,
        "duration_seconds": time.monotonic() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=world_model.Config.seed)
    parser.add_argument("--steps", type=int, default=world_model.Config.steps)
    parser.add_argument(
        "--signal-left-probability",
        type=float,
        default=world_model.Config.signal_left_probability,
    )
    parser.add_argument("--rollouts", type=int, default=2000)
    parser.add_argument("--free-nats", type=float, default=world_model.Config.free_nats)
    parser.add_argument("--device", default="cpu", choices=DEVICES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runtime.configure(args.seed)
    config = world_model.Config(
        seed=args.seed,
        steps=args.steps,
        environment="signal",
        signal_left_probability=args.signal_left_probability,
        free_nats=args.free_nats,
    )
    output = (
        args.output
        or Path("runs")
        / datetime.now(UTC).strftime("stage2h-%Y%m%dT%H%M%S%fZ")
        / "metrics.json"
    )
    result = run(config, args.rollouts, select_device(args.device), output)
    truth = result["truth"]
    print(
        f"True P(signal left) {truth['probability_cue_left']:.3f}, true mean turn-left "
        f"reward {truth['turn_left_reward_mean']:+.3f}, minimum signal NLL "
        f"{truth['minimum_signal_negative_log_likelihood']:.3f}"
    )
    for kind, model in result["models"].items():
        distribution = model["result"]["distribution"]
        signal = model["result"]["held_out_signal"]
        one_step_left = distribution["one_step_probability_cue_left"]
        print(
            f"{kind}: one-step P(left) {one_step_left:.3f}"
            f", imagined left {distribution['imagined_fraction_cue_left']:.3f}, turn "
            f"reward mean {distribution['turn_reward_mean']:+.3f}, P(+1) "
            f"{distribution['turn_reward_fraction_positive']:.3f}, given left "
            f"{distribution['turn_reward_mean_given_cue_left']:+.3f}, given right "
            f"{distribution['turn_reward_mean_given_cue_right']:+.3f}, held-out NLL "
            f"{signal['negative_log_likelihood']:.3f}"
        )
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
