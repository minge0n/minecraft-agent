"""Stage 2C learned dynamics: a recurrent world model of the T-maze.

See docs/stage2c.md. Episodes are collected with a uniformly random policy. A GRU
reads the history of observations and actions and predicts, for every step, the
next observation, the reward and whether the episode continues. Predicting the
junction-turn reward needs the cue seen at the first step, and predicting when the
junction appears needs a count of forward moves, so a no-memory control with the
same prediction heads must fail exactly there.
"""

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

from minecraft_rl.devices import DEVICES, select_device
from minecraft_rl.provenance import git_commit
from minecraft_rl.tmaze import Action, Cue, Observation, TMaze

OBSERVATIONS = len(Observation)
ACTIONS = len(Action)
CHECKPOINT_FORMAT = "tmaze-world-model-v1"
EVALUATION_DATA_SEED_OFFSET = 1_000_000


@dataclass(frozen=True)
class Config:
    seed: int = 0
    corridor_length: int = 4
    max_steps: int = 40
    training_episodes: int = 512
    evaluation_episodes: int = 256
    hidden: int = 32
    learning_rate: float = 0.003
    steps: int = 5000
    batch: int = 32
    report_steps: tuple[int, ...] = (0, 100, 200, 500, 1000, 2000, 3000, 4000, 5000)


@dataclass(frozen=True)
class Episodes:
    """Padded transitions (episodes, max_steps); step t is (o_t, a_t) -> o_{t+1},
    r_{t+1}, c_{t+1}. `cues` is ground truth for evaluation breakdowns only and is
    never a model input."""

    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    rewards: torch.Tensor
    continues: torch.Tensor
    mask: torch.Tensor
    cues: torch.Tensor

    def select(self, index: torch.Tensor) -> "Episodes":
        return Episodes(*(getattr(self, f)[index] for f in self.__dataclass_fields__))

    def to(self, device: torch.device) -> "Episodes":
        return Episodes(
            *(getattr(self, f).to(device) for f in self.__dataclass_fields__)
        )


def collect_episodes(
    config: Config,
    count: int,
    generator: torch.Generator,
    action_weights: tuple[float, ...] | None = None,
) -> Episodes:
    """Run `count` episodes with random cues and a random behavior policy: uniform,
    or drawing actions with the given relative weights."""
    environment = TMaze(config.corridor_length, config.max_steps)
    weights = None if action_weights is None else torch.tensor(action_weights)
    shape = (count, config.max_steps)
    observations = torch.zeros(shape, dtype=torch.long)
    actions = torch.zeros(shape, dtype=torch.long)
    next_observations = torch.zeros(shape, dtype=torch.long)
    rewards = torch.zeros(shape)
    continues = torch.zeros(shape)
    mask = torch.zeros(shape, dtype=torch.bool)
    cues = torch.randint(0, len(Cue), (count,), generator=generator)
    for episode in range(count):
        observation = environment.reset(Cue(int(cues[episode])))
        for t in range(config.max_steps):
            if weights is None:
                index = torch.randint(0, ACTIONS, (1,), generator=generator)
            else:
                index = torch.multinomial(weights, 1, generator=generator)
            action = Action(int(index))
            next_observation, reward, terminated, truncated = environment.step(action)
            observations[episode, t] = observation
            actions[episode, t] = action
            next_observations[episode, t] = next_observation
            rewards[episode, t] = reward
            continues[episode, t] = 0.0 if terminated else 1.0
            mask[episode, t] = True
            if terminated or truncated:
                break
            observation = next_observation
    return Episodes(
        observations, actions, next_observations, rewards, continues, mask, cues
    )


@dataclass(frozen=True)
class Predictions:
    observation_logits: torch.Tensor
    reward: torch.Tensor
    continue_logits: torch.Tensor


class WorldModel(nn.Module):
    """x_t = [onehot(o_t), onehot(a_t)] (batch, time, 8) -> state h_t (batch, time, H)
    -> next-observation logits (.., 5), reward (..), continuation logit (..).

    With `memory`, h_t comes from a GRU over the whole history; without it, from a
    tanh layer over x_t alone (the no-memory control). The heads are identical.
    """

    def __init__(self, hidden: int, memory: bool) -> None:
        super().__init__()
        self.memory = memory
        inputs = OBSERVATIONS + ACTIONS
        self.core = (
            nn.GRU(inputs, hidden, batch_first=True)
            if memory
            else nn.Linear(inputs, hidden)
        )
        self.observation_head = nn.Linear(hidden, OBSERVATIONS)
        self.reward_head = nn.Linear(hidden, 1)
        self.continue_head = nn.Linear(hidden, 1)

    def recurrent(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """States h_t (batch, time, H) after reading each (o_t, a_t), continuing from
        `initial_state` (batch, H) or from zero. The control ignores the state."""
        inputs = torch.cat(
            [
                nn.functional.one_hot(observations, OBSERVATIONS),
                nn.functional.one_hot(actions, ACTIONS),
            ],
            dim=-1,
        ).float()
        if not self.memory:
            return torch.tanh(self.core(inputs))
        initial = None if initial_state is None else initial_state.unsqueeze(0)
        states, _ = self.core(inputs, initial)
        return states

    def predict(self, states: torch.Tensor) -> Predictions:
        return Predictions(
            self.observation_head(states),
            self.reward_head(states).squeeze(-1),
            self.continue_head(states).squeeze(-1),
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> Predictions:
        return self.predict(self.recurrent(observations, actions))


def build(
    config: Config, device: torch.device, memory: bool = True
) -> tuple[WorldModel, torch.optim.Adam]:
    torch.manual_seed(config.seed)
    model = WorldModel(config.hidden, memory).to(device)
    return model, torch.optim.Adam(model.parameters(), lr=config.learning_rate)


def losses(model: WorldModel, episodes: Episodes) -> dict[str, torch.Tensor]:
    """Per-term losses averaged over valid steps; `total` is their sum."""
    predictions = model(episodes.observations, episodes.actions)
    valid = episodes.mask.float()
    count = valid.sum()
    observation = nn.functional.cross_entropy(
        predictions.observation_logits.transpose(1, 2),
        episodes.next_observations,
        reduction="none",
    )
    reward = (predictions.reward - episodes.rewards) ** 2
    continuation = nn.functional.binary_cross_entropy_with_logits(
        predictions.continue_logits, episodes.continues, reduction="none"
    )
    terms = {
        "observation": (observation * valid).sum() / count,
        "reward": (reward * valid).sum() / count,
        "continuation": (continuation * valid).sum() / count,
    }
    return terms | {"total": sum(terms.values())}


def train(
    model: WorldModel,
    optimizer: torch.optim.Optimizer,
    episodes: Episodes,
    config: Config,
    device: torch.device,
    generator: torch.Generator,
    steps: int,
) -> list[float]:
    total_losses = []
    count = episodes.mask.shape[0]
    for _ in range(steps):
        index = torch.randint(0, count, (config.batch,), generator=generator)
        optimizer.zero_grad()
        loss = losses(model, episodes.select(index).to(device))["total"]
        loss.backward()
        optimizer.step()
        total_losses.append(loss.item())
    return total_losses


def evaluate(
    model: WorldModel, episodes: Episodes, device: torch.device
) -> dict[str, Any]:
    """One-step prediction error on all valid steps, plus the two transition kinds
    that need memory."""
    episodes = episodes.to(device)
    with torch.no_grad():
        terms = losses(model, episodes)
        predictions = model(episodes.observations, episodes.actions)
    predicted_observation = predictions.observation_logits.argmax(-1)
    observation_correct = predicted_observation == episodes.next_observations
    continuation_correct = (predictions.continue_logits > 0) == (
        episodes.continues > 0.5
    )
    valid = episodes.mask

    turn = (
        valid
        & (episodes.observations == Observation.JUNCTION)
        & (episodes.actions != Action.FORWARD)
    )
    reward_error = (predictions.reward - episodes.rewards)[turn]
    reward_sign_correct = (predictions.reward[turn] > 0) == (episodes.rewards[turn] > 0)
    cues = episodes.cues.unsqueeze(1).expand_as(turn)
    reward_by_cue_and_turn = {
        f"cue_{cue.name.lower()}_turn_{action.name.lower()}": predictions.reward[
            turn & (cues == cue) & (episodes.actions == action)
        ]
        .mean()
        .item()
        for cue in Cue
        for action in (Action.LEFT, Action.RIGHT)
    }

    forward_in_corridor = (
        valid
        & (episodes.observations == Observation.CORRIDOR)
        & (episodes.actions == Action.FORWARD)
    )
    junction_arrival = forward_in_corridor & (
        episodes.next_observations == Observation.JUNCTION
    )
    return {
        "loss": {name: value.item() for name, value in terms.items()},
        "observation_accuracy": observation_correct[valid].float().mean().item(),
        "continuation_accuracy": continuation_correct[valid].float().mean().item(),
        "valid_steps": int(valid.sum()),
        "junction_turn": {
            "steps": int(turn.sum()),
            "reward_mse": (reward_error**2).mean().item(),
            "reward_sign_accuracy": reward_sign_correct.float().mean().item(),
            "mean_predicted_reward": reward_by_cue_and_turn,
        },
        "corridor_forward": {
            "steps": int(forward_in_corridor.sum()),
            "observation_accuracy": observation_correct[forward_in_corridor]
            .float()
            .mean()
            .item(),
        },
        "junction_arrival": {
            "steps": int(junction_arrival.sum()),
            "observation_accuracy": observation_correct[junction_arrival]
            .float()
            .mean()
            .item(),
        },
    }


def dataset_summary(episodes: Episodes) -> dict[str, Any]:
    lengths = episodes.mask.sum(1)
    terminated = ((episodes.continues == 0) & episodes.mask).any(1)
    junction_steps = (
        (episodes.observations == Observation.JUNCTION) & episodes.mask
    ).sum(1)
    return {
        "episodes": int(episodes.mask.shape[0]),
        "transitions": int(lengths.sum()),
        "mean_length": lengths.float().mean().item(),
        "terminated_fraction": terminated.float().mean().item(),
        "cue_right_fraction": episodes.cues.float().mean().item(),
        "positive_rewards": int((episodes.rewards > 0).sum()),
        "negative_rewards": int((episodes.rewards < 0).sum()),
        "mean_junction_steps": junction_steps.float().mean().item(),
        "max_junction_steps": int(junction_steps.max()),
    }


def save_checkpoint(
    path: Path,
    config: Config,
    step: int,
    model: WorldModel,
    optimizer: torch.optim.Optimizer,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "config": asdict(config),
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[Config, int, WorldModel, torch.optim.Adam]:
    data = torch.load(path, map_location=device, weights_only=True)
    if data.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    raw = data["config"]
    config = Config(**raw | {"report_steps": tuple(raw["report_steps"])})
    model, optimizer = build(config, device)
    model.load_state_dict(data["model"])
    optimizer.load_state_dict(data["optimizer"])
    return config, data["step"], model, optimizer


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train_with_curve(
    memory: bool,
    config: Config,
    training: Episodes,
    evaluation: Episodes,
    device: torch.device,
) -> tuple[WorldModel, torch.optim.Adam, list[float], dict[str, Any]]:
    """Train for `config.steps`, evaluating the held-out episodes at report steps."""
    model, optimizer = build(config, device, memory)
    generator = torch.Generator().manual_seed(config.seed)
    training_losses: list[float] = []
    curve = {}
    for step in sorted({0, *config.report_steps, config.steps}):
        if step > config.steps:
            break
        training_losses += train(
            model,
            optimizer,
            training,
            config,
            device,
            generator,
            step - len(training_losses),
        )
        curve[str(step)] = evaluate(model, evaluation, device)
    return model, optimizer, training_losses, curve


def run(config: Config, device: torch.device, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    evaluation_data_seed = config.seed + EVALUATION_DATA_SEED_OFFSET
    training = collect_episodes(
        config, config.training_episodes, torch.Generator().manual_seed(config.seed)
    )
    evaluation = collect_episodes(
        config,
        config.evaluation_episodes,
        torch.Generator().manual_seed(evaluation_data_seed),
    )
    model, optimizer, training_losses, curve = train_with_curve(
        True, config, training, evaluation, device
    )
    control, _, control_losses, control_curve = train_with_curve(
        False, config, training, evaluation, device
    )

    checkpoint = output.parent / "checkpoint.pt"
    save_checkpoint(checkpoint, config, config.steps, model, optimizer)
    _, restored_step, restored, _ = load_checkpoint(checkpoint, device)
    probe = evaluation.select(torch.arange(8)).to(device)
    with torch.no_grad():
        before = model(probe.observations, probe.actions)
        after = restored(probe.observations, probe.actions)
    roundtrip = all(
        torch.equal(getattr(before, f), getattr(after, f))
        for f in Predictions.__dataclass_fields__
    )

    final = str(config.steps)
    result: dict[str, Any] = {
        "stage": "2c-learned-dynamics-tmaze",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "config": asdict(config),
        "seeds": {
            "torch_initialization": config.seed,
            "training_data": config.seed,
            "evaluation_data": evaluation_data_seed,
            "minibatch_sampling": config.seed,
        },
        "device": str(device),
        "versions": {"python": platform.python_version(), "torch": torch.__version__},
        "environment": {
            "name": "TMaze",
            "observations": {o.name: int(o) for o in Observation},
            "actions": {a.name: int(a) for a in Action},
            "corridor_length": config.corridor_length,
            "max_steps": config.max_steps,
            "behavior_policy": "uniform random",
        },
        "training_data": dataset_summary(training),
        "evaluation_data": dataset_summary(evaluation),
        "model": {
            "name": "WorldModel (GRU)",
            "architecture": f"onehot obs+action ({OBSERVATIONS + ACTIONS}) -> "
            f"GRU({OBSERVATIONS + ACTIONS},{config.hidden}) -> heads: "
            f"Linear({config.hidden},{OBSERVATIONS}) next observation, "
            f"Linear({config.hidden},1) reward, Linear({config.hidden},1) continuation",
            "parameters": parameter_count(model),
            "parameter_shapes": {
                name: list(p.shape) for name, p in model.named_parameters()
            },
        },
        "control": {
            "name": "WorldModel (no memory)",
            "parameters": parameter_count(control),
            "final_training_loss": control_losses[-1],
            "evaluation_curve": control_curve,
            "evaluation": control_curve[final],
        },
        "optimizer": {"name": "Adam", "learning_rate": config.learning_rate},
        "loss": "cross_entropy(next observation) + mse(reward) "
        "+ binary_cross_entropy(continuation), averaged over valid steps",
        "training_loss_at_steps": {
            str(s): training_losses[s]
            for s in config.report_steps
            if s < len(training_losses)
        },
        "evaluation_curve": curve,
        "initial_evaluation": curve["0"],
        "evaluation": curve[final],
        "checkpoint": {
            "path": str(checkpoint),
            "step": restored_step,
            "roundtrip_identical_outputs": roundtrip,
        },
        "duration_seconds": time.monotonic() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--hidden", type=int, default=Config.hidden)
    parser.add_argument("--corridor-length", type=int, default=Config.corridor_length)
    parser.add_argument("--device", default="cpu", choices=DEVICES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.corridor_length < 1:
        parser.error("--steps and --corridor-length must be positive")
    config = Config(
        seed=args.seed,
        steps=args.steps,
        hidden=args.hidden,
        corridor_length=args.corridor_length,
    )
    output = (
        args.output
        or Path("runs")
        / datetime.now(UTC).strftime("stage2c-%Y%m%dT%H%M%S%fZ")
        / "metrics.json"
    )
    result = run(config, select_device(args.device), output)

    print(f"Device {result['device']}, torch {result['versions']['torch']}")
    print(
        f"Model {result['model']['architecture']}, "
        f"{result['model']['parameters']} parameters "
        f"(no-memory control {result['control']['parameters']})"
    )
    for name, shape in result["model"]["parameter_shapes"].items():
        print(f"  {name}: {shape}")
    data = result["training_data"]
    print(
        f"Training data: {data['episodes']} episodes, {data['transitions']} "
        f"transitions, mean length {data['mean_length']:.1f}, "
        f"+1 rewards {data['positive_rewards']}, -1 rewards {data['negative_rewards']}"
    )
    print(
        "Held-out one-step error by training step "
        "(GRU | no-memory control):\n"
        "   step  total   obs acc  turn reward mse  turn sign acc  "
        "junction-arrival acc"
    )
    for step, gru in result["evaluation_curve"].items():
        control = result["control"]["evaluation_curve"][step]
        print(
            f"  {step:>5}  {gru['loss']['total']:.3f}|{control['loss']['total']:.3f}"
            f"  {gru['observation_accuracy']:.0%}|{control['observation_accuracy']:.0%}"
            f"  {gru['junction_turn']['reward_mse']:.3f}|"
            f"{control['junction_turn']['reward_mse']:.3f}"
            f"  {gru['junction_turn']['reward_sign_accuracy']:.0%}|"
            f"{control['junction_turn']['reward_sign_accuracy']:.0%}"
            f"  {gru['junction_arrival']['observation_accuracy']:.0%}|"
            f"{control['junction_arrival']['observation_accuracy']:.0%}"
        )
    print("Mean predicted junction-turn reward (GRU | no-memory control):")
    gru_rewards = result["evaluation"]["junction_turn"]["mean_predicted_reward"]
    control_rewards = result["control"]["evaluation"]["junction_turn"][
        "mean_predicted_reward"
    ]
    for case, value in gru_rewards.items():
        print(f"  {case}: {value:+.3f} | {control_rewards[case]:+.3f}")
    print(
        f"Checkpoint {result['checkpoint']['path']} at step "
        f"{result['checkpoint']['step']}: identical outputs after reload = "
        f"{result['checkpoint']['roundtrip_identical_outputs']}"
    )
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
