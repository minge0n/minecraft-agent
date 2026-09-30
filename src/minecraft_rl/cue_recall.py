"""Stage 2B recurrent sanity check: a GRU recalls a cue across a delay.

See docs/stage2b.md. Each sequence starts with a cue token (A or B), continues with
identical distractor tokens and ends with a query token. The target is the cue,
predicted at the query step. Every input after the first is the same for both
classes, so only a model that carries the cue in its recurrent state can beat
chance. A no-memory control that sees only the current token is trained the same
way and must stay at 50%.
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

CUE_A, CUE_B, DISTRACTOR, QUERY = 0, 1, 2, 3
VOCABULARY = 4
CLASSES = 2
CHECKPOINT_FORMAT = "cue-recall-gru-v1"
REPORT_STEPS = (0, 50, 100, 200, 400, 800, 1199)


@dataclass(frozen=True)
class Config:
    seed: int = 0
    embedding: int = 8
    hidden: int = 16
    learning_rate: float = 0.01
    steps: int = 1200
    batch: int = 64
    min_delay: int = 1
    max_delay: int = 10
    evaluation_delays: tuple[int, ...] = (1, 5, 10, 20, 40)


def cue_recall_batch(
    cues: torch.Tensor, delay: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token sequences (batch, delay + 2) and targets (batch,) for the given cues.

    `delay` is the number of distractor tokens between the cue and the query.
    """
    batch = cues.shape[0]
    tokens = torch.full((batch, delay + 2), DISTRACTOR, dtype=torch.long)
    tokens[:, 0] = cues
    tokens[:, -1] = QUERY
    return tokens, cues.clone()


def sample_batch(
    generator: torch.Generator, config: Config
) -> tuple[torch.Tensor, torch.Tensor]:
    delay = int(
        torch.randint(config.min_delay, config.max_delay + 1, (1,), generator=generator)
    )
    cues = torch.randint(0, CLASSES, (config.batch,), generator=generator)
    return cue_recall_batch(cues, delay)


class CueRecallGRU(nn.Module):
    """tokens (batch, time) -> embedding (batch, time, E) -> GRU hidden (batch, time, H)
    -> logits of the final step (batch, 2)."""

    def __init__(self, embedding: int, hidden: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCABULARY, embedding)
        self.gru = nn.GRU(embedding, hidden, batch_first=True)
        self.readout = nn.Linear(hidden, CLASSES)

    def hidden_states(self, tokens: torch.Tensor) -> torch.Tensor:
        states, _ = self.gru(self.embedding(tokens))
        return states

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.readout(self.hidden_states(tokens)[:, -1])


class CurrentTokenOnly(nn.Module):
    """No-memory control: the prediction depends on the final token alone."""

    def __init__(self, embedding: int, hidden: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCABULARY, embedding)
        self.hidden = nn.Linear(embedding, hidden)
        self.readout = nn.Linear(hidden, CLASSES)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.readout(torch.tanh(self.hidden(self.embedding(tokens[:, -1]))))


def build(
    config: Config, device: torch.device, memory: bool = True
) -> tuple[nn.Module, torch.optim.Adam]:
    torch.manual_seed(config.seed)
    model_type = CueRecallGRU if memory else CurrentTokenOnly
    model = model_type(config.embedding, config.hidden).to(device)
    return model, torch.optim.Adam(model.parameters(), lr=config.learning_rate)


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Config,
    device: torch.device,
    generator: torch.Generator,
    steps: int,
) -> list[float]:
    losses = []
    for _ in range(steps):
        tokens, targets = sample_batch(generator, config)
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(tokens.to(device)), targets.to(device))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def evaluate(model: nn.Module, delay: int, device: torch.device) -> dict[str, float]:
    """Accuracy and loss on both cues at one fixed delay (deterministic)."""
    tokens, targets = cue_recall_batch(torch.tensor([CUE_A, CUE_B]), delay)
    with torch.no_grad():
        logits = model(tokens.to(device))
        loss = nn.functional.cross_entropy(logits, targets.to(device))
        accuracy = (logits.argmax(dim=1).cpu() == targets).float().mean()
        probability_b = torch.softmax(logits, dim=1)[:, 1].cpu()
    return {
        "delay": delay,
        "accuracy": accuracy.item(),
        "loss": loss.item(),
        "probability_b_given_a": probability_b[0].item(),
        "probability_b_given_b": probability_b[1].item(),
    }


def cue_gradient(model: CueRecallGRU, delay: int, device: torch.device) -> float:
    """Norm of d(logit margin z_B - z_A at the query) / d(cue embedding at step 0).

    Nonzero means the final prediction depends on the first input through the
    recurrent state. The margin is used rather than the loss, whose gradient
    vanishes once predictions are confident.
    """
    tokens, _ = cue_recall_batch(torch.tensor([CUE_A, CUE_B]), delay)
    embedded = model.embedding(tokens.to(device)).detach().requires_grad_(True)
    states, _ = model.gru(embedded)
    logits = model.readout(states[:, -1])
    (logits[:, 1] - logits[:, 0]).sum().backward()
    return embedded.grad[:, 0].norm().item()


def state_separation(
    model: CueRecallGRU, delay: int, device: torch.device
) -> list[float]:
    """Distance between the hidden states after cue A and after cue B, per step.

    The two sequences have identical inputs from step 1 on, so any distance there is
    information about the cue that the recurrent state is carrying.
    """
    tokens, _ = cue_recall_batch(torch.tensor([CUE_A, CUE_B]), delay)
    with torch.no_grad():
        states = model.hidden_states(tokens.to(device))
    return (states[0] - states[1]).norm(dim=1).cpu().tolist()


def save_checkpoint(
    path: Path,
    config: Config,
    step: int,
    model: CueRecallGRU,
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
) -> tuple[Config, int, CueRecallGRU, torch.optim.Adam]:
    data = torch.load(path, map_location=device, weights_only=True)
    if data.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    raw = data["config"]
    config = Config(**raw | {"evaluation_delays": tuple(raw["evaluation_delays"])})
    model, optimizer = build(config, device)
    model.load_state_dict(data["model"])
    optimizer.load_state_dict(data["optimizer"])
    return config, data["step"], model, optimizer


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def run(config: Config, device: torch.device, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    model, optimizer = build(config, device)
    initial = [evaluate(model, d, device) for d in config.evaluation_delays]
    initial_cue_gradient = {
        str(d): cue_gradient(model, d, device) for d in config.evaluation_delays
    }
    optimizer.zero_grad()
    losses = train(
        model,
        optimizer,
        config,
        device,
        torch.Generator().manual_seed(config.seed),
        config.steps,
    )
    control, control_optimizer = build(config, device, memory=False)
    control_losses = train(
        control,
        control_optimizer,
        config,
        device,
        torch.Generator().manual_seed(config.seed),
        config.steps,
    )

    checkpoint = output.parent / "checkpoint.pt"
    save_checkpoint(checkpoint, config, config.steps, model, optimizer)
    _, restored_step, restored, _ = load_checkpoint(checkpoint, device)
    probe, _ = cue_recall_batch(torch.tensor([CUE_A, CUE_B]), config.max_delay)
    with torch.no_grad():
        roundtrip = torch.equal(model(probe.to(device)), restored(probe.to(device)))

    result: dict[str, Any] = {
        "stage": "2b-recurrent-cue-recall",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "config": asdict(config),
        "seed_torch": config.seed,
        "device": str(device),
        "versions": {"python": platform.python_version(), "torch": torch.__version__},
        "task": {
            "tokens": {
                "cue_a": CUE_A,
                "cue_b": CUE_B,
                "distractor": DISTRACTOR,
                "query": QUERY,
            },
            "training_delays": [config.min_delay, config.max_delay],
            "sequence_length": "delay + 2",
        },
        "model": {
            "name": "CueRecallGRU",
            "architecture": f"Embedding({VOCABULARY},{config.embedding}) -> "
            f"GRU({config.embedding},{config.hidden}) -> "
            f"Linear({config.hidden},{CLASSES}) on the final step",
            "parameters": parameter_count(model),
            "parameter_shapes": {
                name: list(p.shape) for name, p in model.named_parameters()
            },
        },
        "control": {
            "name": "CurrentTokenOnly",
            "parameters": parameter_count(control),
            "final_training_loss": control_losses[-1],
            "evaluation": [
                evaluate(control, d, device) for d in config.evaluation_delays
            ],
        },
        "optimizer": {"name": "Adam", "learning_rate": config.learning_rate},
        "loss": "cross_entropy at the query step",
        "loss_at_steps": {
            str(step): losses[step] for step in REPORT_STEPS if step < len(losses)
        },
        "initial_evaluation": initial,
        "evaluation": [evaluate(model, d, device) for d in config.evaluation_delays],
        "cue_gradient_norm": {
            "initial": initial_cue_gradient,
            "trained": {
                str(d): cue_gradient(model, d, device) for d in config.evaluation_delays
            },
        },
        "state_separation": {
            str(d): state_separation(model, d, device)
            for d in (config.max_delay, max(config.evaluation_delays))
        },
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
    parser.add_argument("--max-delay", type=int, default=Config.max_delay)
    parser.add_argument("--device", default="cpu", choices=DEVICES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.max_delay < Config.min_delay:
        parser.error("--steps must be positive and --max-delay at least 1")
    config = Config(
        seed=args.seed, steps=args.steps, hidden=args.hidden, max_delay=args.max_delay
    )
    output = (
        args.output
        or Path("runs")
        / datetime.now(UTC).strftime("stage2b-%Y%m%dT%H%M%S%fZ")
        / "metrics.json"
    )
    result = run(config, select_device(args.device), output)

    print(f"Device {result['device']}, torch {result['versions']['torch']}")
    print(
        f"Model {result['model']['architecture']}, "
        f"{result['model']['parameters']} parameters"
    )
    for name, shape in result["model"]["parameter_shapes"].items():
        print(f"  {name}: {shape}")
    print(f"Training delays {config.min_delay}-{config.max_delay}; loss by step:")
    for step, loss in result["loss_at_steps"].items():
        print(f"  {step:>5}: {loss:.5f}")
    print("Delay  GRU acc  P(B|A)  P(B|B)  cue grad init -> trained  | no-memory acc")
    gradients = result["cue_gradient_norm"]
    for gru, control in zip(
        result["evaluation"], result["control"]["evaluation"], strict=True
    ):
        delay = str(gru["delay"])
        print(
            f"  {delay:>3}  {gru['accuracy']:>6.0%}  {gru['probability_b_given_a']:.3f}"
            f"   {gru['probability_b_given_b']:.3f}"
            f"   {gradients['initial'][delay]:.2e} -> {gradients['trained'][delay]:.2e}"
            f"  | {control['accuracy']:.0%}"
        )
    for delay, distances in result["state_separation"].items():
        shown = ", ".join(f"{d:.2f}" for d in distances)
        print(f"State distance A vs B per step, delay {delay}: {shown}")
    print(
        f"Checkpoint {result['checkpoint']['path']} at step "
        f"{result['checkpoint']['step']}: identical outputs after reload = "
        f"{result['checkpoint']['roundtrip_identical_outputs']}"
    )
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
