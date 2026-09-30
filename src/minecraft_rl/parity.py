"""Stage 2A neural sanity check: a tiny MLP learns 4-bit parity (docs/stage2a.md).

The task is a fixed, deterministic supervised mapping, so every run can be checked
exactly: gradients against finite differences, parameter changes after one step,
checkpoint save/load, resuming, and device selection.
"""

import argparse
import itertools
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.func import functional_call

CLASSES = 2
CHECKPOINT_FORMAT = "parity-mlp-v1"
REPORT_STEPS = (0, 1, 10, 50, 100, 200, 400)


@dataclass(frozen=True)
class Config:
    seed: int = 0
    bits: int = 4
    hidden: int = 16
    learning_rate: float = 0.01
    steps: int = 500


def parity_dataset(bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Every `bits`-bit pattern and its parity.

    Returns inputs of shape (2**bits, bits) with bits encoded as -1.0 and +1.0, and
    labels of shape (2**bits,) holding 1 where the number of set bits is odd.
    """
    patterns = torch.tensor(
        list(itertools.product((0, 1), repeat=bits)), dtype=torch.float32
    )
    labels = patterns.sum(dim=1).remainder(2).long()
    return patterns * 2.0 - 1.0, labels


class ParityMLP(nn.Module):
    """inputs (batch, bits) -> tanh hidden (batch, hidden) -> logits (batch, 2)."""

    def __init__(self, bits: int, hidden: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(bits, hidden)
        self.output = nn.Linear(hidden, CLASSES)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(torch.tanh(self.hidden(inputs)))


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this host")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available on this host")
    if requested not in ("cpu", "cuda", "mps"):
        raise ValueError(f"unknown device {requested!r}")
    return torch.device(requested)


def build(config: Config, device: torch.device) -> tuple[ParityMLP, torch.optim.Adam]:
    """A freshly initialized model and its optimizer; seeds torch first."""
    torch.manual_seed(config.seed)
    model = ParityMLP(config.bits, config.hidden).to(device)
    return model, torch.optim.Adam(model.parameters(), lr=config.learning_rate)


def evaluate(
    model: ParityMLP, inputs: torch.Tensor, labels: torch.Tensor
) -> dict[str, float]:
    with torch.no_grad():
        logits = model(inputs)
        loss = nn.functional.cross_entropy(logits, labels)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
    return {"loss": loss.item(), "accuracy": accuracy.item()}


def train_step(
    model: ParityMLP,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    """One full-batch gradient step; returns the loss and norms before the update."""
    optimizer.zero_grad()
    loss = nn.functional.cross_entropy(model(inputs), labels)
    loss.backward()
    gradient_norm = torch.linalg.vector_norm(
        torch.stack([p.grad.norm() for p in model.parameters()])
    )
    optimizer.step()
    return {"loss": loss.item(), "gradient_norm": gradient_norm.item()}


def train(
    model: ParityMLP,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    steps: int,
) -> list[dict[str, float]]:
    return [train_step(model, optimizer, inputs, labels) for _ in range(steps)]


def gradients_match_finite_differences(model: ParityMLP, bits: int) -> bool:
    """Compare autograd gradients of the loss with central finite differences.

    Runs in float64 on a CPU copy, because finite differences in float32 are too
    imprecise to separate a wrong gradient from rounding error.
    """
    copy = ParityMLP(bits, model.hidden.out_features).double()
    copy.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()})
    inputs, labels = parity_dataset(bits)
    inputs = inputs.double()
    names = [name for name, _ in copy.named_parameters()]
    parameters = tuple(
        p.detach().clone().requires_grad_(True) for p in copy.parameters()
    )

    def loss(*values: torch.Tensor) -> torch.Tensor:
        logits = functional_call(copy, dict(zip(names, values, strict=True)), inputs)
        return nn.functional.cross_entropy(logits, labels)

    return torch.autograd.gradcheck(loss, parameters, raise_exception=False)


def parameter_changes(
    model: ParityMLP,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, dict[str, float]]:
    """Take one step and report, per parameter, its gradient and update sizes."""
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    train_step(model, optimizer, inputs, labels)
    return {
        name: {
            "gradient_norm": p.grad.norm().item(),
            "update_norm": (p.detach() - before[name]).norm().item(),
        }
        for name, p in model.named_parameters()
    }


def save_checkpoint(
    path: Path,
    config: Config,
    step: int,
    model: ParityMLP,
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
) -> tuple[Config, int, ParityMLP, torch.optim.Adam]:
    """Restore a checkpoint; `weights_only` refuses anything but tensors and data."""
    data = torch.load(path, map_location=device, weights_only=True)
    if data.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    config = Config(**data["config"])
    model, optimizer = build(config, device)
    model.load_state_dict(data["model"])
    optimizer.load_state_dict(data["optimizer"])
    return config, data["step"], model, optimizer


def optimizer_summary(
    model: ParityMLP, optimizer: torch.optim.Adam
) -> dict[str, dict[str, float]]:
    """Per parameter: Adam's step count and the norms of its two moment estimates."""
    summary = {}
    for name, parameter in model.named_parameters():
        state = optimizer.state[parameter]
        summary[name] = {
            "step": float(state["step"]),
            "first_moment_norm": state["exp_avg"].norm().item(),
            "second_moment_norm": state["exp_avg_sq"].norm().item(),
        }
    return summary


def run(config: Config, device: torch.device, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    inputs, labels = parity_dataset(config.bits)
    inputs, labels = inputs.to(device), labels.to(device)
    model, optimizer = build(config, device)
    initial = evaluate(model, inputs, labels)
    gradients_ok = gradients_match_finite_differences(model, config.bits)
    first_step = parameter_changes(model, optimizer, inputs, labels)
    history = [{"loss": initial["loss"], "gradient_norm": float("nan")}]
    history += train(model, optimizer, inputs, labels, config.steps - 1)
    final = evaluate(model, inputs, labels)

    checkpoint = output.parent / "checkpoint.pt"
    save_checkpoint(checkpoint, config, config.steps, model, optimizer)
    _, restored_step, restored, _ = load_checkpoint(checkpoint, device)
    with torch.no_grad():
        roundtrip = torch.equal(model(inputs), restored(inputs))

    with torch.no_grad():
        probabilities = torch.softmax(model(inputs), dim=1)[:, 1]
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    result: dict[str, Any] = {
        "stage": "2a-neural-sanity-parity",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": git.stdout.strip() if git.returncode == 0 else None,
        "config": asdict(config),
        "seed_torch": config.seed,
        "device": str(device),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "model": {
            "name": "ParityMLP",
            "architecture": f"Linear({config.bits},{config.hidden}) -> tanh -> "
            f"Linear({config.hidden},{CLASSES})",
            "parameters": sum(p.numel() for p in model.parameters()),
            "parameter_shapes": {
                name: list(p.shape) for name, p in model.named_parameters()
            },
        },
        "optimizer": {"name": "Adam", "learning_rate": config.learning_rate},
        "loss": "cross_entropy",
        "dataset": {
            "inputs_shape": list(inputs.shape),
            "labels_shape": list(labels.shape),
        },
        "gradient_check_passed": gradients_ok,
        "first_step": first_step,
        "initial": initial,
        "final": final,
        "loss_at_steps": {
            str(step): history[step]["loss"]
            for step in REPORT_STEPS
            if step < len(history)
        },
        "optimizer_state": optimizer_summary(model, optimizer),
        "predictions": [
            {
                "bits": [int(b > 0) for b in row.tolist()],
                "label": int(label),
                "probability_odd": probability,
            }
            for row, label, probability in zip(
                inputs.cpu(), labels.cpu(), probabilities.cpu().tolist(), strict=True
            )
        ],
        "checkpoint": {
            "path": str(checkpoint),
            "step": restored_step,
            "roundtrip_identical_outputs": roundtrip,
        },
        "duration_seconds": time.monotonic() - started,
    }
    (output.parent).mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--hidden", type=int, default=Config.hidden)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument(
        "--device", default="cpu", choices=("auto", "cpu", "mps", "cuda")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    config = Config(
        seed=args.seed,
        hidden=args.hidden,
        learning_rate=args.learning_rate,
        steps=args.steps,
    )
    output = (
        args.output
        or Path("runs")
        / datetime.now(UTC).strftime("stage2a-%Y%m%dT%H%M%S%fZ")
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
    print(
        f"Gradient check against finite differences: {result['gradient_check_passed']}"
    )
    print("First step (gradient norm, update norm):")
    for name, change in result["first_step"].items():
        print(f"  {name}: {change['gradient_norm']:.4f}, {change['update_norm']:.4f}")
    print("Loss by step:")
    for step, loss in result["loss_at_steps"].items():
        print(f"  {step:>4}: {loss:.5f}")
    print(
        f"Final loss {result['final']['loss']:.5f}, "
        f"accuracy {result['final']['accuracy']:.0%}"
    )
    print("Predictions (bits -> parity, P(odd)):")
    for row in result["predictions"]:
        bits = "".join(str(b) for b in row["bits"])
        print(f"  {bits} -> {row['label']}  {row['probability_odd']:.3f}")
    print(
        f"Checkpoint {result['checkpoint']['path']} at step "
        f"{result['checkpoint']['step']}: identical outputs after reload = "
        f"{result['checkpoint']['roundtrip_identical_outputs']}"
    )
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
