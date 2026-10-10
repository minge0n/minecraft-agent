"""Split one long experiment run into several processes.

A run gets a state file and a time limit. At each boundary between two units
of work (an iteration, a report step, a trained model), the run calls
`Session.boundary`. If the time limit has passed, the run saves everything that
the remaining units need and the process exits with `INCOMPLETE_EXIT_CODE`.
The next process with the same state file loads the state and continues at
the same boundary.

A resumed run gives the same results as a run in one process. This holds
because the state contains every random generator state, every model and
optimizer state, and all data that the run produced. The sweep runner
(`sweep.py`) uses this to keep each process short.
"""

import json
import os
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy
import torch

INCOMPLETE_EXIT_CODE = 75
STATE_FORMAT = "experiment-resume-state-v1"


class Incomplete(Exception):
    """The run saved its state at a boundary and must continue later."""


def tensors_of(value: Any) -> dict[str, torch.Tensor]:
    """The tensor fields of a dataclass such as `Episodes`, for a state file."""
    return {name: getattr(value, name) for name in value.__dataclass_fields__}


def generator_state(generator: torch.Generator) -> torch.Tensor:
    return generator.get_state()


def restore_generator(state: torch.Tensor) -> torch.Generator:
    generator = torch.Generator()
    generator.set_state(state)
    return generator


def _global_random_state() -> dict[str, Any]:
    kind, keys, position, has_gauss, cached = numpy.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "numpy": (
            kind,
            torch.from_numpy(keys.astype(numpy.int64)),
            position,
            has_gauss,
            cached,
        ),
        "python": random.getstate(),
    }


def _restore_global_random_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    kind, keys, position, has_gauss, cached = state["numpy"]
    numpy.random.set_state(
        (kind, keys.numpy().astype(numpy.uint32), position, has_gauss, cached)
    )
    version, values, gauss = state["python"]
    random.setstate((version, tuple(values), gauss))


class Session:
    """One process of a run that can stop at boundaries and continue later.

    `identity` describes the run (experiment name and configuration). A state
    file of another run raises an error instead of being continued. Without a
    state path, the session never stops and never saves.
    """

    def __init__(
        self,
        path: Path | None,
        stop_after_seconds: float | None,
        identity: dict[str, Any],
    ) -> None:
        self.path = path
        self.stop_after_seconds = stop_after_seconds
        self.identity = json.dumps(identity, sort_keys=True, default=str)
        self.started = time.monotonic()
        self.elapsed_before = 0.0
        self.sessions_before = 0
        self.units_done = 0

    def load(self) -> dict[str, Any] | None:
        """The saved state of this run, or None for a new run. Call
        `restore_global_random_state` after the models are built."""
        if self.path is None or not self.path.exists():
            return None
        data = torch.load(self.path, weights_only=True)
        if data.get("format") != STATE_FORMAT:
            raise ValueError(f"{self.path} is not a {STATE_FORMAT} file")
        if data["identity"] != self.identity:
            raise ValueError(
                f"{self.path} belongs to another run configuration; delete it "
                "to start again"
            )
        self.elapsed_before = data["elapsed_seconds"]
        self.sessions_before = data["sessions"]
        self._global_random = data["global_random"]
        return data["state"]

    def restore_global_random_state(self) -> None:
        """Restore the global Python, NumPy and PyTorch generators of the
        stopped process. Building a model reseeds the global PyTorch
        generator, so call this after the models are built."""
        _restore_global_random_state(self._global_random)

    @property
    def elapsed_seconds(self) -> float:
        """Run time over all processes of this run."""
        return self.elapsed_before + time.monotonic() - self.started

    @property
    def sessions(self) -> int:
        return self.sessions_before + 1

    def boundary(
        self, state: Callable[[], dict[str, Any]], force: bool = False
    ) -> None:
        """Mark the end of one unit of work. If the time limit has passed,
        or `force` is set, save `state()` and raise `Incomplete`. Each process
        does at least one unit, so a run always makes progress. `force` is for
        a planned stop, such as a staged training budget."""
        self.units_done += 1
        if self.path is None:
            return
        if not force and (
            self.stop_after_seconds is None
            or time.monotonic() - self.started < self.stop_after_seconds
        ):
            return
        data = {
            "format": STATE_FORMAT,
            "identity": self.identity,
            "elapsed_seconds": self.elapsed_seconds,
            "sessions": self.sessions,
            "global_random": _global_random_state(),
            "state": state(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_name(self.path.name + ".partial")
        torch.save(data, partial)
        os.replace(partial, self.path)
        raise Incomplete(f"saved state after {self.units_done} units to {self.path}")

    def finish(self) -> None:
        """Delete the state file after the run wrote its final results."""
        if self.path is not None:
            self.path.unlink(missing_ok=True)


def add_arguments(parser: Any) -> None:
    parser.add_argument(
        "--state",
        type=Path,
        help="state file: continue the run from it if it exists, save to it when "
        "--stop-after passes",
    )
    parser.add_argument(
        "--stop-after",
        type=float,
        help="seconds after which the run saves its state at the next boundary "
        f"and exits with code {INCOMPLETE_EXIT_CODE}",
    )


def session_from(args: Any, identity: dict[str, Any]) -> Session:
    if args.stop_after is not None and args.state is None:
        raise SystemExit("--stop-after needs --state")
    return Session(args.state, args.stop_after, identity)
