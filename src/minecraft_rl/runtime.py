"""The canonical runtime for the controlled CPU experiments of Stage 2.

See docs/decisions/reproducibility.md. Seed-level results of the toy experiments
depend on the order of floating-point additions, and the thread count changes
that order. `configure` fixes the thread counts, the seeds of every random
number generator, the default dtype and the deterministic-algorithm mode.
Call it once, at the start of `main`, before any tensor work.
"""

import os
import platform
import random
from typing import Any

import numpy
import torch

CANONICAL_INTRA_OP_THREADS = 1
CANONICAL_INTER_OP_THREADS = 1

_configured_seed: int | None = None


def configure(
    seed: int,
    intra_op_threads: int = CANONICAL_INTRA_OP_THREADS,
    inter_op_threads: int = CANONICAL_INTER_OP_THREADS,
) -> None:
    """Fix threads, seeds, dtype and deterministic algorithms for this process.

    PyTorch accepts a new inter-op thread count only before its first parallel
    work. A process that already did such work keeps its count, and
    `metadata()` then reports the count in use.
    """
    global _configured_seed
    torch.set_num_threads(intra_op_threads)
    if torch.get_num_interop_threads() != inter_op_threads:
        try:
            torch.set_num_interop_threads(inter_op_threads)
        except RuntimeError:
            pass
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)
    _configured_seed = seed


def metadata(device: torch.device) -> dict[str, Any]:
    """The runtime settings in effect, for run metadata."""
    return {
        "canonical": (
            _configured_seed is not None
            and torch.get_num_threads() == CANONICAL_INTRA_OP_THREADS
            and torch.get_num_interop_threads() == CANONICAL_INTER_OP_THREADS
        ),
        "global_seed_python_numpy_torch": _configured_seed,
        "intra_op_threads": torch.get_num_threads(),
        "inter_op_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "default_dtype": str(torch.get_default_dtype()),
        "device": str(device),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": numpy.__version__,
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
            "cpu_count": os.cpu_count(),
        },
        "environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
