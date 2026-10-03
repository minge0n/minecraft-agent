"""Run one Stage 2 experiment over many seeds with few processes at a time.

macOS only. Each seed runs in its own process under `taskpolicy -b`, the
background QoS of macOS. The scheduler then uses lower clock speeds and yields
to other work, so a long sweep makes less heat. The canonical runtime uses one
thread per process (docs/decisions/reproducibility.md), so neither the number
of parallel jobs nor the QoS changes the results.

Example:

    .venv/bin/python -m minecraft_rl.sweep dreamer_loop --seeds 0-9 --jobs 2 \\
        --output-root runs/stage2f -- --world-model rssm
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

TASKPOLICY = Path("/usr/sbin/taskpolicy")
EXPERIMENTS = (
    "parity",
    "cue_recall",
    "world_model",
    "imagination",
    "actor_critic",
    "dreamer_loop",
)


def parse_seeds(text: str) -> list[int]:
    """Seeds from "0-9", "0,3,5" or a mix such as "0-2,7"."""
    seeds: list[int] = []
    for part in text.split(","):
        first, _, last = part.partition("-")
        seeds.extend(range(int(first), int(last or first) + 1))
    return seeds


def command(
    experiment: str,
    seed: int,
    output: Path,
    extra: list[str],
    background: bool,
) -> list[str]:
    run = [
        sys.executable,
        "-m",
        f"minecraft_rl.{experiment}",
        "--seed",
        str(seed),
        "--output",
        str(output),
        *extra,
    ]
    return [str(TASKPOLICY), "-b", *run] if background else run


def sweep(
    experiment: str,
    seeds: list[int],
    jobs: int,
    output_root: Path,
    extra: list[str],
    background: bool,
) -> dict[int, int]:
    """Run every seed, at most `jobs` at a time. Returns the exit code per seed."""
    pending = list(seeds)
    running: dict[int, tuple[subprocess.Popen[bytes], object]] = {}
    codes: dict[int, int] = {}
    while pending or running:
        while pending and len(running) < jobs:
            seed = pending.pop(0)
            directory = output_root / f"seed{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            log = (directory / "log.txt").open("wb")
            process = subprocess.Popen(
                command(
                    experiment,
                    seed,
                    directory / "metrics.json",
                    extra,
                    background,
                ),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            running[seed] = (process, log)
        for seed, (process, log) in list(running.items()):
            if process.poll() is None:
                continue
            log.close()
            codes[seed] = process.returncode
            status = "done" if process.returncode == 0 else "FAILED"
            print(f"seed {seed}: {status} (exit {process.returncode})", flush=True)
            del running[seed]
        time.sleep(0.5)
    return codes


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="run at normal priority instead of macOS background QoS",
    )
    parser.add_argument("extra", nargs="*", help="arguments after -- go to the run")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if not args.foreground and not TASKPOLICY.exists():
        parser.error(f"{TASKPOLICY} not found; this runner supports macOS only")
    codes = sweep(
        args.experiment,
        parse_seeds(args.seeds),
        args.jobs,
        args.output_root,
        args.extra,
        background=not args.foreground,
    )
    failed = sorted(seed for seed, code in codes.items() if code != 0)
    if failed:
        raise SystemExit(f"failed seeds: {failed}")


if __name__ == "__main__":
    main()
