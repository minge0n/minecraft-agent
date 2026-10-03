"""Run one Stage 2 experiment over many seeds with little heat.

macOS only. Three mechanisms limit heat, and none of them changes results,
because the canonical runtime uses one thread per process
(docs/decisions/reproducibility.md):

1. `--jobs` limits the number of seed processes that run at the same time.
2. Each process runs under `taskpolicy -b`, the background QoS of macOS. The
   scheduler then uses lower clock speeds and yields to other work.
3. Before it starts a new seed, the runner reads the thermal pressure level of
   macOS. If the level is above `--max-thermal-level`, it waits until the level
   drops. `--cooldown` adds a fixed pause after each finished seed.

Example:

    .venv/bin/python -m minecraft_rl.sweep dreamer_loop --seeds 0-9 --jobs 2 \\
        --output-root runs/stage2f -- --world-model rssm
"""

import argparse
import ctypes
import ctypes.util
import subprocess
import sys
import time
from pathlib import Path

TASKPOLICY = Path("/usr/sbin/taskpolicy")
THERMAL_NOTIFICATION = b"com.apple.system.thermalpressurelevel"
THERMAL_LEVELS = ("nominal", "moderate", "heavy", "trapping", "sleeping")
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


def thermal_pressure() -> int | None:
    """The current macOS thermal pressure level (0 nominal, 1 moderate,
    2 heavy, 3 trapping, 4 sleeping), or None where it cannot be read."""
    library = ctypes.util.find_library("System")
    if library is None:
        return None
    system = ctypes.CDLL(library)
    token = ctypes.c_int()
    if system.notify_register_check(THERMAL_NOTIFICATION, ctypes.byref(token)):
        return None
    level = ctypes.c_uint64()
    status = system.notify_get_state(token, ctypes.byref(level))
    system.notify_cancel(token)
    return None if status else int(level.value)


def wait_for_cool_system(max_level: int, poll_seconds: float) -> None:
    """Block while the thermal pressure level is above `max_level`."""
    reported = False
    while (level := thermal_pressure()) is not None and level > max_level:
        if not reported:
            name = THERMAL_LEVELS[min(level, len(THERMAL_LEVELS) - 1)]
            print(f"thermal pressure {name}: waiting before next seed", flush=True)
            reported = True
        time.sleep(poll_seconds)


def sweep(
    experiment: str,
    seeds: list[int],
    jobs: int,
    output_root: Path,
    extra: list[str],
    background: bool,
    max_thermal_level: int = 0,
    cooldown_seconds: float = 0.0,
) -> dict[int, int]:
    """Run every seed, at most `jobs` at a time. Returns the exit code per seed."""
    pending = list(seeds)
    running: dict[int, tuple[subprocess.Popen[bytes], object]] = {}
    codes: dict[int, int] = {}
    while pending or running:
        while pending and len(running) < jobs:
            wait_for_cool_system(max_thermal_level, poll_seconds=10.0)
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
            if pending and cooldown_seconds > 0:
                time.sleep(cooldown_seconds)
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
    parser.add_argument(
        "--max-thermal-level",
        type=int,
        default=0,
        help="start a new seed only at or below this macOS thermal pressure level "
        "(0 nominal, 1 moderate, 2 heavy)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=0.0,
        help="seconds to pause after each finished seed",
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
        max_thermal_level=args.max_thermal_level,
        cooldown_seconds=args.cooldown,
    )
    failed = sorted(seed for seed, code in codes.items() if code != 0)
    if failed:
        raise SystemExit(f"failed seeds: {failed}")


if __name__ == "__main__":
    main()
