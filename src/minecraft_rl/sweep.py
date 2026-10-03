"""Run one Stage 2 experiment over many seeds with little heat.

These mechanisms limit heat. None of them changes results, because the
canonical runtime uses one thread per process and every random draw comes
from a seeded generator (docs/decisions/reproducibility.md):

1. `--jobs` limits the number of seed processes that run at the same time
   (default 1).
2. `--duty-cycle` limits the share of time that a process may compute. In
   every period of `--period` seconds, the runner stops all seed processes
   with SIGSTOP for the remaining share and resumes them with SIGCONT. The
   default 0.5 halves the average power of the sweep.
3. On macOS, each process runs under `taskpolicy -b`, the background QoS of
   macOS. The scheduler then prefers low clock speeds and lets other work go
   first.
4. On macOS, the runner reads the thermal pressure level. While the level is
   above `--max-thermal-level`, it stops all seed processes and starts no new
   seed. `--cooldown` adds a pause after each finished seed.

On other systems, the runner uses only 1 and 2. A seed whose `metrics.json`
already exists is skipped, so an interrupted sweep continues where it
stopped. `--rerun` disables this. Next to each `metrics.json`, the runner
writes `sweep.json` with the throttle settings and the measured pauses.

Example:

    .venv/bin/python -m minecraft_rl.sweep dreamer_loop --seeds 0-9 \\
        --output-root runs/stage2f -- --world-model rssm
"""

import argparse
import ctypes
import ctypes.util
import json
import os
import platform
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import IO, Any

TASKPOLICY = Path("/usr/sbin/taskpolicy")
THERMAL_NOTIFICATION = b"com.apple.system.thermalpressurelevel"
THERMAL_LEVELS = ("nominal", "moderate", "heavy", "trapping", "sleeping")
THERMAL_POLL_SECONDS = 10.0
TERMINATE_GRACE_SECONDS = 10.0
EXPERIMENTS = (
    "parity",
    "cue_recall",
    "world_model",
    "imagination",
    "actor_critic",
    "dreamer_loop",
    "stochastic_world",
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
    if platform.system() != "Darwin":
        return None
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


def too_hot(max_level: int) -> bool:
    level = thermal_pressure()
    return level is not None and level > max_level


def signal_all(processes: list[subprocess.Popen[bytes]], number: int) -> None:
    """Send a signal to the process group of every process that still runs.
    Each seed process leads its own group, so the signal also reaches a
    program that `taskpolicy` started. A process can exit between the check
    and the signal. macOS then reports ESRCH, or EPERM while the exited
    process is not yet reaped, and both are ignored."""
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, number)
            except (ProcessLookupError, PermissionError):
                pass


def wait_for_cool_system(
    processes: list[subprocess.Popen[bytes]], max_level: int
) -> bool:
    """Stop all seed processes while the thermal pressure level is above
    `max_level`, then resume them. Returns whether a pause occurred."""
    if not too_hot(max_level):
        return False
    level = thermal_pressure() or 0
    name = THERMAL_LEVELS[min(level, len(THERMAL_LEVELS) - 1)]
    print(f"thermal pressure {name}: pausing seed processes", flush=True)
    signal_all(processes, signal.SIGSTOP)
    try:
        while too_hot(max_level):
            time.sleep(THERMAL_POLL_SECONDS)
    finally:
        signal_all(processes, signal.SIGCONT)
    print("thermal pressure back at the limit: resuming", flush=True)
    return True


def duty_cycle_period(
    processes: list[subprocess.Popen[bytes]], duty_cycle: float, period: float
) -> float:
    """Let the processes compute for `duty_cycle * period` seconds, then stop
    them for the rest of the period. Returns the seconds they were stopped."""
    if duty_cycle >= 1.0 or not processes:
        time.sleep(min(period, 0.5))
        return 0.0
    time.sleep(period * duty_cycle)
    signal_all(processes, signal.SIGSTOP)
    stopped = time.monotonic()
    try:
        time.sleep(period * (1.0 - duty_cycle))
    finally:
        signal_all(processes, signal.SIGCONT)
    return time.monotonic() - stopped


def pending_seeds(seeds: list[int], output_root: Path, rerun: bool) -> list[int]:
    """The seeds to run: all with `rerun`, else those without a metrics file."""
    if rerun:
        return list(seeds)
    return [
        s for s in seeds if not (output_root / f"seed{s}" / "metrics.json").exists()
    ]


def stop_processes(processes: list[subprocess.Popen[bytes]]) -> None:
    """Resume, then terminate every process, so that none stays stopped."""
    signal_all(processes, signal.SIGCONT)
    signal_all(processes, signal.SIGTERM)
    deadline = time.monotonic() + TERMINATE_GRACE_SECONDS
    for process in processes:
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            signal_all([process], signal.SIGKILL)
            process.wait()


@dataclass
class Settings:
    jobs: int
    background: bool
    duty_cycle: float = 1.0
    period: float = 1.0
    max_thermal_level: int = 0
    cooldown_seconds: float = 0.0


@dataclass
class SeedRun:
    seed: int
    command: list[str]
    process: subprocess.Popen[bytes]
    log: IO[bytes]
    started: float
    thermal_at_start: int | None
    duty_cycle_pause_seconds: float = 0.0
    thermal_pause_seconds: float = 0.0
    thermal_pause_events: int = 0


def child_threads(metrics: Path) -> dict[str, Any]:
    """The thread counts that the finished experiment recorded."""
    try:
        recorded = json.loads(metrics.read_text(encoding="utf-8"))["runtime"]
    except (OSError, KeyError, ValueError):
        return {"intra_op": None, "inter_op": None, "canonical": None}
    return {
        "intra_op": recorded.get("intra_op_threads"),
        "inter_op": recorded.get("inter_op_threads"),
        "canonical": recorded.get("canonical"),
    }


def sweep_record(
    experiment: str, run: SeedRun, settings: Settings, directory: Path
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "seed": run.seed,
        "command": run.command,
        "exit_code": run.process.returncode,
        "jobs": settings.jobs,
        "duty_cycle": settings.duty_cycle,
        "period_seconds": settings.period,
        "background_qos": settings.background,
        "max_thermal_level": settings.max_thermal_level,
        "cooldown_seconds": settings.cooldown_seconds,
        "wall_seconds": time.monotonic() - run.started,
        "duty_cycle_pause_seconds": run.duty_cycle_pause_seconds,
        "thermal_pause_seconds": run.thermal_pause_seconds,
        "thermal_pause_events": run.thermal_pause_events,
        "thermal_pressure_at_start": run.thermal_at_start,
        "thermal_pressure_at_end": thermal_pressure(),
        "experiment_threads": child_threads(directory / "metrics.json"),
        "system": platform.system(),
    }


def sweep(
    experiment: str,
    seeds: list[int],
    output_root: Path,
    extra: list[str],
    settings: Settings,
) -> dict[int, int]:
    """Run every seed, at most `settings.jobs` at a time. Returns the exit
    code per seed."""
    pending = list(seeds)
    running: dict[int, SeedRun] = {}
    codes: dict[int, int] = {}
    try:
        while pending or running:
            processes = [run.process for run in running.values()]
            paused = time.monotonic()
            if wait_for_cool_system(processes, settings.max_thermal_level):
                for run in running.values():
                    run.thermal_pause_events += 1
                    run.thermal_pause_seconds += time.monotonic() - paused
            while pending and len(running) < settings.jobs:
                seed = pending.pop(0)
                directory = output_root / f"seed{seed}"
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "metrics.json").unlink(missing_ok=True)
                run_command = command(
                    experiment,
                    seed,
                    directory / "metrics.json",
                    extra,
                    settings.background,
                )
                log = (directory / "log.txt").open("wb")
                process = subprocess.Popen(
                    run_command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                running[seed] = SeedRun(
                    seed,
                    run_command,
                    process,
                    log,
                    time.monotonic(),
                    thermal_pressure(),
                )
            stopped = duty_cycle_period(
                [run.process for run in running.values()],
                settings.duty_cycle,
                settings.period,
            )
            for run in running.values():
                run.duty_cycle_pause_seconds += stopped
            for seed, run in list(running.items()):
                if run.process.poll() is None:
                    continue
                run.log.close()
                directory = output_root / f"seed{seed}"
                record = sweep_record(experiment, run, settings, directory)
                (directory / "sweep.json").write_text(
                    json.dumps(record, indent=2) + "\n", encoding="utf-8"
                )
                codes[seed] = run.process.returncode
                status = "done" if run.process.returncode == 0 else "FAILED"
                print(
                    f"seed {seed}: {status} (exit {run.process.returncode}, "
                    f"{record['wall_seconds']:.0f} s)",
                    flush=True,
                )
                del running[seed]
                if pending and settings.cooldown_seconds > 0:
                    time.sleep(settings.cooldown_seconds)
    finally:
        if running:
            print(f"stopping {len(running)} unfinished seed processes", flush=True)
            stop_processes([run.process for run in running.values()])
            for run in running.values():
                run.log.close()
    return codes


def raise_interrupt(number: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--duty-cycle",
        type=float,
        default=0.5,
        help="share of time in (0, 1] that the seed processes may compute",
    )
    parser.add_argument(
        "--period",
        type=float,
        default=1.0,
        help="length in seconds of one compute-and-pause period",
    )
    parser.add_argument(
        "--max-thermal-level",
        type=int,
        default=0,
        help="on macOS, pause seed processes above this thermal pressure level "
        "(0 nominal, 1 moderate, 2 heavy)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=0.0,
        help="seconds to pause after each finished seed",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="also run seeds whose metrics.json already exists",
    )
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="on macOS, run at normal priority instead of background QoS",
    )
    parser.add_argument("extra", nargs="*", help="arguments after -- go to the run")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if not 0.0 < args.duty_cycle <= 1.0:
        parser.error("--duty-cycle must be in (0, 1]")
    if args.period <= 0.0:
        parser.error("--period must be positive")
    settings = Settings(
        jobs=args.jobs,
        background=not args.foreground and TASKPOLICY.exists(),
        duty_cycle=args.duty_cycle,
        period=args.period,
        max_thermal_level=args.max_thermal_level,
        cooldown_seconds=args.cooldown,
    )
    requested = parse_seeds(args.seeds)
    seeds = pending_seeds(requested, args.output_root, args.rerun)
    if len(seeds) < len(requested):
        skipped = sorted(set(requested) - set(seeds))
        print(f"skipping seeds with existing metrics.json: {skipped}", flush=True)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, raise_interrupt)
    try:
        codes = sweep(args.experiment, seeds, args.output_root, args.extra, settings)
    except KeyboardInterrupt:
        raise SystemExit("sweep interrupted; run the same command to resume") from None
    failed = sorted(seed for seed, code in codes.items() if code != 0)
    if failed:
        raise SystemExit(f"failed seeds: {failed}")


if __name__ == "__main__":
    main()
