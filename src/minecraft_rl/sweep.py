"""Run one Stage 2 experiment over many seeds on a Mac with little heat.

This runner supports macOS only. Its settings change only when a seed
process computes, never what it computes. The canonical runtime uses one
thread per process and a seeded generator for every random draw
(docs/decisions/reproducibility.md), so the results stay the same.

The runner limits heat in four ways:

1. It runs one seed process at a time (`--jobs`, default 1).
2. Each process runs under `taskpolicy -b`. macOS then runs it on the
   efficiency cores at a low clock speed.
3. `--duty-cycle` (default 0.5) limits the share of each period of
   `--period` seconds in which the process computes. The runner stops the
   process with SIGSTOP for the rest of the period and resumes it with
   SIGCONT.
4. A thermostat reads the die temperature of the SoC once per period. If the
   temperature is above `--max-temperature`, or the macOS thermal pressure is
   above `--max-thermal-level`, the runner stops all seed processes. It
   resumes them when the temperature is at or below `--resume-temperature`
   and the pressure is back at the limit, or after `--max-pause` seconds.

A seed whose `metrics.json` exists is skipped, so the same command resumes an
interrupted sweep. `--rerun` disables this. On SIGINT or SIGTERM, the runner
resumes and then terminates every unfinished seed process. Next to each
`metrics.json`, it writes `sweep.json` with the settings, the measured pauses
and the measured temperatures.

Example:

    .venv/bin/python -m minecraft_rl.sweep dreamer_loop --seeds 0-9 \\
        --output-root runs/stage2f -- --world-model rssm
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import IO, Any

from minecraft_rl.macos_thermal import THERMAL_LEVELS, CpuTemperature, thermal_pressure

TASKPOLICY = Path("/usr/sbin/taskpolicy")
THERMOSTAT_POLL_SECONDS = 2.0
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


def command(experiment: str, seed: int, output: Path, extra: list[str]) -> list[str]:
    return [
        str(TASKPOLICY),
        "-b",
        sys.executable,
        "-m",
        f"minecraft_rl.{experiment}",
        "--seed",
        str(seed),
        "--output",
        str(output),
        *extra,
    ]


def signal_all(processes: list[subprocess.Popen[bytes]], number: int) -> None:
    """Send a signal to the process group of every process that still runs.
    Each seed process leads its own group, so the signal also reaches the
    program that `taskpolicy` started. A process can exit between the check
    and the signal. macOS then reports ESRCH, or EPERM while the exited
    process is not yet reaped, and both are ignored."""
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, number)
            except (ProcessLookupError, PermissionError):
                pass


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


def pending_seeds(seeds: list[int], output_root: Path, rerun: bool) -> list[int]:
    """The seeds to run: all with `rerun`, else those without a metrics file."""
    if rerun:
        return list(seeds)
    return [
        s for s in seeds if not (output_root / f"seed{s}" / "metrics.json").exists()
    ]


@dataclass
class Settings:
    jobs: int = 1
    duty_cycle: float = 0.5
    period: float = 1.0
    max_temperature: float = 65.0
    resume_temperature: float = 62.0
    max_thermal_level: int = 0
    cooldown_seconds: float = 0.0
    max_pause_seconds: float = 120.0


@dataclass
class Reading:
    temperature: float | None
    pressure: int | None


class Thermostat:
    """Decides from the die temperature and the thermal pressure level whether
    the seed processes must stop. The readers can be replaced in tests."""

    def __init__(
        self,
        settings: Settings,
        temperature: Callable[[], float | None] | None = None,
        pressure: Callable[[], int | None] = thermal_pressure,
    ) -> None:
        self.settings = settings
        self._temperature = temperature or CpuTemperature().read
        self._pressure = pressure

    def read(self) -> Reading:
        return Reading(self._temperature(), self._pressure())

    def too_hot(self, reading: Reading) -> bool:
        return (
            reading.temperature is not None
            and reading.temperature > self.settings.max_temperature
        ) or (
            reading.pressure is not None
            and reading.pressure > self.settings.max_thermal_level
        )

    def cool_enough(self, reading: Reading) -> bool:
        return (
            reading.temperature is None
            or reading.temperature <= self.settings.resume_temperature
        ) and (
            reading.pressure is None
            or reading.pressure <= self.settings.max_thermal_level
        )


def describe(reading: Reading) -> str:
    temperature = (
        "unknown" if reading.temperature is None else f"{reading.temperature:.1f} C"
    )
    level = reading.pressure
    pressure = (
        "unknown"
        if level is None
        else THERMAL_LEVELS[min(level, len(THERMAL_LEVELS) - 1)]
    )
    return f"die temperature {temperature}, thermal pressure {pressure}"


@dataclass
class SeedRun:
    seed: int
    command: list[str]
    process: subprocess.Popen[bytes]
    log: IO[bytes]
    started: float
    pressure_at_start: int | None
    duty_cycle_pause_seconds: float = 0.0
    thermostat_pause_seconds: float = 0.0
    thermostat_pause_events: int = 0
    temperatures: list[float] = field(default_factory=list)


def cool_down(
    processes: list[subprocess.Popen[bytes]],
    thermostat: Thermostat,
    reading: Reading,
) -> float:
    """Stop the processes until the thermostat reports cool enough, then
    resume them. Returns the seconds that the pause took.

    The pause ends after `max_pause_seconds` even if the chip is still warm.
    Other programs or the room can keep the chip above the resume
    temperature, and the sweep must not wait forever."""
    print(f"{describe(reading)}: pausing seed processes", flush=True)
    started = time.monotonic()
    limit = thermostat.settings.max_pause_seconds
    signal_all(processes, signal.SIGSTOP)
    try:
        while not thermostat.cool_enough(reading := thermostat.read()):
            if time.monotonic() - started >= limit:
                print(f"pause reached its limit of {limit:.0f} s", flush=True)
                break
            time.sleep(THERMOSTAT_POLL_SECONDS)
    finally:
        signal_all(processes, signal.SIGCONT)
    paused = time.monotonic() - started
    print(f"{describe(reading)}: resuming after {paused:.0f} s", flush=True)
    return paused


def throttle_period(
    runs: list[SeedRun], settings: Settings, thermostat: Thermostat
) -> None:
    """One period: compute for `duty_cycle * period` seconds, then read the
    thermostat. Too hot: pause until cool. Otherwise stop the processes for
    the rest of the period."""
    processes = [run.process for run in runs]
    time.sleep(settings.period * settings.duty_cycle)
    reading = thermostat.read()
    if reading.temperature is not None:
        for run in runs:
            run.temperatures.append(reading.temperature)
    if thermostat.too_hot(reading):
        paused = cool_down(processes, thermostat, reading)
        for run in runs:
            run.thermostat_pause_seconds += paused
            run.thermostat_pause_events += 1
        return
    if settings.duty_cycle >= 1.0:
        return
    signal_all(processes, signal.SIGSTOP)
    stopped = time.monotonic()
    try:
        time.sleep(settings.period * (1.0 - settings.duty_cycle))
    finally:
        signal_all(processes, signal.SIGCONT)
    for run in runs:
        run.duty_cycle_pause_seconds += time.monotonic() - stopped


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
    temperatures = run.temperatures
    return {
        "experiment": experiment,
        "seed": run.seed,
        "command": run.command,
        "exit_code": run.process.returncode,
        "jobs": settings.jobs,
        "duty_cycle": settings.duty_cycle,
        "period_seconds": settings.period,
        "background_qos": True,
        "max_temperature_celsius": settings.max_temperature,
        "resume_temperature_celsius": settings.resume_temperature,
        "max_thermal_level": settings.max_thermal_level,
        "cooldown_seconds": settings.cooldown_seconds,
        "wall_seconds": time.monotonic() - run.started,
        "duty_cycle_pause_seconds": run.duty_cycle_pause_seconds,
        "thermostat_pause_seconds": run.thermostat_pause_seconds,
        "thermostat_pause_events": run.thermostat_pause_events,
        "temperature_samples": len(temperatures),
        "temperature_mean_celsius": sum(temperatures) / len(temperatures)
        if temperatures
        else None,
        "temperature_max_celsius": max(temperatures) if temperatures else None,
        "thermal_pressure_at_start": run.pressure_at_start,
        "thermal_pressure_at_end": thermal_pressure(),
        "experiment_threads": child_threads(directory / "metrics.json"),
    }


def start_seed(
    experiment: str, seed: int, output_root: Path, extra: list[str]
) -> SeedRun:
    directory = output_root / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metrics.json").unlink(missing_ok=True)
    run_command = command(experiment, seed, directory / "metrics.json", extra)
    log = (directory / "log.txt").open("wb")
    process = subprocess.Popen(
        run_command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    return SeedRun(
        seed, run_command, process, log, time.monotonic(), thermal_pressure()
    )


def sweep(
    experiment: str,
    seeds: list[int],
    output_root: Path,
    extra: list[str],
    settings: Settings,
    thermostat: Thermostat | None = None,
) -> dict[int, int]:
    """Run every seed, at most `settings.jobs` at a time. Returns the exit
    code per seed."""
    thermostat = thermostat or Thermostat(settings)
    pending = list(seeds)
    running: dict[int, SeedRun] = {}
    codes: dict[int, int] = {}
    try:
        while pending or running:
            if pending and len(running) < settings.jobs:
                reading = thermostat.read()
                if not running and thermostat.too_hot(reading):
                    cool_down([], thermostat, reading)
            while pending and len(running) < settings.jobs:
                seed = pending.pop(0)
                running[seed] = start_seed(experiment, seed, output_root, extra)
            throttle_period(list(running.values()), settings, thermostat)
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
                mean = record["temperature_mean_celsius"]
                print(
                    f"seed {seed}: {status} (exit {run.process.returncode}, "
                    f"{record['wall_seconds']:.0f} s, mean die temperature "
                    f"{'unknown' if mean is None else f'{mean:.1f} C'})",
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
    defaults = Settings()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--jobs", type=int, default=defaults.jobs)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--duty-cycle",
        type=float,
        default=defaults.duty_cycle,
        help="share of time in (0, 1] that the seed processes may compute",
    )
    parser.add_argument(
        "--period",
        type=float,
        default=defaults.period,
        help="length in seconds of one compute-and-pause period",
    )
    parser.add_argument(
        "--max-temperature",
        type=float,
        default=defaults.max_temperature,
        help="pause the seed processes above this die temperature (Celsius)",
    )
    parser.add_argument(
        "--resume-temperature",
        type=float,
        help="resume at or below this die temperature (default: 3 C below "
        "--max-temperature)",
    )
    parser.add_argument(
        "--max-thermal-level",
        type=int,
        default=defaults.max_thermal_level,
        help="pause the seed processes above this macOS thermal pressure level "
        "(0 nominal, 1 moderate, 2 heavy)",
    )
    parser.add_argument(
        "--max-pause",
        type=float,
        default=defaults.max_pause_seconds,
        help="longest thermostat pause in seconds before the processes resume",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=defaults.cooldown_seconds,
        help="seconds to pause after each finished seed",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="also run seeds whose metrics.json already exists",
    )
    parser.add_argument("extra", nargs="*", help="arguments after -- go to the run")
    args = parser.parse_args()
    if not TASKPOLICY.exists():
        parser.error(f"{TASKPOLICY} not found; this runner supports macOS only")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if not 0.0 < args.duty_cycle <= 1.0:
        parser.error("--duty-cycle must be in (0, 1]")
    if args.period <= 0.0:
        parser.error("--period must be positive")
    resume = (
        args.max_temperature - 3.0
        if args.resume_temperature is None
        else args.resume_temperature
    )
    if resume > args.max_temperature:
        parser.error("--resume-temperature must not exceed --max-temperature")
    settings = Settings(
        jobs=args.jobs,
        duty_cycle=args.duty_cycle,
        period=args.period,
        max_temperature=args.max_temperature,
        resume_temperature=resume,
        max_thermal_level=args.max_thermal_level,
        cooldown_seconds=args.cooldown,
        max_pause_seconds=args.max_pause,
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
