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

The runner reads the CPU energy counter of each seed process
(`macos_energy.py`). A pause can only remove the heat of the seed processes.
If the die is above the limit but the seed processes use less than
`--min-pause-power` watts together, other programs make the heat, and a
pause would only make the run longer. The runner then does not pause and
counts the event. A thermal pressure above the limit always pauses.

A seed whose `metrics.json` exists is skipped, so the same command resumes an
interrupted sweep. `--rerun` disables this. With `--time-budget`, the sweep
starts no new seed after that many seconds. A resumable experiment (imagination,
actor-critic, Dreamer loop) then saves its state at its next boundary and exits.
The same command continues it, with results identical to a run in one process.
On SIGINT or SIGTERM, the runner
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

from minecraft_rl import macos_energy
from minecraft_rl.macos_thermal import THERMAL_LEVELS, CpuTemperature, thermal_pressure
from minecraft_rl.resumable import INCOMPLETE_EXIT_CODE

TASKPOLICY = Path("/usr/sbin/taskpolicy")
THERMOSTAT_POLL_SECONDS = 2.0
TERMINATE_GRACE_SECONDS = 10.0
STATE_FILE = "state.pt"
RESUMABLE = (
    "imagination",
    "actor_critic",
    "dreamer_loop",
    "minecraft_world_model_train",
)
EXPERIMENTS = (
    "parity",
    "cue_recall",
    "world_model",
    "imagination",
    "actor_critic",
    "dreamer_loop",
    "stochastic_world",
    "minecraft_world_model_train",
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
    min_pause_watts: float = 2.0
    time_budget: float | None = None


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

    def pressure_too_high(self, reading: Reading) -> bool:
        return (
            reading.pressure is not None
            and reading.pressure > self.settings.max_thermal_level
        )

    def too_hot(self, reading: Reading) -> bool:
        return (
            reading.temperature is not None
            and reading.temperature > self.settings.max_temperature
        ) or self.pressure_too_high(reading)

    def pause_helps(self, reading: Reading, seed_watts: float | None) -> bool:
        """A pause on temperature alone helps only if the seed processes make
        a real part of the heat. Unknown power counts as a real part."""
        return (
            self.pressure_too_high(reading)
            or seed_watts is None
            or seed_watts >= self.settings.min_pause_watts
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
    energy: macos_energy.Counters | None = None
    energy_time: float | None = None
    recent_watts: float | None = None
    skipped_pause_events: int = 0

    def sample_energy(self) -> None:
        """Keep the last CPU energy counters of the process, and its mean
        power since the previous sample. macOS drops the counters when the
        process exits, so the runner reads them every period. `taskpolicy`
        replaces itself with the experiment, so the process ID stays the
        same."""
        counters = macos_energy.read(self.process.pid)
        if counters is None:
            return
        now = time.monotonic()
        if self.energy is not None and self.energy_time is not None:
            seconds = now - self.energy_time
            if seconds > 0:
                self.recent_watts = (counters - self.energy).energy_joules / seconds
        self.energy, self.energy_time = counters, now


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
    for run in runs:
        run.sample_energy()
    reading = thermostat.read()
    if reading.temperature is not None:
        for run in runs:
            run.temperatures.append(reading.temperature)
    if thermostat.too_hot(reading):
        watts = [run.recent_watts for run in runs]
        seed_watts = None if None in watts else sum(watts)
        if thermostat.pause_helps(reading, seed_watts):
            paused = cool_down(processes, thermostat, reading)
            for run in runs:
                run.thermostat_pause_seconds += paused
                run.thermostat_pause_events += 1
            return
        for run in runs:
            run.skipped_pause_events += 1
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
    energy = run.energy
    compute_seconds = (
        time.monotonic()
        - run.started
        - run.duty_cycle_pause_seconds
        - run.thermostat_pause_seconds
    )
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
        "min_pause_watts": settings.min_pause_watts,
        "skipped_pause_events": run.skipped_pause_events,
        "temperature_samples": len(temperatures),
        "temperature_mean_celsius": sum(temperatures) / len(temperatures)
        if temperatures
        else None,
        "temperature_max_celsius": max(temperatures) if temperatures else None,
        "cpu_energy_joules": None if energy is None else energy.energy_joules,
        "performance_core_energy_joules": (
            None if energy is None else energy.performance_core_energy_joules
        ),
        "mean_cpu_power_watts_while_computing": (
            None
            if energy is None or compute_seconds <= 0
            else energy.energy_joules / compute_seconds
        ),
        "thermal_pressure_at_start": run.pressure_at_start,
        "thermal_pressure_at_end": thermal_pressure(),
        "experiment_threads": child_threads(directory / "metrics.json"),
    }


def start_seed(
    experiment: str,
    seed: int,
    output_root: Path,
    extra: list[str],
    stop_after: float | None,
) -> SeedRun:
    """Start one seed process. With `stop_after`, a resumable experiment gets
    its state file and stops at the first boundary after that many seconds."""
    directory = output_root / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metrics.json").unlink(missing_ok=True)
    resume = []
    if experiment in RESUMABLE:
        resume = ["--state", str(directory / STATE_FILE)]
        if stop_after is not None:
            resume += ["--stop-after", f"{max(0.0, stop_after):.1f}"]
    run_command = command(
        experiment, seed, directory / "metrics.json", [*extra, *resume]
    )
    log = (directory / "log.txt").open("ab")
    process = subprocess.Popen(
        run_command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    return SeedRun(
        seed, run_command, process, log, time.monotonic(), thermal_pressure()
    )


def write_record(directory: Path, record: dict[str, Any], complete: bool) -> None:
    """Write `sweep.json`. A seed that ran in several processes keeps one
    record per process in `processes`. The top-level fields describe the last
    process, and the totals cover all of them."""
    path = directory / "sweep.json"
    earlier: list[dict[str, Any]] = []
    if (directory / STATE_FILE).exists() or complete:
        try:
            earlier = json.loads(path.read_text(encoding="utf-8")).get("processes", [])
        except (OSError, ValueError):
            earlier = []
        if earlier and earlier[-1].get("complete"):
            earlier = []
    processes = [*earlier, record | {"complete": complete}]
    total = record | {
        "complete": complete,
        "processes": processes,
        "total_wall_seconds": sum(p["wall_seconds"] for p in processes),
        "total_thermostat_pause_seconds": sum(
            p["thermostat_pause_seconds"] for p in processes
        ),
        "total_thermostat_pause_events": sum(
            p["thermostat_pause_events"] for p in processes
        ),
        "total_cpu_energy_joules": sum(
            p.get("cpu_energy_joules") or 0.0 for p in processes
        ),
    }
    path.write_text(json.dumps(total, indent=2) + "\n", encoding="utf-8")


def sweep(
    experiment: str,
    seeds: list[int],
    output_root: Path,
    extra: list[str],
    settings: Settings,
    thermostat: Thermostat | None = None,
) -> dict[int, int]:
    """Run every seed, at most `settings.jobs` at a time. Returns the exit
    code per seed.

    With `settings.time_budget`, the sweep starts no seed after the budget
    passes, and a resumable seed saves its state and stops at its first
    boundary after the budget. The same command continues it later."""
    thermostat = thermostat or Thermostat(settings)
    started = time.monotonic()
    budget = settings.time_budget
    pending = list(seeds)
    running: dict[int, SeedRun] = {}
    codes: dict[int, int] = {}
    try:
        while pending or running:
            remaining = (
                None if budget is None else budget - (time.monotonic() - started)
            )
            if remaining is not None and remaining <= 0 and not running:
                print(
                    f"time budget used; seeds left for the next call: {pending}",
                    flush=True,
                )
                break
            if (
                pending
                and len(running) < settings.jobs
                and (remaining is None or remaining > 0)
            ):
                reading = thermostat.read()
                if not running and thermostat.too_hot(reading):
                    cool_down([], thermostat, reading)
                while pending and len(running) < settings.jobs:
                    seed = pending.pop(0)
                    stop_after = (
                        None
                        if budget is None
                        else budget - (time.monotonic() - started)
                    )
                    running[seed] = start_seed(
                        experiment, seed, output_root, extra, stop_after
                    )
            throttle_period(list(running.values()), settings, thermostat)
            for seed, run in list(running.items()):
                if run.process.poll() is None:
                    continue
                run.log.close()
                directory = output_root / f"seed{seed}"
                record = sweep_record(experiment, run, settings, directory)
                code = run.process.returncode
                paused = code == INCOMPLETE_EXIT_CODE
                write_record(directory, record, complete=code == 0)
                del running[seed]
                mean = record["temperature_mean_celsius"]
                status = (
                    "state saved" if paused else ("done" if code == 0 else "FAILED")
                )
                print(
                    f"seed {seed}: {status} (exit {code}, "
                    f"{record['wall_seconds']:.0f} s, mean die temperature "
                    f"{'unknown' if mean is None else f'{mean:.1f} C'})",
                    flush=True,
                )
                if paused:
                    pending.insert(0, seed)
                    budget = 0.0
                    continue
                codes[seed] = code
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
        "--min-pause-power",
        type=float,
        default=defaults.min_pause_watts,
        help="pause on die temperature only if the seed processes use at "
        "least this many watts together; a thermal pressure above the limit "
        "always pauses",
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
    parser.add_argument(
        "--time-budget",
        type=float,
        help="seconds after which the sweep starts no new seed; a resumable "
        f"experiment ({', '.join(RESUMABLE)}) saves its state at its next "
        "boundary and the same command continues it",
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
        min_pause_watts=args.min_pause_power,
        time_budget=args.time_budget,
    )
    requested = parse_seeds(args.seeds)
    seeds = pending_seeds(requested, args.output_root, args.rerun)
    if args.rerun:
        for seed in seeds:
            (args.output_root / f"seed{seed}" / STATE_FILE).unlink(missing_ok=True)
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
    left = sorted(set(seeds) - set(codes))
    if left:
        raise SystemExit(f"unfinished seeds, run the same command again: {left}")


if __name__ == "__main__":
    main()
