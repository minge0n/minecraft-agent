"""Benchmark the client render distance (docs/decisions/simulation-throughput.md).

The policy observation is built on the integrated server, which takes its view
distance, and so the set of loaded chunks, from the client render distance. A
lower render distance renders less, but it must not change what the sensor sees
within its 32-block range or what the server simulates. One invocation measures
one render distance in a fresh client process:

- `controlled`: the controlled replay scene in a flat world, unpaced. Its trace is
  exact under identical actions (docs/replay-characterization.md), so traces of
  different render distances must be equal.
- `walk`: a fixed sprint walk in a `normal` world, unpaced, run twice. After every
  step the probe reads which chunks the server has within sensor range and within
  the simulation distance. The ray grids are not compared: two resets of the same
  `normal` seed already differ at the first step, because the spawn point varies.
- `throughput`: the same walk with rendering, once without and once with
  recording, timed. CPU time, energy and memory of the game process are read
  from the operating system.

`--compare` reads the result of every render distance and checks them against the
reference distance. Recorded sessions are written under the run directory.
"""

import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from minecraft_rl import macos_energy
from minecraft_rl.minecraft_client import MinecraftClient
from minecraft_rl.minecraft_interface import PlayerAction
from minecraft_rl.minecraft_launch import launched_client
from minecraft_rl.provenance import git_commit
from minecraft_rl.recording import WorkerRecorder, write_json
from minecraft_rl.replay import (
    build_controlled_replay,
    compare_traces,
    play_replay,
    unexplained_fields,
)

WORKER = "env-0000"
WALK_SEED = 12345
RECORDING = {"every_ticks": 1, "width": 426, "height": 240, "queue_frames": 64}


def walk_actions() -> list[PlayerAction]:
    """A fixed open-loop walk: sprint forward and turn now and then, so the player
    crosses several chunk borders and the ray grid keeps changing."""
    actions = [PlayerAction()] * 20
    for index in range(400):
        turn = 30.0 if index % 100 == 50 else 0.0
        jump = index % 25 == 0
        actions.append(
            PlayerAction(forward=True, sprint=True, jump=jump, yaw_delta=turn)
        )
    return actions


def game_pid(port: int) -> int | None:
    """The process that listens on the lockstep port: the game itself, not Gradle."""
    found = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    return int(found[0]) if found else None


def process_usage(pid: int | None) -> dict[str, Any]:
    """CPU seconds and resident memory from `ps`, energy from `proc_pid_rusage`."""
    if pid is None:
        return {}
    fields = subprocess.run(
        ["ps", "-o", "cputime=,rss=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    out: dict[str, Any] = {}
    if len(fields) == 2:
        minutes, seconds = fields[0].split(":")
        out["cpu_seconds"] = int(minutes) * 60 + float(seconds)
        out["resident_mib"] = int(fields[1]) / 1024
    energy = macos_energy.read(pid)
    if energy is not None:
        out["energy_joules"] = energy.energy_joules
    return out


def usage_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    out = {
        key: after[key] - before[key]
        for key in ("cpu_seconds", "energy_joules")
        if key in before and key in after
    }
    if "resident_mib" in after:
        out["resident_mib_after"] = after["resident_mib"]
    return out


def run_controlled(client: MinecraftClient, seed: int) -> list[dict[str, Any]]:
    scene = build_controlled_replay(client, seed)
    client.set_pacing("unpaced")
    records = play_replay(client, scene)
    client.set_pacing("paced")
    return records


def run_walk(client: MinecraftClient, seed: int) -> dict[str, Any]:
    """The walk with a chunk query after every step. Not timed."""
    client.reset(seed, "normal")
    client.set_pacing("unpaced")
    privileged = client.privileged()
    start = privileged.chunks()
    chunk_series = []
    for action in walk_actions():
        result = client.step(action)
        if result.info.tick_after != result.info.tick_before + 1:
            raise AssertionError("a step did not advance exactly one tick")
        chunks = privileged.chunks()
        chunk_series.append(
            (
                chunks["sensor_chunks"],
                chunks["sensor_chunks_loaded"],
                chunks["sensor_chunks_entity_ticking"],
                chunks["simulation_chunks"],
                chunks["simulation_chunks_entity_ticking"],
                chunks["server_loaded_chunks"],
            )
        )
    client.set_pacing("paced")
    series = np.array(chunk_series)
    return {
        "start": start,
        "end": privileged.chunks(),
        "steps": len(series),
        "steps_with_unloaded_sensor_chunk": int((series[:, 1] < series[:, 0]).sum()),
        "steps_with_non_ticking_sensor_chunk": int((series[:, 2] < series[:, 0]).sum()),
        "steps_with_non_ticking_simulation_chunk": int(
            (series[:, 4] < series[:, 3]).sum()
        ),
        "simulation_ticking_fraction_min": float((series[:, 4] / series[:, 3]).min()),
        "server_loaded_chunks_mean": float(series[:, 5].mean()),
    }


def run_throughput(
    client: MinecraftClient,
    seed: int,
    pid: int | None,
    recorder: WorkerRecorder | None,
) -> dict[str, Any]:
    """The walk with rendering and no queries, timed from the first step."""
    client.reset(seed, "normal")
    client.set_pacing("unpaced", render_frames=True)
    if recorder is not None:
        recorder.start()
    actions = walk_actions()
    step_ms = []
    before = process_usage(pid)
    started = time.perf_counter()
    for action in actions:
        step_started = time.perf_counter()
        client.step(action)
        step_ms.append((time.perf_counter() - step_started) * 1000.0)
    elapsed = time.perf_counter() - started
    usage = usage_delta(before, process_usage(pid))
    out: dict[str, Any] = {
        "recording": recorder is not None,
        "steps": len(actions),
        "steps_per_second": len(actions) / elapsed,
        "median_step_ms": statistics.median(step_ms),
        "p95_step_ms": statistics.quantiles(step_ms, n=20)[-1],
        "game_process": usage,
    }
    if "cpu_seconds" in usage:
        out["game_cpu_ms_per_step"] = usage["cpu_seconds"] * 1000.0 / len(actions)
        out["game_cpu_cores"] = usage["cpu_seconds"] / elapsed
    if "energy_joules" in usage:
        out["game_joules_per_step"] = usage["energy_joules"] / len(actions)
    if recorder is not None:
        capture, recording = recorder.stop()
        out["capture"] = capture
        out["video"] = recording.summary()
    client.set_pacing("paced")
    return out


def measure(args: argparse.Namespace) -> None:
    output = args.output / f"render-{args.render_distance}"
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "tool": "minecraft-render-distance-probe",
        "commit": git_commit(),
        "render_distance": args.render_distance,
        "seed": args.seed,
        "walk_seed": WALK_SEED,
        "host": {"system": platform.system(), "machine": platform.machine()},
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    recorder: WorkerRecorder | None = None
    launched = time.perf_counter()
    try:
        with launched_client(
            args.port,
            args.seed,
            output / "workers" / WORKER / "minecraft.log",
            args.startup_timeout,
            extra_environment={"MCBOT_RENDER_DISTANCE": str(args.render_distance)},
        ) as client:
            result["startup_seconds"] = time.perf_counter() - launched
            status = client.status_json()
            result["client_settings"] = status["client_settings"]
            settings = status["client_settings"]["render_distance_chunks"]
            if settings != args.render_distance:
                raise AssertionError(f"client render distance is {settings}")
            pid = game_pid(args.port)
            result["game_pid_found"] = pid is not None
            controlled = run_controlled(client, args.seed)
            (output / "controlled.json").write_text(json.dumps(controlled))
            result["walks"] = [run_walk(client, WALK_SEED) for _ in range(2)]
            recorder = WorkerRecorder(
                client,
                output,
                WORKER,
                RECORDING,
                {
                    "run": str(output),
                    "mode": "debug",
                    "world_seed": WALK_SEED,
                    "commit": result["commit"],
                    "checkpoint": None,
                    "minecraft_version": "26.3",
                    "render_distance": args.render_distance,
                },
            )
            result["throughput"] = [
                run_throughput(client, WALK_SEED, pid, None),
                run_throughput(client, WALK_SEED, pid, recorder),
            ]
            result["game_process_end"] = process_usage(pid)
        result["passed"] = True
    except Exception as error:
        result |= {"passed": False, "error": f"{type(error).__name__}: {error}"}
    finally:
        if recorder is not None:
            recorder.close()
    result["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(output / "metadata.json", result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


def compare(args: argparse.Namespace) -> None:
    distances = sorted(
        int(path.name.split("-")[1]) for path in args.output.glob("render-*")
    )
    reference = args.reference
    if reference not in distances:
        raise SystemExit(f"no result for the reference render distance {reference}")

    def load(distance: int) -> dict[str, Any]:
        directory = args.output / f"render-{distance}"
        return {
            "metadata": json.loads((directory / "metadata.json").read_text()),
            "controlled": json.loads((directory / "controlled.json").read_text()),
        }

    runs = {distance: load(distance) for distance in distances}
    out: dict[str, Any] = {"reference": reference, "distances": {}}
    base = runs[reference]
    for distance, run in runs.items():
        comparison = compare_traces(base["controlled"], run["controlled"])
        unexplained = unexplained_fields(comparison)
        out["distances"][str(distance)] = {
            "controlled_policy_steps_differing": comparison["policy_observation"][
                "differing_steps"
            ],
            "controlled_unexplained_fields": unexplained,
            "walk_chunks": [
                {
                    key: walk[key]
                    for key in (
                        "steps_with_unloaded_sensor_chunk",
                        "steps_with_non_ticking_sensor_chunk",
                        "steps_with_non_ticking_simulation_chunk",
                    )
                }
                | {
                    "simulation_chunks_ticking_at_end": walk["end"][
                        "simulation_chunks_entity_ticking"
                    ],
                    "simulation_chunks": walk["end"]["simulation_chunks"],
                }
                for walk in run["metadata"]["walks"]
            ],
            "passed": run["metadata"]["passed"],
        }
    write_json(args.output / "comparison.json", out)
    print(json.dumps(out, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-distance", type=int)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--reference", type=int, default=8)
    parser.add_argument("--port", type=int, default=47130)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--startup-timeout", type=float, default=240.0)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/minecraft-render-distance")
    )
    args = parser.parse_args()
    if args.compare:
        compare(args)
    elif args.render_distance is None:
        parser.error("give --render-distance or --compare")
    else:
        measure(args)


if __name__ == "__main__":
    main()
