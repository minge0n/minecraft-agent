"""Measure the recorded-frames-only render mode on macOS
(docs/decisions/simulation-throughput.md, "Recorded frames only").

In unpaced lockstep the render thread normally draws as many frames as it
can (frame limit 260), although the recording keeps one frame every 2 ticks.
With `recorded_frames_only` the client draws a frame only when the recording
needs one. This probe runs, in one client process and with the game on the
efficiency cores when `--background` is given:

- the controlled replay scene twice, once drawing every frame and once in
  the new mode, and compares the traces: they must be identical;
- the fixed sprint walk with recording every 2 ticks (the training setting),
  once per mode, with the game's CPU time, energy, frames drawn and skipped,
  and the recording counters.

One unmeasured warm-up walk comes first. Then each mode is measured
`--repeats` times, interleaved.
"""

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path
from typing import Any

from minecraft_rl.minecraft_launch import launched_client
from minecraft_rl.provenance import git_commit
from minecraft_rl.recording import WorkerRecorder, write_json
from minecraft_rl.replay import (
    build_controlled_replay,
    compare_traces,
    play_replay,
    unexplained_fields,
)

RENDER_DISTANCE_PROBE = Path(__file__).with_name("minecraft-render-distance-probe.py")
WORKER = "env-0000"
WALK_SEED = 12345
RECORDING = {"every_ticks": 2, "width": 426, "height": 240, "queue_frames": 64}
MODES = {"every_frame": False, "recorded_frames_only": True}


def load_probe_helpers() -> dict[str, Any]:
    """The walk and the process-usage helpers of the render-distance probe."""
    spec = importlib.util.spec_from_file_location(
        "render_distance_probe", RENDER_DISTANCE_PROBE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return vars(module)


def controlled(client, seed: int, recorded_only: bool) -> list[dict[str, Any]]:
    scene = build_controlled_replay(client, seed)
    client.set_pacing("unpaced", recorded_frames_only=recorded_only)
    records = play_replay(client, scene)
    client.set_pacing("paced")
    return records


def walk(client, helpers, pid, recorder, recorded_only: bool) -> dict[str, Any]:
    client.reset(WALK_SEED, "normal")
    client.set_pacing("unpaced", recorded_frames_only=recorded_only)
    frames_before = client.status_json()["pacing"]
    recorder.start()
    actions = helpers["walk_actions"]()
    before = helpers["process_usage"](pid)
    started = time.perf_counter()
    for action in actions:
        result = client.step(action)
        if result.info.tick_after != result.info.tick_before + 1:
            raise AssertionError("a step did not advance exactly one tick")
    elapsed = time.perf_counter() - started
    usage = helpers["usage_delta"](before, helpers["process_usage"](pid))
    capture, recording = recorder.stop()
    frames_after = client.status_json()["pacing"]
    client.set_pacing("paced")
    steps = len(actions)
    out = {
        "mode": "recorded_frames_only" if recorded_only else "every_frame",
        "steps": steps,
        "steps_per_second": steps / elapsed,
        "frames_drawn": frames_after["frames_drawn"] - frames_before["frames_drawn"],
        "frames_skipped": frames_after["frames_skipped"]
        - frames_before["frames_skipped"],
        "capture": {
            key: capture[key]
            for key in (
                "due",
                "captured",
                "sent",
                "dropped_gpu_busy",
                "dropped_queue_full",
                "missed_unrendered",
            )
        },
        "video": recording.summary(),
        "game_process": usage,
    }
    if "cpu_seconds" in usage:
        out["game_cpu_ms_per_step"] = usage["cpu_seconds"] * 1000.0 / steps
        out["game_cpu_cores"] = usage["cpu_seconds"] / elapsed
    if "energy_joules" in usage:
        out["game_joules_per_step"] = usage["energy_joules"] / steps
        out["game_watts"] = usage["energy_joules"] / elapsed
    return out


def summary(walks: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for mode in MODES:
        runs = [w for w in walks if w["mode"] == mode]
        out[mode] = {
            key: statistics.fmean(r[key] for r in runs)
            for key in (
                "steps_per_second",
                "frames_drawn",
                "game_cpu_ms_per_step",
                "game_cpu_cores",
                "game_joules_per_step",
                "game_watts",
            )
            if all(key in r for r in runs)
        } | {
            "frames_drawn_per_step": statistics.fmean(
                r["frames_drawn"] / r["steps"] for r in runs
            ),
            "missed_or_dropped": sum(
                r["capture"]["missed_unrendered"]
                + r["capture"]["dropped_gpu_busy"]
                + r["capture"]["dropped_queue_full"]
                for r in runs
            ),
            "videos": [r["video"]["status"] for r in runs],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/minecraft-frames"))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--port", type=int, default=47150)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    helpers = load_probe_helpers()
    output = args.output / ("background" if args.background else "default")
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "tool": "minecraft-frame-probe",
        "commit": git_commit(),
        "background_policy": args.background,
        "recording": RECORDING,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    recorder = None
    try:
        with launched_client(
            args.port,
            args.seed,
            output / "workers" / WORKER / "minecraft.log",
            240.0,
            background=args.background,
        ) as client:
            pid = helpers["game_pid"](args.port)
            reference = controlled(client, args.seed, False)
            other = controlled(client, args.seed, True)
            comparison = compare_traces(reference, other)
            result["controlled"] = {
                "steps": len(reference),
                "policy_steps_differing": comparison["policy_observation"][
                    "differing_steps"
                ],
                "unexplained_fields": unexplained_fields(comparison),
            }
            recorder = WorkerRecorder(
                client,
                output,
                WORKER,
                RECORDING,
                {"run": str(output), "mode": "debug", "world_seed": WALK_SEED},
            )
            result["warm_up"] = walk(client, helpers, pid, recorder, False)
            walks = []
            for _ in range(args.repeats):
                for recorded_only in MODES.values():
                    walks.append(walk(client, helpers, pid, recorder, recorded_only))
                    print(json.dumps(walks[-1]), flush=True)
            result["walks"] = walks
            result["summary"] = summary(walks)
        result["passed"] = (
            result["controlled"]["policy_steps_differing"] == 0
            and not result["controlled"]["unexplained_fields"]
            and all(s["missed_or_dropped"] == 0 for s in result["summary"].values())
        )
    except Exception as error:
        result |= {"passed": False, "error": f"{type(error).__name__}: {error}"}
    finally:
        if recorder is not None:
            recorder.close()
    write_json(output / "metadata.json", result)
    print(
        json.dumps(
            {k: result.get(k) for k in ("controlled", "summary", "passed", "error")},
            indent=2,
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
