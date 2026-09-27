"""Characterize Minecraft replay reproducibility under identical inputs.

Launches one or more development clients. Each run resets to a fresh world from the
same seed, builds the deterministic `replay` debug scene, and plays the same
open-loop action script while recording a per-step trace. Runs are then compared
pairwise within one process and across processes. The tool measures what is and is
not reproducible; it does not try to make Minecraft deterministic.
"""

import argparse
import itertools
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from minecraft_rl.minecraft_client import MinecraftClient
from minecraft_rl.minecraft_launch import REPOSITORY_ROOT, launched_client
from minecraft_rl.replay import (
    REMOVE_SAND_SUPPORT,
    client_view_alignment,
    compare_traces,
    scripted_actions,
    scripted_events,
    trace_record,
)

SCENE_COLUMN = (8, 8)
# natural: vanilla world randomness plus an AI mob. controlled: no AI mob, random
# block ticks and block drops disabled, to attribute divergence to those sources.
MODES = {"natural": {"ai": True}, "controlled": {"controlled": True}}


def run_replay(client: MinecraftClient, seed: int, mode: str) -> dict[str, Any]:
    _, reset = client.reset(seed, "flat")
    privileged = client.privileged()
    scene = privileged.scene("replay", at=SCENE_COLUMN, **MODES[mode])
    roles = {scene["husk"]: "husk"}
    if "pig" in scene:
        roles[scene["pig"]] = "pig"
    region = (scene["arena_from"], scene["arena_to"])
    events = scripted_events()
    records = []
    started = time.perf_counter()
    for step, action in enumerate(scripted_actions()):
        event = events.get(step)
        if event == REMOVE_SAND_SUPPORT:
            support = scene["sand_support"]
            privileged.fill(support, support, "minecraft:air")
        result = client.step(action)
        trace = privileged.trace(*region)
        records.append(trace_record(step, action, event, result, trace, roles))
    elapsed = time.perf_counter() - started
    return {
        "reset": {"level_id": reset.level_id, "reset_ms": reset.reset_ms},
        "scene": scene,
        "wall_seconds": elapsed,
        "records": records,
    }


def git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return completed.stdout.strip() + ("-dirty" if dirty else "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=47125)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--processes", type=int, default=2)
    parser.add_argument("--runs-per-process", type=int, default=3)
    parser.add_argument("--startup-timeout", type=float, default=240.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/minecraft-replay") / time.strftime("%Y%m%d-%H%M%S"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "tool": "minecraft-replay-probe",
        "git_commit": git_commit(),
        "seed": args.seed,
        "world_preset": "flat",
        "scene": "replay",
        "scene_column": SCENE_COLUMN,
        "steps": len(scripted_actions()),
        "events": {str(step): name for step, name in scripted_events().items()},
        "processes": args.processes,
        "runs_per_process": args.runs_per_process,
        "modes": MODES,
        "host": {"system": platform.system(), "machine": platform.machine()},
        "recording": "non-recording infrastructure test",
    }
    traces: dict[str, list[dict[str, Any]]] = {}
    runs: dict[str, Any] = {}
    try:
        for process in range(args.processes):
            log_path = args.output / f"process-{process}.minecraft.log"
            with launched_client(
                args.port, args.seed, log_path, args.startup_timeout
            ) as client:
                metadata.setdefault("schema", client.schema().__dict__)
                for run in range(args.runs_per_process):
                    for mode in MODES:
                        label = f"{mode}-p{process}r{run}"
                        replay = run_replay(client, args.seed, mode)
                        traces[label] = replay.pop("records")
                        runs[label] = replay | {
                            "client_view_alignment": client_view_alignment(
                                traces[label]
                            )
                        }
                        trace_path = args.output / f"trace-{label}.jsonl"
                        with trace_path.open("w", encoding="utf-8") as trace_file:
                            for record in traces[label]:
                                trace_file.write(json.dumps(record) + "\n")
        metadata["passed"] = True
    except Exception as error:
        metadata |= {"passed": False, "error": f"{type(error).__name__}: {error}"}

    comparisons = {}
    for a, b in itertools.combinations(sorted(traces), 2):
        mode_a, run_a = a.split("-")
        mode_b, run_b = b.split("-")
        if mode_a != mode_b:
            continue
        kind = "same_process" if run_a[:2] == run_b[:2] else "cross_process"
        comparisons[f"{a}-vs-{run_b}"] = {"mode": mode_a, "kind": kind} | (
            compare_traces(traces[a], traces[b])
        )
    result = metadata | {"runs": runs, "comparisons": comparisons}
    (args.output / "comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        name: {
            "kind": comparison["kind"],
            "identical": comparison["identical"],
            "first_difference_step": comparison["first_difference_step"],
            "first_divergence_by_field": dict(
                itertools.islice(comparison["first_divergence_by_field"].items(), 6)
            ),
            "observation_differing_steps": comparison["policy_observation"][
                "differing_steps"
            ],
            "player_max_distance": comparison["player_position"]["max_distance"],
        }
        for name, comparison in comparisons.items()
    }
    print(json.dumps({"passed": metadata["passed"], "output": str(args.output)}))
    print(json.dumps(summary, indent=2))
    if not metadata["passed"]:
        print(metadata["error"])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
