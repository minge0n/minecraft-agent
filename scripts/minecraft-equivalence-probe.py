"""Validate accelerated (unpaced) lockstep against paced lockstep.

Every run resets to a fresh world from the same seed, builds the `replay` debug scene
and plays the fixed replay action script, alternating paced and unpaced modes inside
each client process. The `controlled` scene removes world RNG consumers, so paced and
unpaced traces must be identical there except for the per-run player `tick_count`
offset characterized in docs/replay-characterization.md. The `natural` scene keeps mob
AI and drop randomness; acceleration must not add divergence beyond the paced-vs-paced
baseline. The probe also checks the invariants directly: one world tick and one client
tick per STEP, no progress while idle, same-step action effects, and the tick counts
for mining and combat.
"""

import argparse
import itertools
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

from minecraft_rl.minecraft_client import MinecraftClient
from minecraft_rl.minecraft_launch import REPOSITORY_ROOT, launched_client
from minecraft_rl.replay import (
    REMOVE_SAND_SUPPORT,
    compare_traces,
    scripted_actions,
    scripted_events,
    trace_record,
    unexplained_fields,
)

SCENE_COLUMN = (8, 8)
MODES = {"natural": {"ai": True}, "controlled": {"controlled": True}}
PACINGS = ("paced", "unpaced")
IDLE_SECONDS = 2.0


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_replay(
    client: MinecraftClient, seed: int, mode: str, pacing: str
) -> dict[str, Any]:
    client.set_pacing("paced")
    client.reset(seed, "flat")
    privileged = client.privileged()
    scene = privileged.scene("replay", at=SCENE_COLUMN, **MODES[mode])
    roles = {scene["husk"]: "husk"}
    if "pig" in scene:
        roles[scene["pig"]] = "pig"
    region = (scene["arena_from"], scene["arena_to"])
    events = scripted_events()
    client.set_pacing(pacing)
    records = []
    step_seconds = []
    for step, action in enumerate(scripted_actions()):
        if events.get(step) == REMOVE_SAND_SUPPORT:
            support = scene["sand_support"]
            privileged.fill(support, support, "minecraft:air")
        started = time.perf_counter()
        result = client.step(action)
        step_seconds.append(time.perf_counter() - started)
        info = result.info
        check(info.tick_after == info.tick_before + 1, f"step {step} skipped ticks")
        check(info.pacing == pacing, f"step {step} ran with {info.pacing} pacing")
        trace = privileged.trace(*region)
        records.append(
            trace_record(step, action, events.get(step), result, trace, roles)
        )
    for previous, current in itertools.pairwise(records):
        check(
            current["info"]["client_tick"] == previous["info"]["client_tick"] + 1,
            f"client ticks not consecutive at step {current['step']}",
        )
        check(
            current["info"]["tick_before"] == previous["info"]["tick_after"],
            f"world ticks not consecutive at step {current['step']}",
        )
    idle_before = client.status()
    time.sleep(IDLE_SECONDS)
    idle_after = client.status()
    check(
        (idle_before.game_time, idle_before.client_ticks)
        == (idle_after.game_time, idle_after.client_ticks),
        f"{pacing} simulation advanced while idle",
    )
    client.set_pacing("paced")
    return {
        "records": records,
        "steps_per_second": len(step_seconds) / sum(step_seconds),
        "median_step_ms": statistics.median(step_seconds) * 1000.0,
        "idle_frozen_seconds": IDLE_SECONDS,
    }


def milestones(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Tick-level facts that must match between pacing modes."""
    husk_health = [
        record["privileged"]["entities"]["husk"]["health"] for record in records
    ]
    hits = [
        index
        for index in range(1, len(husk_health))
        if husk_health[index] < husk_health[index - 1]
    ]
    positions = [
        (
            record["privileged"]["server_player"]["x"],
            record["privileged"]["server_player"]["y"],
            record["privileged"]["server_player"]["z"],
        )
        for record in records
    ]
    block_changes = [
        index
        for index in range(1, len(records))
        if records[index]["privileged"]["block_crc32"]
        != records[index - 1]["privileged"]["block_crc32"]
    ]
    return {
        "husk_hit_steps": hits,
        "husk_health_final": husk_health[-1],
        "block_change_steps": block_changes,
        "max_player_y": max(y for _, y, _ in positions),
        "final_player_position": positions[-1],
        "final_inventory": records[-1]["privileged"]["server_player"]["inventory"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=47126)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--processes", type=int, default=2)
    parser.add_argument("--runs-per-process", type=int, default=2)
    parser.add_argument("--startup-timeout", type=float, default=240.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/minecraft-equivalence") / time.strftime("%Y%m%d-%H%M%S"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    traces: dict[str, list[dict[str, Any]]] = {}
    runs: dict[str, Any] = {}
    result: dict[str, Any] = {
        "tool": "minecraft-equivalence-probe",
        "seed": args.seed,
        "scene": "replay",
        "steps": len(scripted_actions()),
        "host": {"system": platform.system(), "machine": platform.machine()},
        "recording": "non-recording infrastructure test",
    }
    try:
        for process in range(args.processes):
            log_path = args.output / f"process-{process}.minecraft.log"
            with launched_client(
                args.port, args.seed, log_path, args.startup_timeout
            ) as client:
                result.setdefault("client_status", client.request("STATUS"))
                for run, mode, pacing in itertools.product(
                    range(args.runs_per_process), MODES, PACINGS
                ):
                    label = f"{mode}-{pacing}-p{process}r{run}"
                    replay = run_replay(client, args.seed, mode, pacing)
                    traces[label] = replay.pop("records")
                    runs[label] = replay
                    path = args.output / f"trace-{label}.jsonl"
                    with path.open("w", encoding="utf-8") as trace_file:
                        for record in traces[label]:
                            trace_file.write(json.dumps(record) + "\n")
        result["passed"] = True
    except Exception as error:
        result |= {"passed": False, "error": f"{type(error).__name__}: {error}"}

    comparisons: dict[str, Any] = {}
    for a, b in itertools.combinations(sorted(traces), 2):
        mode_a, pacing_a, run_a = a.split("-")
        mode_b, pacing_b, run_b = b.split("-")
        if mode_a != mode_b:
            continue
        pair = "-".join(sorted((pacing_a, pacing_b)))
        comparison = compare_traces(traces[a], traces[b])
        comparisons[f"{a}-vs-{b}"] = {
            "mode": mode_a,
            "pacing_pair": pair,
            "same_process": run_a[:2] == run_b[:2],
            "unexplained_fields": unexplained_fields(comparison),
        } | comparison

    summary: dict[str, Any] = {}
    for mode, pair in itertools.product(MODES, ("paced-paced", "paced-unpaced")):
        group = [
            c
            for c in comparisons.values()
            if c["mode"] == mode and c["pacing_pair"] == pair
        ]
        if not group:
            continue
        summary[f"{mode}/{pair}"] = {
            "pairs": len(group),
            "pairs_with_unexplained_fields": sum(
                bool(c["unexplained_fields"]) for c in group
            ),
            "observation_differing_steps": sorted(
                c["policy_observation"]["differing_steps"] for c in group
            ),
            "player_max_distance": max(
                c["player_position"]["max_distance"] for c in group
            ),
        }
    if result.get("passed"):
        # Mining, combat, placement and falling-sand timing must be tick-identical
        # across pacing modes in the controlled scene.
        controlled = sorted(label for label in traces if label.startswith("controlled"))
        facts = {label: milestones(traces[label]) for label in controlled}
        reference = facts[controlled[0]]
        summary["controlled/milestones"] = reference
        mismatched = [label for label, fact in facts.items() if fact != reference]
        if mismatched:
            result |= {"passed": False, "error": f"milestones differ: {mismatched}"}
        if summary.get("controlled/paced-unpaced", {}).get(
            "pairs_with_unexplained_fields"
        ):
            result |= {
                "passed": False,
                "error": "controlled paced and unpaced traces differ",
            }
    throughput = {
        pacing: statistics.median(
            run["steps_per_second"]
            for label, run in runs.items()
            if f"-{pacing}-" in label
        )
        for pacing in PACINGS
        if any(f"-{pacing}-" in label for label in runs)
    }
    result |= {
        "summary": summary,
        "throughput_steps_per_second": throughput,
        "runs": runs,
        "comparisons": comparisons,
    }
    (args.output / "equivalence.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "error": result.get("error"),
                "output": str(args.output.relative_to(REPOSITORY_ROOT))
                if args.output.is_absolute()
                else str(args.output),
                "summary": summary,
                "throughput_steps_per_second": throughput,
            },
            indent=2,
            default=str,
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
