"""Validate and benchmark session recording (docs/decisions/recording.md).

Runs the controlled replay script with recording on and checks that:

- recording never changes a step: the policy observation and privileged trace of a
  recorded run equal those of an unrecorded run (except the characterized per-run
  player `tick_count` offset), and every step still advances exactly one tick;
- every due frame is accounted for (captured and written, or counted as dropped), and
  `frames.jsonl` maps each video frame to a strictly increasing client tick;
- a recorder that disappears mid-episode marks the recording failed while the
  simulation keeps stepping;
- the cost of simulation only, plus rendering, plus capture, plus encoding.

This probe is itself a recorded integration session: its videos are written under the
run directory with the layout from docs/decisions/recording.md.
"""

import argparse
import json
import platform
import socket
import statistics
import threading
import time
from pathlib import Path
from typing import Any

from minecraft_rl.minecraft_client import MinecraftClient
from minecraft_rl.minecraft_interface import PlayerAction
from minecraft_rl.minecraft_launch import launched_client
from minecraft_rl.provenance import git_commit
from minecraft_rl.recording import (
    EpisodeRecording,
    WorkerRecorder,
    read_stream,
    storage_summary,
    write_json,
)
from minecraft_rl.replay import (
    build_controlled_replay,
    compare_traces,
    play_replay,
    unexplained_fields,
)

WORKER = "env-0000"
BENCHMARK_STEPS = 400


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def replay(
    client: MinecraftClient, seed: int, recorder: WorkerRecorder | None
) -> dict[str, Any]:
    scene = build_controlled_replay(client, seed)
    client.set_pacing("unpaced")
    directory = recorder.start() if recorder else None
    out: dict[str, Any] = {"records": play_replay(client, scene)}
    if recorder:
        capture, recording = recorder.stop()
        out |= {"capture": capture, "recording": recording, "directory": directory}
    client.set_pacing("paced")
    return out


def check_recording(
    capture: dict[str, Any], recording: EpisodeRecording, directory: Path
) -> dict[str, Any]:
    check(capture["status"] == "complete", f"capture failed: {capture}")
    check(recording.status == "complete", f"encoding failed: {recording.error}")
    dropped = (
        capture["dropped_gpu_busy"]
        + capture["dropped_queue_full"]
        + capture["dropped_after_failure"]
        + capture["missed_unrendered"]
    )
    check(
        capture["due"] == capture["captured"] + dropped,
        f"due frames not accounted for: {capture}",
    )
    check(capture["sent"] == capture["captured"], f"captured frames lost: {capture}")
    check(
        recording.frames_written == capture["sent"],
        f"recorder wrote {recording.frames_written} of {capture['sent']} frames",
    )
    ticks = [
        json.loads(line)["client_tick"]
        for line in (directory / "frames.jsonl").read_text().splitlines()
    ]
    check(ticks == sorted(set(ticks)), "frame ticks are not strictly increasing")
    check(
        ticks[0] >= capture["start_client_tick"]
        and ticks[-1] <= capture["stop_client_tick"],
        "frame ticks outside the episode",
    )
    return {
        "due": capture["due"],
        "written": recording.frames_written,
        "dropped": dropped,
        "non_consecutive_frames": recording.non_consecutive_frames,
        "video_bytes": recording.video_bytes,
        "readback_ms_mean": capture["readback_ms_mean"],
        "queue_high_water": capture["queue_high_water"],
    }


def check_equivalence(
    reference: list[dict[str, Any]], recorded: list[dict[str, Any]]
) -> dict[str, Any]:
    comparison = compare_traces(reference, recorded)
    unexplained = unexplained_fields(comparison)
    check(not unexplained, f"recording changed the trace: {unexplained}")
    return {
        "steps": comparison["steps"],
        "policy_observation_differing_steps": comparison["policy_observation"][
            "differing_steps"
        ],
        "unexplained_fields": unexplained,
    }


def run_failure(
    client: MinecraftClient, recorder: WorkerRecorder, seed: int
) -> dict[str, Any]:
    """The recorder vanishes mid-episode; the simulation must keep stepping."""
    client.reset(seed, "flat")
    client.set_pacing("unpaced")
    listener = socket.create_server(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    client.start_recording(recorder.episode, port, **recorder.settings)
    connection, _ = listener.accept()
    connection.close()
    listener.close()
    steps = 0
    for _ in range(200):
        result = client.step(PlayerAction())
        check(result.info.tick_after == result.info.tick_before + 1, "skipped tick")
        steps += 1
    capture = client.stop_recording()
    recorder.episode += 1
    client.set_pacing("paced")
    check(capture["status"] == "failed", f"lost recorder not reported: {capture}")
    return {"steps_after_failure": steps, "capture": capture}


def measure(
    client: MinecraftClient,
    seed: int,
    label: str,
    render_frames: bool,
    recorder: WorkerRecorder,
    sink: str | None,
) -> dict[str, Any]:
    """Unpaced steps with no recording, capture into a discarding sink, or encoding."""
    client.reset(seed, "flat")
    client.set_pacing("unpaced", render_frames=render_frames)
    discard: DiscardSink | None = None
    if sink == "encode":
        recorder.start()
    elif sink == "discard":
        discard = DiscardSink()
        client.start_recording(recorder.episode, discard.port, **recorder.settings)
    step_ms = []
    started = time.perf_counter()
    for index in range(BENCHMARK_STEPS):
        action = PlayerAction(yaw_delta=4.5, forward=index % 40 < 20)
        step_started = time.perf_counter()
        client.step(action)
        step_ms.append((time.perf_counter() - step_started) * 1000.0)
    elapsed = time.perf_counter() - started
    out: dict[str, Any] = {
        "configuration": label,
        "render_frames": render_frames,
        "steps": BENCHMARK_STEPS,
        "steps_per_second": BENCHMARK_STEPS / elapsed,
        "median_step_ms": statistics.median(step_ms),
        "p95_step_ms": statistics.quantiles(step_ms, n=20)[-1],
    }
    if sink == "encode":
        capture, recording = recorder.stop()
        out |= {
            "capture": capture,
            "recording": recording.summary(),
            "encode_ms_per_frame": recording.encode_seconds
            * 1000.0
            / max(recording.frames_written, 1),
        }
    elif discard is not None:
        capture = client.stop_recording()
        recorder.episode += 1
        out |= {"capture": capture, "frames_discarded": discard.finish()}
    client.set_pacing("paced")
    return out


class DiscardSink:
    """Reads a frame stream and discards the frames: capture cost without encoding."""

    def __init__(self) -> None:
        self.listener = socket.create_server(("127.0.0.1", 0))
        self.port = self.listener.getsockname()[1]
        self.frames = 0
        self.thread = threading.Thread(target=self._drain, daemon=True)
        self.thread.start()

    def _drain(self) -> None:
        connection, _ = self.listener.accept()
        with connection, connection.makefile("rb", buffering=1 << 20) as stream:
            for message in read_stream(stream):
                if message is not None and hasattr(message, "rgba"):
                    self.frames += 1

    def finish(self) -> int:
        self.thread.join(30)
        self.listener.close()
        return self.frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=47128)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--every-ticks", type=int, default=1)
    parser.add_argument("--width", type=int, default=426)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--queue-frames", type=int, default=64)
    parser.add_argument("--startup-timeout", type=float, default=240.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/minecraft-recording") / time.strftime("%Y%m%d-%H%M%S"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {
        "every_ticks": args.every_ticks,
        "width": args.width,
        "height": args.height,
        "queue_frames": args.queue_frames,
    }
    result: dict[str, Any] = {
        "tool": "minecraft-recording-probe",
        "commit": git_commit(),
        "seed": args.seed,
        "worker": WORKER,
        "mode": "debug",
        "host": {"system": platform.system(), "machine": platform.machine()},
        "recording_settings": settings,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    recorder: WorkerRecorder | None = None
    try:
        with launched_client(
            args.port,
            args.seed,
            args.output / "workers" / WORKER / "minecraft.log",
            args.startup_timeout,
        ) as client:
            result["client_status"] = client.status_json()
            recorder = WorkerRecorder(
                client,
                args.output,
                WORKER,
                settings,
                {
                    "run": str(args.output),
                    "mode": "debug",
                    "world_seed": args.seed,
                    "commit": result["commit"],
                    "checkpoint": None,
                    "minecraft_version": "26.3",
                },
            )
            unrecorded = replay(client, args.seed, None)
            recorded = replay(client, args.seed, recorder)
            result["equivalence"] = check_equivalence(
                unrecorded["records"], recorded["records"]
            )
            result["recorded_replay"] = check_recording(
                recorded["capture"], recorded["recording"], recorded["directory"]
            )
            result["failure_isolation"] = run_failure(client, recorder, args.seed)
            result["benchmark"] = [
                measure(client, args.seed, label, render, recorder, sink)
                for label, render, sink in (
                    ("simulation_only", False, None),
                    ("rendering", True, None),
                    ("rendering_capture", True, "discard"),
                    ("rendering_capture_encode", True, "encode"),
                )
            ]
            simulated = sum(
                capture["stop_client_tick"] - capture["start_client_tick"]
                for capture in recorder.capture_stats
            )
            result["storage"] = storage_summary(recorder.recordings, simulated)
        result["passed"] = True
    except Exception as error:
        result |= {"passed": False, "error": f"{type(error).__name__}: {error}"}
    finally:
        if recorder is not None:
            recorder.close()
            result["episodes"] = [
                recording.summary() for recording in recorder.recordings
            ]
    result["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(args.output / "metadata.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
