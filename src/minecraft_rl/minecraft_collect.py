"""Collect real Minecraft episodes with the exploration policy.

One process launches the game in observer mode, then plays episodes. Each
episode starts with a fresh world (`RESET`) of its own seed and ends on death
(terminated) or after `--max-steps` steps (truncated). The exploration policy
never reads the observation, except the camera pitch to stay near the
horizon. Every episode is recorded on video unless `--no-video` marks the run
as a non-recording infrastructure test (docs/decisions/recording.md).

The collector is resumable. It skips episodes that already exist and stops
starting new episodes after `--time-budget` seconds, so each call stays short.
The same command continues the dataset. World seeds are fixed per split and
episode index, so a resumed collection uses the same seeds.

Example:

    .venv/bin/python -m minecraft_rl.minecraft_collect --dataset runs/stage3/data \\
        --split train --episodes 8 --max-steps 1200 --time-budget 420
"""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

from minecraft_rl import runtime
from minecraft_rl.exploration import ExplorationConfig, ExplorationPolicy
from minecraft_rl.minecraft_client import MinecraftClient
from minecraft_rl.minecraft_dataset import (
    DATASET_FORMAT,
    EpisodeMeta,
    EpisodeWriter,
    episode_directories,
    write_json,
)
from minecraft_rl.minecraft_interface import OBSERVATION_SCHEMA
from minecraft_rl.minecraft_launch import launched_client
from minecraft_rl.provenance import git_commit
from minecraft_rl.recording import WorkerRecorder

# World seeds per split: disjoint ranges, so held-out seeds never train.
SPLIT_SEED_BASE = {"train": 100_000, "eval_episode": 200_000, "eval_seed": 300_000}
POLICY_SEED_OFFSET = 7_919
WORKER = "env-0000"


def world_seed(split: str, episode: int) -> int:
    return SPLIT_SEED_BASE[split] + episode


def policy_seed(split: str, episode: int) -> int:
    return world_seed(split, episode) * POLICY_SEED_OFFSET % 2_147_483_647


def run_episode(
    client: MinecraftClient,
    split: str,
    episode: int,
    preset: str,
    max_steps: int,
    config: ExplorationConfig,
    recorder: WorkerRecorder | None,
    directory: Path,
) -> dict[str, Any]:
    seed = world_seed(split, episode)
    observation, reset = client.reset(seed, preset)
    policy = ExplorationPolicy(config, policy_seed(split, episode))
    writer = EpisodeWriter(observation)
    if recorder is not None:
        recorder.episode = episode
        recorder.start()
    started = time.perf_counter()
    step_ms: list[float] = []
    terminated = False
    first_tick = last_tick = reset.game_time
    while writer.steps < max_steps and not terminated:
        action = policy.act(pitch=observation.pitch)
        step_started = time.perf_counter()
        result = client.step(action)
        step_ms.append((time.perf_counter() - step_started) * 1000.0)
        if result.info.tick_after != result.info.tick_before + 1:
            raise RuntimeError(
                f"step advanced {result.info.tick_before}->{result.info.tick_after}"
            )
        observation = result.observation
        terminated = result.terminated
        last_tick = result.info.tick_after
        writer.add(
            action,
            observation,
            terminated,
            result.info.step_id,
            result.info.tick_before,
            result.info.tick_after,
            result.info.client_tick,
        )
    wall = time.perf_counter() - started
    recording: dict[str, Any] = {"status": "disabled"}
    if recorder is not None:
        capture, encoded = recorder.stop()
        recording = {
            "status": encoded.status,
            "capture_status": capture.get("status"),
            "video_bytes": encoded.video_bytes,
            "directory": str(encoded.directory),
        }
    meta = EpisodeMeta(
        episode=episode,
        world_seed=seed,
        preset=preset,
        level_id=reset.level_id,
        split=split,
        policy_seed=policy_seed(split, episode),
        steps=writer.steps,
        terminated=terminated,
        truncated=not terminated,
        first_tick=first_tick,
        last_tick=last_tick,
    )
    write_started = time.perf_counter()
    writer.write(directory, meta)
    stats = {
        "steps": writer.steps,
        "terminated": terminated,
        "reset_ms": reset.reset_ms,
        "step_wall_seconds": wall,
        "steps_per_second": writer.steps / wall if wall else 0.0,
        "median_step_ms": statistics.median(step_ms) if step_ms else 0.0,
        "write_seconds": time.perf_counter() - write_started,
        "npz_bytes": (directory / "episode.npz").stat().st_size,
        "recording": recording,
    }
    write_json(directory / "collection.json", stats)
    return stats


def manifest(args: argparse.Namespace, config: ExplorationConfig) -> dict[str, Any]:
    return {
        "format": DATASET_FORMAT,
        "observation_schema": OBSERVATION_SCHEMA,
        "git_commit": git_commit(),
        "exploration_policy": config.to_json(),
        "split_seed_base": SPLIT_SEED_BASE,
        "policy_seed_rule": f"world_seed * {POLICY_SEED_OFFSET} mod 2^31 - 1",
        "preset": args.preset,
        "max_steps": args.max_steps,
        "pacing": "unpaced",
        "video": not args.no_video,
        "recording_settings": recording_settings(args),
        "worker": WORKER,
        "platform": {"system": platform.system(), "machine": platform.machine()},
    }


def recording_settings(args: argparse.Namespace) -> dict[str, int]:
    return {
        "every_ticks": args.every_ticks,
        "width": 426,
        "height": 240,
        "queue_frames": 64,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(SPLIT_SEED_BASE), required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--preset", choices=("flat", "normal"), default="normal")
    parser.add_argument("--time-budget", type=float, default=420.0)
    parser.add_argument("--every-ticks", type=int, default=2)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--port", type=int, default=47130)
    parser.add_argument("--startup-timeout", type=float, default=240.0)
    args = parser.parse_args()
    runtime.configure(0)
    config = ExplorationConfig()
    root = args.dataset / args.split
    existing = {path.name for path in episode_directories(root)}
    todo = [e for e in range(args.episodes) if f"{e:06d}" not in existing]
    if not todo:
        print(f"{args.split}: all {args.episodes} episodes exist")
        return
    info = manifest(args, config)
    manifest_path = args.dataset / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        changed = [
            k
            for k in ("exploration_policy", "observation_schema", "preset", "max_steps")
            if previous.get(k) != info[k]
        ]
        if changed:
            raise SystemExit(f"dataset settings differ from {manifest_path}: {changed}")
    write_json(manifest_path, info)

    started = time.monotonic()
    log = args.dataset / "logs" / f"{args.split}-{int(time.time())}.minecraft.log"
    with launched_client(
        args.port, world_seed(args.split, todo[0]), log, args.startup_timeout
    ) as client:
        client.set_pacing("unpaced", render_frames=True)
        recorder = None
        if not args.no_video:
            recorder = WorkerRecorder(
                client,
                args.dataset / "video" / args.split,
                WORKER,
                recording_settings(args),
                {"dataset": str(args.dataset), "split": args.split},
            )
        try:
            for episode in todo:
                if time.monotonic() - started > args.time_budget:
                    break
                directory = root / "episodes" / f"{episode:06d}"
                stats = run_episode(
                    client,
                    args.split,
                    episode,
                    args.preset,
                    args.max_steps,
                    config,
                    recorder,
                    directory,
                )
                outcome = "died" if stats["terminated"] else "truncated"
                print(
                    f"{args.split} episode {episode}: {stats['steps']} steps, "
                    f"{outcome}, {stats['steps_per_second']:.0f} steps/s, "
                    f"video {stats['recording']['status']}",
                    flush=True,
                )
        finally:
            if recorder is not None:
                recorder.close()
    left = [
        e
        for e in range(args.episodes)
        if not (root / "episodes" / f"{e:06d}" / "episode.json").exists()
    ]
    if left:
        raise SystemExit(f"unfinished episodes, run the same command again: {left}")


if __name__ == "__main__":
    main()
