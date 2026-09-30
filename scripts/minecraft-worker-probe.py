"""Two-worker isolation smoke test (docs/decisions/parallel-workers.md).

Launches two development clients concurrently as workers env-0000 and env-0001. Each
has its own game directory under `.runtime/workers/NNNN/client`, the offline
development identity AgentNNNN, its own tick-control port, world seed and recorder.
The test checks that:

- identities, UUIDs (vanilla offline UUIDs), game directories and ports differ and
  match the decided scheme;
- each worker's saves, logs and options stay inside its own game directory, and the
  shared single-client directory `.runtime/minecraft/client` is untouched;
- stepping one worker never advances the other, and both still advance exactly one
  tick per STEP while the other is idle or stepping;
- a controlled replay played on both workers at the same time is identical across
  workers (except the characterized per-run tick_count offset), so concurrent
  workers do not interfere with each other's transitions;
- each worker records its own session video.
"""

import argparse
import hashlib
import json
import platform
import threading
import time
import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from minecraft_rl.minecraft_client import MinecraftClient
from minecraft_rl.minecraft_interface import PlayerAction
from minecraft_rl.minecraft_launch import (
    REPOSITORY_ROOT,
    launched_client,
    worker_game_directory,
    worker_username,
)
from minecraft_rl.recording import WorkerRecorder, git_commit, write_json
from minecraft_rl.replay import (
    build_controlled_replay,
    compare_traces,
    play_replay,
    unexplained_fields,
)

WORKERS = (0, 1)
BASE_PORT = 47150
REPLAY_SEED = 12345
RECORDING = {"every_ticks": 1, "width": 426, "height": 240, "queue_frames": 64}
SHARED_CLIENT_DIRECTORY = REPOSITORY_ROOT / ".runtime" / "minecraft" / "client"


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def offline_uuid(username: str) -> str:
    digest = hashlib.md5(f"OfflinePlayer:{username}".encode()).digest()
    return str(uuid.UUID(bytes=digest, version=3))


def worker_name(worker: int) -> str:
    return f"env-{worker:04d}"


def snapshot(directory: Path) -> dict[str, float]:
    if not directory.exists():
        return {}
    return {
        str(path.relative_to(directory)): path.stat().st_mtime
        for path in directory.rglob("*")
        if path.is_file()
    }


def check_identity(worker: int, status: dict[str, Any], port: int) -> dict[str, Any]:
    identity = status["identity"]
    username = worker_username(worker)
    check(identity["username"] == username, f"worker {worker} is {identity}")
    check(identity["uuid"] == offline_uuid(username), f"unexpected UUID {identity}")
    check(
        identity["game_directory"] == str(worker_game_directory(worker)),
        f"unexpected game directory {identity}",
    )
    return identity | {"port": port}


def check_directory(worker: int) -> dict[str, Any]:
    directory = worker_game_directory(worker)
    saves = sorted(path.name for path in (directory / "saves").iterdir())
    log = (directory / "logs" / "latest.log").read_text(encoding="utf-8")
    check(
        any(name.startswith("mcbot-episode-") for name in saves),
        f"worker {worker} saves hold no episode world: {saves}",
    )
    check(
        f"Setting user: {worker_username(worker)}" in log,
        f"worker {worker} log does not name its identity",
    )
    check((directory / "options.txt").exists(), f"worker {worker} has no options")
    return {"game_directory": str(directory), "saves": saves}


def check_step_isolation(
    clients: dict[int, MinecraftClient],
) -> dict[str, Any]:
    """Step one worker while the other idles, in both directions."""
    for client in clients.values():
        client.set_pacing("unpaced")
    counts = {}
    for stepped, idle in ((0, 1), (1, 0)):
        idle_before = clients[idle].status()
        for _ in range(50):
            result = clients[stepped].step(PlayerAction())
            check(
                result.info.tick_after == result.info.tick_before + 1,
                f"worker {stepped} skipped a tick",
            )
        idle_after = clients[idle].status()
        check(
            (idle_before.game_time, idle_before.client_ticks)
            == (idle_after.game_time, idle_after.client_ticks),
            f"stepping worker {stepped} advanced worker {idle}",
        )
        counts[f"worker_{stepped}_stepped"] = {
            "steps": 50,
            f"worker_{idle}_game_time": idle_after.game_time,
        }
    for client in clients.values():
        client.set_pacing("paced")
    return counts


def concurrent_replays(
    clients: dict[int, MinecraftClient], recorders: dict[int, WorkerRecorder]
) -> dict[int, Any]:
    """Play the controlled replay on every worker at the same time."""
    scenes = {w: build_controlled_replay(c, REPLAY_SEED) for w, c in clients.items()}
    for client in clients.values():
        client.set_pacing("unpaced")
    results: dict[int, Any] = {}
    barrier = threading.Barrier(len(clients))

    def run(worker: int) -> None:
        try:
            recorders[worker].start()
            barrier.wait()
            started = time.perf_counter()
            records = play_replay(clients[worker], scenes[worker])
            elapsed = time.perf_counter() - started
            capture, recording = recorders[worker].stop()
            results[worker] = {
                "records": records,
                "steps_per_second": len(records) / elapsed,
                "capture": capture,
                "recording": recording,
            }
        except Exception as error:  # noqa: BLE001 - reported per worker below.
            results[worker] = {"error": f"{type(error).__name__}: {error}"}

    threads = [threading.Thread(target=run, args=(w,)) for w in clients]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(600)
    for client in clients.values():
        client.set_pacing("paced")
    for worker in clients:
        check(worker in results, f"worker {worker} replay did not finish")
        check("error" not in results[worker], f"worker {worker}: {results[worker]}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/minecraft-workers") / time.strftime("%Y%m%d-%H%M%S"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "tool": "minecraft-worker-probe",
        "commit": git_commit(),
        "mode": "debug",
        "host": {"system": platform.system(), "machine": platform.machine()},
        "workers": {},
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    shared_before = snapshot(SHARED_CLIENT_DIRECTORY)
    recorders: dict[int, WorkerRecorder] = {}
    try:
        with ExitStack() as stack:
            launches = {}
            launch_errors = {}

            def launch(worker: int) -> None:
                try:
                    launches[worker] = stack.enter_context(
                        launched_client(
                            BASE_PORT + worker,
                            1000 + worker,
                            args.output
                            / "workers"
                            / worker_name(worker)
                            / "gradle.log",
                            args.startup_timeout,
                            worker=worker,
                        )
                    )
                except Exception as error:  # noqa: BLE001
                    launch_errors[worker] = f"{type(error).__name__}: {error}"

            started = time.perf_counter()
            threads = [threading.Thread(target=launch, args=(w,)) for w in WORKERS]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            check(not launch_errors, f"launch failed: {launch_errors}")
            result["concurrent_startup_seconds"] = time.perf_counter() - started
            clients = {w: launches[w] for w in WORKERS}

            for worker, client in clients.items():
                status = client.status_json()
                _, reset = client.reset(1000 + worker, "flat")
                result["workers"][worker_name(worker)] = {
                    "identity": check_identity(worker, status, BASE_PORT + worker),
                    "world_seed": reset.seed,
                    "level_id": reset.level_id,
                }
                recorders[worker] = WorkerRecorder(
                    client,
                    args.output,
                    worker_name(worker),
                    RECORDING,
                    {
                        "run": str(args.output),
                        "mode": "debug",
                        "world_seed": REPLAY_SEED,
                        "commit": result["commit"],
                        "checkpoint": None,
                        "minecraft_version": "26.3",
                        "username": worker_username(worker),
                    },
                )
            identities = [w["identity"] for w in result["workers"].values()]
            for key in ("username", "uuid", "game_directory", "port"):
                values = [identity[key] for identity in identities]
                check(len(set(values)) == len(values), f"workers share {key}")

            result["step_isolation"] = check_step_isolation(clients)
            replays = concurrent_replays(clients, recorders)
            comparison = compare_traces(replays[0]["records"], replays[1]["records"])
            unexplained = unexplained_fields(comparison)
            check(not unexplained, f"concurrent replays differ: {unexplained}")
            result["concurrent_replay"] = {
                "steps": comparison["steps"],
                "unexplained_fields": unexplained,
                "steps_per_second": {
                    worker_name(w): replays[w]["steps_per_second"] for w in WORKERS
                },
            }
            for worker in WORKERS:
                recording = replays[worker]["recording"]
                capture = replays[worker]["capture"]
                check(
                    capture["status"] == "complete" and recording.status == "complete",
                    f"worker {worker} recording failed: {capture} {recording.error}",
                )
                check(
                    Path(recording.directory).is_relative_to(
                        args.output / "workers" / worker_name(worker)
                    ),
                    f"worker {worker} recorded outside its directory",
                )
                entry = result["workers"][worker_name(worker)]
                entry["recording"] = {
                    "directory": recording.directory,
                    "frames_written": recording.frames_written,
                    "due": capture["due"],
                    "dropped": capture["due"] - capture["captured"],
                    "video_bytes": recording.video_bytes,
                }
        for worker in WORKERS:
            entry = result["workers"][worker_name(worker)]
            entry["directory"] = check_directory(worker)
        check(
            snapshot(SHARED_CLIENT_DIRECTORY) == shared_before,
            "a worker wrote into the shared .runtime/minecraft/client directory",
        )
        result["shared_client_directory_untouched"] = True
        result["passed"] = True
    except Exception as error:
        result |= {"passed": False, "error": f"{type(error).__name__}: {error}"}
    finally:
        for recorder in recorders.values():
            recorder.close()
    result["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(args.output / "metadata.json", result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
