"""Launching and connecting to the Fabric development client for runtime probes.

Integration tooling only; nothing here is part of the policy or environment API.
"""

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from minecraft_rl.minecraft_client import MinecraftClient

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
# Client renderer (docs/decisions/simulation-throughput.md): `vanilla` (the
# default, the measured best), `sodium`, or `sodium-extra` (Sodium and Sodium
# Extra with Reduce Resolution on macOS). Select it with MCBOT_RENDERER.
RENDERERS = ("vanilla", "sodium", "sodium-extra")
DEFAULT_RENDERER = "vanilla"


def renderer_of(environment: Mapping[str, str]) -> str:
    renderer = environment.get("MCBOT_RENDERER", DEFAULT_RENDERER)
    if renderer not in RENDERERS:
        raise ValueError(f"MCBOT_RENDERER must be one of {RENDERERS}, got {renderer!r}")
    return renderer


def wait_for_world(port: int, deadline: float) -> MinecraftClient:
    """Connect once the client reports a frozen, gated, unpaused world."""
    while time.monotonic() < deadline:
        try:
            client = MinecraftClient.connect(port)
        except OSError:
            time.sleep(1)
            continue
        try:
            status = client.status()
            if status.frozen and status.client_gated and not status.paused:
                return client
        except (OSError, RuntimeError, ValueError, KeyError):
            pass
        client.close()
        time.sleep(1)
    raise TimeoutError("Minecraft did not reach a frozen, gated, unpaused world")


SODIUM_EXTRA_CONFIG = "sodium-extra-options.json"


def client_game_directory(worker: int | None) -> Path:
    if worker is not None:
        return worker_game_directory(worker)
    return REPOSITORY_ROOT / ".runtime" / "minecraft" / "client"


def write_renderer_config(game_directory: Path, renderer: str) -> None:
    """Turn on Reduce Resolution on macOS in the project-owned Sodium Extra
    config before the client starts. Other settings keep their values; a
    missing file becomes one with this setting, which Sodium Extra completes
    with its defaults on load. Never touches a personal installation."""
    if renderer != "sodium-extra":
        return
    path = game_directory / "config" / SODIUM_EXTRA_CONFIG
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    data.setdefault("extra_settings", {})["reduce_resolution_on_mac"] = True
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def worker_game_directory(worker: int) -> Path:
    return REPOSITORY_ROOT / ".runtime" / "workers" / f"{worker:04d}" / "client"


def worker_username(worker: int) -> str:
    return f"Agent{worker:04d}"


@contextmanager
def launched_client(
    port: int,
    seed: int,
    log_path: Path,
    startup_timeout: float,
    extra_environment: Mapping[str, str] | None = None,
    worker: int | None = None,
    background: bool = False,
) -> Iterator[MinecraftClient]:
    """Run `./scripts/gradle runClient` in observer mode and yield a connection.

    With `worker`, the client runs as isolated worker env-NNNN: its own game directory
    and the offline development identity AgentNNNN
    (docs/decisions/parallel-workers.md). On exit the client is asked to quit; a
    client that does not stop is killed.

    With `background` on macOS, the game starts under `taskpolicy -b`: Gradle
    and the game process it starts inherit the background policy and run on
    the efficiency cores. It changes where and how fast the game computes,
    never what a step does.

    The renderer comes from MCBOT_RENDERER (`renderer_of`).
    """
    environment = (
        os.environ
        | {"MCBOT_TICK_PORT": str(port), "MCBOT_WORLD_SEED": str(seed)}
        | dict(extra_environment or {})
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        command = [str(REPOSITORY_ROOT / "scripts" / "gradle"), "runClient"]
        if background and sys.platform == "darwin":
            command = ["taskpolicy", "-b", *command]
        renderer = renderer_of(environment)
        write_renderer_config(client_game_directory(worker), renderer)
        command.append(f"-Pmcbot.renderer={renderer}")
        if worker is not None:
            command.append(f"-Pmcbot.worker={worker:04d}")
        game = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            with wait_for_world(port, time.monotonic() + startup_timeout) as client:
                try:
                    yield client
                finally:
                    try:
                        client.quit()
                    except (OSError, RuntimeError):
                        pass
        finally:
            try:
                game.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(game.pid, signal.SIGTERM)
                game.wait(timeout=30)
