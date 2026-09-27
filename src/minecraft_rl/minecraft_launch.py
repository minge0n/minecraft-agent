"""Launching and connecting to the Fabric development client for runtime probes.

Integration tooling only; nothing here is part of the policy or environment API.
"""

import os
import signal
import subprocess
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from minecraft_rl.minecraft_client import MinecraftClient

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


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


@contextmanager
def launched_client(
    port: int,
    seed: int,
    log_path: Path,
    startup_timeout: float,
    extra_environment: Mapping[str, str] | None = None,
) -> Iterator[MinecraftClient]:
    """Run `./scripts/gradle runClient` in observer mode and yield a connection.

    On exit the client is asked to quit; a client that does not stop is killed.
    """
    environment = (
        os.environ
        | {"MCBOT_TICK_PORT": str(port), "MCBOT_WORLD_SEED": str(seed)}
        | dict(extra_environment or {})
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        game = subprocess.Popen(
            [str(REPOSITORY_ROOT / "scripts" / "gradle"), "runClient"],
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
