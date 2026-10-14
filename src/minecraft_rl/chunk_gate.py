"""Run the game on the efficiency cores of a Mac without stepping into
unloaded chunks (docs/decisions/simulation-throughput.md).

The game makes almost all of the heat of a Minecraft run: about 11 W at
default priority against 0.4 W for a trainer on the efficiency cores. Under
the macOS background policy the game needs about 1.6 W, but chunk loading
then falls behind lockstep, so the sensor can look into chunks that are not
loaded yet. A pause does not help: it spreads the same energy over a longer
time.

`ChunkGate` fixes the cause instead. The game runs with the background
policy. Before the first observation of an episode and before every step,
the gate asks the server (a privileged debug request, never policy input)
whether every chunk in sensor range is loaded and entity-ticking and every
chunk inside the simulation distance is entity-ticking. If not, it waits
while the world is frozen: no tick runs, chunk loading continues. If the
wait is longer than `boost_after` seconds, the game moves to default
priority until the chunks are ready, then back to the background policy.

The gate never changes what a step does: it only waits between steps, and
a step runs only when the chunks around the player are ready. This module
supports macOS only; elsewhere priority changes do nothing.
"""

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Protocol


class ChunkSource(Protocol):
    def chunks(self) -> dict[str, Any]: ...


def listening_pid(port: int) -> int | None:
    """The process that listens on the lockstep port: the game, not Gradle."""
    found = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    return int(found[0]) if found else None


def set_background(pid: int, background: bool) -> None:
    if sys.platform == "darwin":
        os.setpriority(
            os.PRIO_DARWIN_PROCESS, pid, os.PRIO_DARWIN_BG if background else 0
        )


def chunks_ready(chunks: dict[str, Any]) -> bool:
    sensor = chunks["sensor_chunks"]
    return (
        chunks["sensor_chunks_loaded"] == sensor
        and chunks["sensor_chunks_entity_ticking"] == sensor
        and chunks["simulation_chunks_entity_ticking"] == chunks["simulation_chunks"]
    )


@dataclass
class ChunkGate:
    source: ChunkSource
    pid: int | None
    boost_after: float = 0.5
    timeout: float = 180.0
    poll: float = 0.02
    checks: int = 0
    waits: int = 0
    wait_seconds: float = 0.0
    boosts: int = 0
    boost_seconds: float = 0.0
    longest_wait: float = 0.0

    def wait(self) -> None:
        """Return once the chunks around the player are ready."""
        self.checks += 1
        if chunks_ready(self.source.chunks()):
            return
        started = time.monotonic()
        boosted_at = None
        try:
            while True:
                time.sleep(self.poll)
                if chunks_ready(self.source.chunks()):
                    break
                waited = time.monotonic() - started
                if boosted_at is None and self.pid and waited >= self.boost_after:
                    set_background(self.pid, False)
                    boosted_at = time.monotonic()
                    self.boosts += 1
                if waited > self.timeout:
                    raise RuntimeError(f"chunks not ready after {waited:.0f} s")
        finally:
            if boosted_at is not None:
                set_background(self.pid, True)
                self.boost_seconds += time.monotonic() - boosted_at
        waited = time.monotonic() - started
        self.waits += 1
        self.wait_seconds += waited
        self.longest_wait = max(self.longest_wait, waited)

    def summary(self) -> dict[str, Any]:
        return {
            "checks": self.checks,
            "waits": self.waits,
            "wait_seconds": self.wait_seconds,
            "longest_wait_seconds": self.longest_wait,
            "boosts": self.boosts,
            "boost_seconds": self.boost_seconds,
        }
