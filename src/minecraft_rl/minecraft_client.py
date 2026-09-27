import json
import socket
import time
from dataclasses import dataclass
from typing import Any, Self

from minecraft_rl.minecraft_interface import (
    ObservationSchema,
    PlayerAction,
    PolicyObservation,
)
from minecraft_rl.tick_control import Status

WORLD_PRESETS = ("flat", "normal")


class ProtocolError(RuntimeError):
    pass


def encode_request(command: str, payload: dict[str, Any] | None = None) -> bytes:
    body = "" if payload is None else " " + json.dumps(payload, separators=(",", ":"))
    return f"v2 {command}{body}\n".encode()


def parse_reply(command: str, line: str) -> dict[str, Any]:
    if not line:
        raise ConnectionError("tick control closed the connection")
    version, _, rest = line.rstrip("\n").partition(" ")
    reply_command, _, body = rest.partition(" ")
    if version != "v2":
        raise ProtocolError(f"unexpected protocol version in {line.strip()!r}")
    if reply_command == "ERROR":
        raise ProtocolError(f"{command} rejected: {body}")
    if reply_command != command:
        raise ProtocolError(f"unexpected reply to {command}: {reply_command}")
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ProtocolError(f"{command} reply is not a JSON object")
    return payload


@dataclass(frozen=True)
class StepTiming:
    """Wall-clock phases of one step in milliseconds; diagnostic only."""

    client_wait_ms: float
    client_tick_ms: float
    tick_end_wait_ms: float
    server_step_ms: float
    observation_ms: float
    total_ms: float
    round_trip_ms: float


@dataclass(frozen=True)
class StepInfo:
    """Diagnostic step metadata; never a policy input."""

    step_id: int
    tick_before: int
    tick_after: int
    client_tick: int
    game_time: int
    timing: StepTiming


@dataclass(frozen=True)
class StepResult:
    observation: PolicyObservation
    terminated: bool
    info: StepInfo


@dataclass(frozen=True)
class ResetInfo:
    """Diagnostic reset metadata; never a policy input."""

    seed: int
    preset: str
    level_id: str
    game_time: int
    reset_ms: float


class MinecraftClient:
    def __init__(self, connection: socket.socket) -> None:
        self.connection = connection
        self.reader = connection.makefile("r", encoding="utf-8")
        self._schema: ObservationSchema | None = None

    @classmethod
    def connect(cls, port: int, timeout: float = 200.0) -> Self:
        connection = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return cls(connection)

    def close(self) -> None:
        self.reader.close()
        self.connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def request(
        self, command: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        self.connection.sendall(encode_request(command, payload))
        return parse_reply(command, self.reader.readline())

    def schema(self) -> ObservationSchema:
        if self._schema is None:
            self._schema = ObservationSchema.from_json(self.request("SCHEMA"))
        return self._schema

    def status(self) -> Status:
        reply = self.request("STATUS")
        return Status(
            reply["game_time"],
            reply["server_tick"],
            reply["frozen"],
            reply["paused"],
            reply["client_gated"],
            reply["client_ticks"],
        )

    def observe(self) -> PolicyObservation:
        schema = self.schema()
        return PolicyObservation.from_json(
            self.request("OBSERVE")["observation"], schema
        )

    def step(self, action: PlayerAction) -> StepResult:
        schema = self.schema()
        started = time.perf_counter()
        reply = self.request("STEP", action.to_json())
        round_trip_ms = (time.perf_counter() - started) * 1000.0
        info = reply["info"]
        return StepResult(
            observation=PolicyObservation.from_json(reply["observation"], schema),
            terminated=reply["terminated"],
            info=StepInfo(
                step_id=info["step_id"],
                tick_before=info["tick_before"],
                tick_after=info["tick_after"],
                client_tick=info["client_tick"],
                game_time=info["game_time"],
                timing=StepTiming(**info["timing"], round_trip_ms=round_trip_ms),
            ),
        )

    def reset(
        self, seed: int, preset: str = "flat"
    ) -> tuple[PolicyObservation, ResetInfo]:
        if preset not in WORLD_PRESETS:
            raise ValueError(f"unknown world preset {preset!r}")
        schema = self.schema()
        reply = self.request("RESET", {"seed": seed, "preset": preset})
        info = reply["info"]
        return PolicyObservation.from_json(reply["observation"], schema), ResetInfo(
            seed=info["seed"],
            preset=info["preset"],
            level_id=info["level_id"],
            game_time=info["game_time"],
            reset_ms=info["reset_ms"],
        )

    def quit(self) -> None:
        self.request("QUIT")

    def privileged(self) -> "PrivilegedProbe":
        return PrivilegedProbe(self)


class PrivilegedProbe:
    """Test-only access to full game state; outputs must never reach a policy."""

    def __init__(self, client: MinecraftClient) -> None:
        self.client = client

    def scene(
        self,
        name: str,
        ai: bool = False,
        at: tuple[int, int] | None = None,
        controlled: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": name, "ai": ai, "controlled": controlled}
        if at is not None:
            payload["at"] = list(at)
        return self.client.request("DEBUG_SCENE", payload)

    def trace(self, start: list[int], end: list[int]) -> dict[str, Any]:
        return self.client.request("DEBUG_TRACE", {"from": start, "to": end})

    def fill(self, start: list[int], end: list[int], block: str) -> dict[str, Any]:
        return self.client.request(
            "DEBUG_FILL", {"from": start, "to": end, "block": block}
        )

    def block(self, pos: list[int]) -> str:
        return self.client.request("DEBUG_BLOCK", {"pos": pos})["block"]

    def entity(self, entity_id: int) -> dict[str, Any]:
        return self.client.request("DEBUG_ENTITY", {"id": entity_id})

    def player(self) -> dict[str, Any]:
        return self.client.request("DEBUG_PLAYER")

    def kill(self) -> bool:
        return self.client.request("DEBUG_KILL")["dead"]

    def nearby(self, radius: int) -> dict[str, Any]:
        return self.client.request("DEBUG_NEARBY", {"radius": radius})

    def registry(self) -> dict[str, list[str]]:
        return self.client.request("DEBUG_REGISTRY")
