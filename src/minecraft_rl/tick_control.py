import socket
from dataclasses import dataclass
from typing import Self


class TickControlError(RuntimeError):
    pass


@dataclass(frozen=True)
class Status:
    game_time: int
    server_tick: int
    frozen: bool
    paused: bool


@dataclass(frozen=True)
class Step:
    step_id: int
    tick_before: int
    tick_after: int


@dataclass(frozen=True)
class ProbeSnapshot:
    """Privileged test-only state; never a policy observation."""

    game_time: int
    armor_stand_y: float
    husk_x: float
    husk_z: float
    husk_player_distance: float


def encode_request(command: str) -> bytes:
    return f"v1 {command}\n".encode()


def parse_reply(command: str, line: str) -> list[str]:
    if not line:
        raise ConnectionError("tick control closed the connection")
    fields = line.split()
    if fields[:2] == ["v1", "ERROR"]:
        raise TickControlError(f"{command} rejected: {' '.join(fields[2:])}")
    if fields[:2] != ["v1", command]:
        raise TickControlError(f"unexpected reply to {command}: {line.strip()!r}")
    return fields[2:]


def _parse_bool(value: str) -> bool:
    if value not in ("true", "false"):
        raise TickControlError(f"expected true or false, got {value!r}")
    return value == "true"


class TickControlClient:
    def __init__(self, connection: socket.socket) -> None:
        self.connection = connection
        self.reader = connection.makefile("r", encoding="utf-8")

    @classmethod
    def connect(cls, port: int, timeout: float = 10.0) -> Self:
        return cls(socket.create_connection(("127.0.0.1", port), timeout=timeout))

    def close(self) -> None:
        self.reader.close()
        self.connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def request(self, command: str) -> list[str]:
        self.connection.sendall(encode_request(command))
        return parse_reply(command, self.reader.readline())

    def status(self) -> Status:
        game_time, server_tick, frozen, paused = self.request("STATUS")
        return Status(
            int(game_time), int(server_tick), _parse_bool(frozen), _parse_bool(paused)
        )

    def step(self) -> Step:
        step_id, tick_before, tick_after = self.request("STEP")
        return Step(int(step_id), int(tick_before), int(tick_after))

    def quit(self) -> None:
        self.request("QUIT")

    def debug_spawn(self) -> None:
        self.request("DEBUG_SPAWN")

    def debug_probe(self) -> ProbeSnapshot:
        game_time, stand_y, husk_x, husk_z, distance = self.request("DEBUG_PROBE")
        return ProbeSnapshot(
            int(game_time),
            float(stand_y),
            float(husk_x),
            float(husk_z),
            float(distance),
        )
