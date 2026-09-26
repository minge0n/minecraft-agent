import socket
from dataclasses import dataclass
from typing import Self


class TickControlError(RuntimeError):
    pass


ACTIONS = ("NOOP", "FORWARD")


@dataclass(frozen=True)
class Status:
    game_time: int
    server_tick: int
    frozen: bool
    paused: bool
    client_gated: bool
    client_ticks: int


@dataclass(frozen=True)
class Step:
    step_id: int
    tick_before: int
    tick_after: int
    client_tick: int


@dataclass(frozen=True)
class ProbeSnapshot:
    """Privileged test-only state; never a policy observation."""

    game_time: int
    armor_stand_y: float
    husk_x: float
    husk_z: float
    husk_player_distance: float


@dataclass(frozen=True)
class PlayerSnapshot:
    """Privileged test-only server and client player state; not a policy input."""

    game_time: int
    server_x: float
    server_z: float
    server_tick_count: int
    server_play_time: int
    client_tick: int
    client_x: float
    client_z: float
    client_tick_count: int


def encode_request(command: str) -> bytes:
    return f"v1 {command}\n".encode()


def parse_reply(command: str, line: str) -> list[str]:
    if not line:
        raise ConnectionError("tick control closed the connection")
    fields = line.split()
    if fields[:2] == ["v1", "ERROR"]:
        raise TickControlError(f"{command} rejected: {' '.join(fields[2:])}")
    if fields[:2] != ["v1", command.split()[0]]:
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
        game_time, server_tick, frozen, paused, gated, client_ticks = self.request(
            "STATUS"
        )
        return Status(
            int(game_time),
            int(server_tick),
            _parse_bool(frozen),
            _parse_bool(paused),
            _parse_bool(gated),
            int(client_ticks),
        )

    def step(self, action: str = "NOOP") -> Step:
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}")
        step_id, tick_before, tick_after, client_tick = self.request(f"STEP {action}")
        return Step(int(step_id), int(tick_before), int(tick_after), int(client_tick))

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

    def debug_player(self) -> PlayerSnapshot:
        fields = self.request("DEBUG_PLAYER")
        (game_time, server_x, server_z, server_ticks, play_time, client_tick) = fields[
            :6
        ]
        client_x, client_z, client_ticks = fields[6:]
        return PlayerSnapshot(
            int(game_time),
            float(server_x),
            float(server_z),
            int(server_ticks),
            int(play_time),
            int(client_tick),
            float(client_x),
            float(client_z),
            int(client_ticks),
        )
