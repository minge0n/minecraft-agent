import json
import socket

import pytest

from minecraft_rl.minecraft_client import (
    MinecraftClient,
    ProtocolError,
    encode_request,
    parse_reply,
)
from minecraft_rl.minecraft_interface import PlayerAction, PolicyObservation
from test_minecraft_interface import SCHEMA_JSON, observation_json

TIMING = {
    "dispatch_ms": 0.1,
    "client_wait_ms": 1.0,
    "client_tick_ms": 2.0,
    "tick_end_sync_ms": 3.0,
    "server_wait_ms": 0.5,
    "server_tick_ms": 4.0,
    "observation_ms": 5.0,
    "encode_ms": 0.4,
    "server_total_ms": 16.0,
}


def reply(command: str, payload: dict) -> bytes:
    return f"v2 {command} {json.dumps(payload)}\n".encode()


@pytest.fixture
def connected():
    client_side, server_side = socket.socketpair()
    server_side.settimeout(1)
    client = MinecraftClient(client_side)
    yield client, server_side
    client.close()
    server_side.close()


def received_lines(server: socket.socket) -> list[str]:
    return server.recv(65536).decode().splitlines()


def test_encode_request_uses_v2_prefix_and_compact_json() -> None:
    assert encode_request("OBSERVE") == b"v2 OBSERVE\n"
    assert encode_request("RESET", {"seed": 7}) == b'v2 RESET {"seed":7}\n'


def test_parse_reply_rejects_errors_mismatches_and_eof() -> None:
    assert parse_reply("STATUS", 'v2 STATUS {"a": 1}\n') == {"a": 1}
    with pytest.raises(ProtocolError, match="step_pending"):
        parse_reply("STEP", "v2 ERROR step_pending\n")
    with pytest.raises(ProtocolError, match="unexpected reply"):
        parse_reply("STEP", "v2 OBSERVE {}\n")
    with pytest.raises(ProtocolError, match="version"):
        parse_reply("STEP", "v1 STEP 1 2 3 4\n")
    with pytest.raises(ConnectionError):
        parse_reply("STEP", "")


def test_step_separates_policy_observation_from_diagnostics(connected) -> None:
    client, server = connected
    server.sendall(
        reply("SCHEMA", SCHEMA_JSON)
        + reply(
            "STEP",
            {
                "observation": observation_json(),
                "terminated": False,
                "info": {
                    "game_time": 41,
                    "step_id": 3,
                    "tick_before": 40,
                    "tick_after": 41,
                    "client_tick": 90,
                    "pacing": "unpaced",
                    "timing": TIMING,
                },
            },
        )
    )
    result = client.step(PlayerAction(forward=True, yaw_delta=10.0))

    assert isinstance(result.observation, PolicyObservation)
    assert (result.info.tick_before, result.info.tick_after) == (40, 41)
    assert result.info.timing.server_tick_ms == 4.0
    assert result.info.pacing == "unpaced"
    assert result.info.timing.ipc_ms == pytest.approx(
        result.info.timing.round_trip_ms - 16.0
    )
    assert result.info.timing.round_trip_ms >= 0.0
    schema_line, step_line = received_lines(server)
    assert schema_line == "v2 SCHEMA"
    assert json.loads(step_line.removeprefix("v2 STEP "))["yaw_delta"] == 10.0


def test_observation_with_privileged_field_is_rejected(connected) -> None:
    client, server = connected
    leaked = observation_json(player_x=12.5)
    server.sendall(
        reply("SCHEMA", SCHEMA_JSON) + reply("OBSERVE", {"observation": leaked})
    )
    with pytest.raises(ValueError, match="fields differ"):
        client.observe()


def test_reset_rejects_unknown_preset_before_sending(connected) -> None:
    client, _ = connected
    with pytest.raises(ValueError, match="preset"):
        client.reset(1, "amplified")


def test_pacing_rejects_unknown_mode_and_sends_valid_ones(connected) -> None:
    client, server = connected
    with pytest.raises(ValueError, match="pacing"):
        client.set_pacing("sprint")
    server.sendall(reply("PACING", {"mode": "unpaced", "render_frames": False}))
    assert client.set_pacing("unpaced", render_frames=False)["mode"] == "unpaced"
    assert received_lines(server) == [
        'v2 PACING {"mode":"unpaced","render_frames":false}'
    ]


def test_privileged_probe_uses_debug_commands(connected) -> None:
    client, server = connected
    server.sendall(reply("DEBUG_BLOCK", {"pos": [1, 2, 3], "block": "minecraft:stone"}))
    assert client.privileged().block([1, 2, 3]) == "minecraft:stone"
    assert received_lines(server) == ['v2 DEBUG_BLOCK {"pos":[1,2,3]}']
