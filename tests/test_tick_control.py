import socket

import pytest

from minecraft_rl.tick_control import (
    PlayerSnapshot,
    ProbeSnapshot,
    Status,
    Step,
    TickControlClient,
    TickControlError,
    encode_request,
    parse_reply,
)


@pytest.fixture
def connected():
    client_side, server_side = socket.socketpair()
    server_side.settimeout(1)
    client = TickControlClient(client_side)
    yield client, server_side
    client.close()
    server_side.close()


def test_encode_request_uses_versioned_line() -> None:
    assert encode_request("STEP NOOP") == b"v1 STEP NOOP\n"


def test_parse_reply_returns_payload_fields() -> None:
    assert parse_reply("STEP", "v1 STEP 3 7 8 12\n") == ["3", "7", "8", "12"]


def test_parse_reply_matches_command_word_not_arguments() -> None:
    assert parse_reply("STEP FORWARD", "v1 STEP 3 7 8 12\n") == ["3", "7", "8", "12"]


def test_parse_reply_rejects_error_mismatch_and_eof() -> None:
    with pytest.raises(TickControlError, match="not_frozen"):
        parse_reply("STEP", "v1 ERROR not_frozen\n")
    with pytest.raises(TickControlError, match="unexpected reply"):
        parse_reply("STEP", "v1 STATUS 1 2 true false true 3\n")
    with pytest.raises(ConnectionError):
        parse_reply("STEP", "")


def test_status_sends_request_and_parses_reply(connected) -> None:
    client, server = connected
    server.sendall(b"v1 STATUS 7 90 true false true 12\n")
    assert client.status() == Status(7, 90, True, False, True, 12)
    assert server.recv(64) == b"v1 STATUS\n"


def test_status_rejects_invalid_boolean(connected) -> None:
    client, server = connected
    server.sendall(b"v1 STATUS 7 90 yes false true 12\n")
    with pytest.raises(TickControlError, match="true or false"):
        client.status()


def test_step_sends_action_and_parses_ticks(connected) -> None:
    client, server = connected
    server.sendall(b"v1 STEP 3 7 8 12\n")
    assert client.step("FORWARD") == Step(3, 7, 8, 12)
    assert server.recv(64) == b"v1 STEP FORWARD\n"


def test_step_defaults_to_noop_and_rejects_unknown_action(connected) -> None:
    client, server = connected
    server.sendall(b"v1 STEP 1 0 1 1\n")
    assert client.step() == Step(1, 0, 1, 1)
    assert server.recv(64) == b"v1 STEP NOOP\n"
    with pytest.raises(ValueError):
        client.step("JUMP")


def test_debug_probes_parse_numeric_fields(connected) -> None:
    client, server = connected
    server.sendall(
        b"v1 DEBUG_PROBE 8 -50.5 2.5 -3.5 6.0\n"
        b"v1 DEBUG_PLAYER 8 0.5 1.25 30 30 30 0.5 1.25 30\n"
    )
    assert client.debug_probe() == ProbeSnapshot(8, -50.5, 2.5, -3.5, 6.0)
    assert client.debug_player() == PlayerSnapshot(
        8, 0.5, 1.25, 30, 30, 30, 0.5, 1.25, 30
    )
    assert server.recv(64) == b"v1 DEBUG_PROBE\nv1 DEBUG_PLAYER\n"
