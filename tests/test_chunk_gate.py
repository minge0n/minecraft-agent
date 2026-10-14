"""Tests of the chunk gate that lets the game run on the efficiency cores."""

import pytest

from minecraft_rl import chunk_gate
from minecraft_rl.chunk_gate import ChunkGate, chunks_ready

READY = {
    "sensor_chunks": 13,
    "sensor_chunks_loaded": 13,
    "sensor_chunks_entity_ticking": 13,
    "simulation_chunks": 169,
    "simulation_chunks_entity_ticking": 169,
}


class Source:
    """Not ready for the first `pending` queries, then ready."""

    def __init__(self, pending: int) -> None:
        self.pending = pending
        self.queries = 0

    def chunks(self):
        self.queries += 1
        if self.queries <= self.pending:
            return READY | {
                "sensor_chunks_loaded": 11,
                "simulation_chunks_entity_ticking": 81,
            }
        return READY


def test_every_chunk_count_must_be_complete():
    assert chunks_ready(READY)
    assert not chunks_ready(READY | {"sensor_chunks_loaded": 12})
    assert not chunks_ready(READY | {"sensor_chunks_entity_ticking": 12})
    assert not chunks_ready(READY | {"simulation_chunks_entity_ticking": 168})


def test_ready_chunks_pass_without_waiting():
    gate = ChunkGate(Source(0), pid=None)
    gate.wait()
    assert gate.summary()["waits"] == 0 and gate.checks == 1


def test_the_gate_waits_and_boosts_then_returns_to_background(monkeypatch):
    calls = []
    monkeypatch.setattr(chunk_gate, "set_background", lambda pid, bg: calls.append(bg))
    gate = ChunkGate(Source(5), pid=123, boost_after=0.0, poll=0.0)
    gate.wait()
    summary = gate.summary()
    assert summary["waits"] == 1 and summary["boosts"] == 1
    # Boost to default priority once, then back to the background policy.
    assert calls == [False, True]


def test_the_gate_gives_up_after_the_timeout(monkeypatch):
    calls = []
    monkeypatch.setattr(chunk_gate, "set_background", lambda pid, bg: calls.append(bg))
    gate = ChunkGate(Source(10**9), pid=123, boost_after=0.0, timeout=0.01, poll=0.0)
    with pytest.raises(RuntimeError, match="not ready"):
        gate.wait()
    assert calls[-1] is True
