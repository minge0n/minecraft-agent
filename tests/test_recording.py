import io
import json
import socket
import threading

import pytest

from minecraft_rl.recording import (
    Frame,
    Recorder,
    StreamError,
    StreamStart,
    encode_end,
    encode_frame,
    encode_start,
    episode_directory,
    finalize_episode,
    read_stream,
    storage_summary,
    write_episode,
)

WIDTH, HEIGHT = 32, 18


def frame(tick: int, shade: int) -> Frame:
    return Frame(
        tick, 40 + tick, 1.0, tick * 1000, bytes([shade]) * (WIDTH * HEIGHT * 4)
    )


def stream_bytes(ticks: list[int], end: bool = True) -> bytes:
    data = encode_start(StreamStart(3, ticks[0], 2, WIDTH, HEIGHT))
    for index, tick in enumerate(ticks):
        data += encode_frame(frame(tick, 40 * index % 256))
    return data + (encode_end() if end else b"")


def test_stream_round_trip():
    messages = list(read_stream(io.BytesIO(stream_bytes([10, 12]))))
    assert messages[0] == StreamStart(3, 10, 2, WIDTH, HEIGHT)
    assert [m.client_tick for m in messages[1:3]] == [10, 12]
    assert messages[1].rgba == frame(10, 0).rgba
    assert messages[3] is None


def test_truncated_stream_is_an_error():
    with pytest.raises(EOFError):
        list(read_stream(io.BytesIO(stream_bytes([10, 12], end=False))))


def test_bad_magic_is_rejected():
    with pytest.raises(StreamError):
        list(read_stream(io.BytesIO(b"\x00" * 7)))


def test_write_episode_encodes_video_and_frame_map(tmp_path):
    result = write_episode(io.BytesIO(stream_bytes([10, 12, 16])), tmp_path)
    assert result.status == "complete", result.error
    assert result.episode == 3
    assert result.frames_written == 3
    assert result.non_consecutive_frames == 1
    assert (tmp_path / "video.mp4").stat().st_size == result.video_bytes > 0
    lines = (tmp_path / "frames.jsonl").read_text().splitlines()
    assert [json.loads(line)["client_tick"] for line in lines] == [10, 12, 16]
    assert json.loads((tmp_path / "metrics.json").read_text())["status"] == "complete"


def test_write_episode_marks_truncated_stream_failed(tmp_path):
    result = write_episode(io.BytesIO(stream_bytes([10, 12], end=False)), tmp_path)
    assert result.status == "failed"
    assert "EOFError" in result.error
    assert result.frames_written == 2
    assert json.loads((tmp_path / "metrics.json").read_text())["status"] == "failed"


def test_recorder_accepts_one_episode_per_connection(tmp_path):
    recorder = Recorder()
    try:
        for episode in range(2):
            directory = episode_directory(tmp_path, "env-0000", episode)
            recorder.next_episode(directory)
            with socket.create_connection(("127.0.0.1", recorder.port)) as sender:
                sender.sendall(stream_bytes([episode * 100, episode * 100 + 2]))
            result = recorder.finish_episode(timeout=10)
            assert result.status == "complete", result.error
            assert (directory / "video.mp4").exists()
    finally:
        recorder.close()


def test_recorder_cannot_start_overlapping_episodes(tmp_path):
    recorder = Recorder()
    try:
        recorder.next_episode(tmp_path / "a")
        with pytest.raises(RuntimeError):
            recorder.next_episode(tmp_path / "b")
        threading.Thread(
            target=lambda: socket.create_connection(
                ("127.0.0.1", recorder.port)
            ).sendall(stream_bytes([0])),
            daemon=True,
        ).start()
        assert recorder.finish_episode(timeout=10).status == "complete"
    finally:
        recorder.close()


def test_finalize_episode_fails_when_capture_failed(tmp_path):
    recording = write_episode(io.BytesIO(stream_bytes([10, 12])), tmp_path)
    capture = {"status": "failed", "start_client_tick": 10, "stop_client_tick": 13}
    metrics = finalize_episode(recording, capture, {"worker": "env-0000"})
    assert metrics["recording_status"] == "failed"
    assert metrics["simulated_ticks"] == 3
    stored = json.loads((tmp_path / "metrics.json").read_text())
    assert stored["worker"] == "env-0000"
    assert stored["encoding"]["frames_written"] == 2


def test_storage_summary_scales_to_simulated_hour():
    class Fake:
        video_bytes = 1000

    summary = storage_summary([Fake(), Fake()], simulated_ticks=7200)
    assert summary["bytes_per_episode"] == 1000
    assert summary["bytes_per_simulated_hour"] == 20_000
