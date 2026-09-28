"""Session recording: receive rendered frames from the mod and encode episode videos.

Recording is research instrumentation for humans (docs/decisions/recording.md). Frames
never reach the policy, and nothing here can change a step: the mod captures frames
asynchronously and drops them rather than wait, and the recorder runs in its own
process. Each episode produces `video.mp4` (H.264, macOS VideoToolbox encoder), a
`frames.jsonl` map from video frame to client tick, and `metrics.json`.

The frame stream format mirrors `minecraft/fabric-mod/.../FrameStream.java`.
"""

import json
import os
import socket
import struct
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

MAGIC = 0x4D434652
VERSION = 1
START, FRAME, END = 0, 1, 2
HEADER = struct.Struct(">IhB")
START_BODY = struct.Struct(">qqiii")
FRAME_BODY = struct.Struct(">qqfqi")

ENCODER = "h264_videotoolbox"
PIXEL_FORMAT = "nv12"
PLAYBACK_FPS = 20
BIT_RATE = 600_000


class StreamError(RuntimeError):
    pass


@dataclass(frozen=True)
class StreamStart:
    episode: int
    start_client_tick: int
    every_ticks: int
    width: int
    height: int


@dataclass(frozen=True)
class Frame:
    client_tick: int
    game_time: int
    partial_tick: float
    capture_nanos: int
    rgba: bytes


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            raise EOFError("frame stream ended mid-message")
        data += chunk
    return bytes(data)


def _read_header(stream: BinaryIO) -> int | None:
    first = stream.read(1)
    if not first:
        return None
    magic, version, kind = HEADER.unpack(first + _read_exact(stream, HEADER.size - 1))
    if magic != MAGIC:
        raise StreamError(f"bad magic {magic:#x}")
    if version != VERSION:
        raise StreamError(f"unsupported frame stream version {version}")
    return kind


def read_stream(stream: BinaryIO) -> Iterator[StreamStart | Frame | None]:
    """Yield the START message, then frames, then `None` on END.

    A stream that stops before END raises EOFError, so a truncated recording is
    never mistaken for a complete one.
    """
    if _read_header(stream) != START:
        raise StreamError("frame stream must begin with START")
    yield StreamStart(*START_BODY.unpack(_read_exact(stream, START_BODY.size)))
    while True:
        kind = _read_header(stream)
        if kind is None:
            raise EOFError("frame stream ended without END")
        if kind == END:
            yield None
            return
        if kind != FRAME:
            raise StreamError(f"unknown message type {kind}")
        tick, game_time, partial, nanos, length = FRAME_BODY.unpack(
            _read_exact(stream, FRAME_BODY.size)
        )
        yield Frame(tick, game_time, partial, nanos, _read_exact(stream, length))


def encode_start(start: StreamStart) -> bytes:
    return HEADER.pack(MAGIC, VERSION, START) + START_BODY.pack(
        start.episode,
        start.start_client_tick,
        start.every_ticks,
        start.width,
        start.height,
    )


def encode_frame(frame: Frame) -> bytes:
    return (
        HEADER.pack(MAGIC, VERSION, FRAME)
        + FRAME_BODY.pack(
            frame.client_tick,
            frame.game_time,
            frame.partial_tick,
            frame.capture_nanos,
            len(frame.rgba),
        )
        + frame.rgba
    )


def encode_end() -> bytes:
    return HEADER.pack(MAGIC, VERSION, END)


@dataclass
class EpisodeRecording:
    """What the recorder wrote for one episode; `status` is `complete` or `failed`."""

    episode: int
    directory: str
    status: str = "pending"
    error: str | None = None
    start_client_tick: int | None = None
    every_ticks: int | None = None
    width: int | None = None
    height: int | None = None
    frames_written: int = 0
    first_client_tick: int | None = None
    last_client_tick: int | None = None
    non_consecutive_frames: int = 0
    video_bytes: int = 0
    encode_seconds: float = 0.0
    wall_seconds: float = 0.0
    encoder: str = ENCODER
    playback_fps: int = PLAYBACK_FPS
    bit_rate: int = BIT_RATE
    frame_ticks: list[int] = field(default_factory=list, repr=False)

    def summary(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("frame_ticks")
        return data


def write_episode(stream: BinaryIO, directory: Path) -> EpisodeRecording:
    """Encode one frame stream into `directory`; failures are recorded, not raised."""
    import av  # Imported lazily: only recorder processes need the video codec.

    directory.mkdir(parents=True, exist_ok=True)
    video_path = directory / "video.mp4"
    started = time.perf_counter()
    result = EpisodeRecording(episode=-1, directory=str(directory))
    container = None
    video_stream = None
    try:
        with (directory / "frames.jsonl").open("w", encoding="utf-8") as frame_log:
            for message in read_stream(stream):
                if isinstance(message, StreamStart):
                    result.episode = message.episode
                    result.start_client_tick = message.start_client_tick
                    result.every_ticks = message.every_ticks
                    result.width, result.height = message.width, message.height
                    container = av.open(str(video_path), "w")
                    video_stream = container.add_stream(ENCODER, rate=PLAYBACK_FPS)
                    video_stream.width, video_stream.height = (
                        message.width,
                        message.height,
                    )
                    video_stream.pix_fmt = PIXEL_FORMAT
                    video_stream.bit_rate = BIT_RATE
                    continue
                if message is None:
                    break
                encode_started = time.perf_counter()
                image = av.VideoFrame.from_bytes(
                    message.rgba,
                    result.width,
                    result.height,
                    format="rgba",
                    flip_vertical=True,
                )
                image.pts = result.frames_written
                for packet in video_stream.encode(image.reformat(format=PIXEL_FORMAT)):
                    container.mux(packet)
                result.encode_seconds += time.perf_counter() - encode_started
                if (
                    result.last_client_tick is not None
                    and message.client_tick
                    != result.last_client_tick + result.every_ticks
                ):
                    result.non_consecutive_frames += 1
                frame_log.write(
                    json.dumps(
                        {
                            "video_frame": result.frames_written,
                            "client_tick": message.client_tick,
                            "game_time": message.game_time,
                            "partial_tick": message.partial_tick,
                        }
                    )
                    + "\n"
                )
                result.frame_ticks.append(message.client_tick)
                result.first_client_tick = (
                    message.client_tick
                    if result.first_client_tick is None
                    else result.first_client_tick
                )
                result.last_client_tick = message.client_tick
                result.frames_written += 1
        result.status = "complete"
    except Exception as error:  # noqa: BLE001 - every failure is reported in metadata.
        result.status = "failed"
        result.error = f"{type(error).__name__}: {error}"
    finally:
        if container is not None:
            try:
                for packet in video_stream.encode():
                    container.mux(packet)
                container.close()
            except Exception as error:  # noqa: BLE001
                result.status = "failed"
                result.error = result.error or f"{type(error).__name__}: {error}"
    result.wall_seconds = time.perf_counter() - started
    if video_path.exists():
        result.video_bytes = video_path.stat().st_size
    (directory / "metrics.json").write_text(
        json.dumps(result.summary(), indent=2) + "\n", encoding="utf-8"
    )
    return result


class Recorder:
    """A loopback server that records one episode per accepted connection.

    The mod connects on RECORD_START; `next_episode(directory)` must be called first
    so the connection is written to the right place. Encoding runs on a background
    thread, so a slow encoder backs up only the mod's bounded frame queue, never the
    simulation.
    """

    def __init__(self) -> None:
        self._listener = socket.create_server(("127.0.0.1", 0))
        self.port = self._listener.getsockname()[1]
        self._thread: threading.Thread | None = None
        self._result: EpisodeRecording | None = None

    def next_episode(self, directory: Path) -> None:
        if self._thread is not None:
            raise RuntimeError("previous episode not finished")
        self._result = None
        self._thread = threading.Thread(
            target=self._accept_one, args=(directory,), daemon=True
        )
        self._thread.start()

    def _accept_one(self, directory: Path) -> None:
        try:
            connection, _ = self._listener.accept()
        except OSError as error:
            self._result = EpisodeRecording(
                episode=-1, directory=str(directory), status="failed", error=str(error)
            )
            return
        with connection, connection.makefile("rb", buffering=1 << 20) as stream:
            self._result = write_episode(stream, directory)

    def finish_episode(self, timeout: float = 60.0) -> EpisodeRecording:
        """Wait for the episode's encoder to finish writing."""
        if self._thread is None:
            raise RuntimeError("no episode in progress")
        self._thread.join(timeout)
        thread, self._thread = self._thread, None
        if thread.is_alive() or self._result is None:
            return EpisodeRecording(
                episode=-1, directory="", status="failed", error="encoder_timeout"
            )
        return self._result

    def close(self) -> None:
        self._listener.close()


def storage_summary(
    recordings: list[EpisodeRecording], simulated_ticks: int
) -> dict[str, Any]:
    """Video storage per episode and per simulated hour (72,000 ticks at 20 TPS)."""
    total = sum(recording.video_bytes for recording in recordings)
    return {
        "episodes": len(recordings),
        "video_bytes": total,
        "bytes_per_episode": total / len(recordings) if recordings else 0.0,
        "simulated_ticks": simulated_ticks,
        "bytes_per_simulated_hour": total * 72_000 / simulated_ticks
        if simulated_ticks
        else 0.0,
    }


def episode_directory(run: Path, worker: str, episode: int) -> Path:
    return run / "workers" / worker / "episodes" / f"{episode:06d}"


def finalize_episode(
    recording: EpisodeRecording, capture: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    """Write the episode's `metrics.json`: session context, capture and encoding.

    The episode counts as recorded only if the mod-side capture and the encoder both
    completed; either failure marks the whole recording failed.
    """
    failed = capture.get("status") != "complete" or recording.status != "complete"
    metrics = context | {
        "recording_status": "failed" if failed else "complete",
        "simulated_ticks": capture["stop_client_tick"] - capture["start_client_tick"],
        "capture": capture,
        "encoding": recording.summary(),
    }
    write_json(Path(recording.directory) / "metrics.json", metrics)
    return metrics


def git_commit() -> str | None:
    head = Path(__file__).resolve().parents[2] / ".git" / "HEAD"
    try:
        ref = head.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not ref.startswith("ref: "):
        return ref
    target = head.parent / ref.removeprefix("ref: ")
    try:
        return target.read_text(encoding="utf-8").strip()
    except OSError:
        packed = head.parent / "packed-refs"
        name = ref.removeprefix("ref: ")
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line.endswith(" " + name):
                return line.split(" ", 1)[0]
    return None


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
