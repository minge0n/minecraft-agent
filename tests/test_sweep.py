import signal
import subprocess
import sys
from pathlib import Path

from minecraft_rl import sweep
from minecraft_rl.sweep import (
    TASKPOLICY,
    command,
    parse_seeds,
    pending_seeds,
    thermal_pressure,
)


def test_seed_ranges_and_lists():
    assert parse_seeds("0-3") == [0, 1, 2, 3]
    assert parse_seeds("5") == [5]
    assert parse_seeds("0-1,7,9-10") == [0, 1, 7, 9, 10]


def test_background_runs_go_through_taskpolicy():
    run = command("world_model", 3, Path("out/metrics.json"), ["--steps", "5"], True)
    assert run[:2] == [str(TASKPOLICY), "-b"]
    assert run[3:] == [
        "-m",
        "minecraft_rl.world_model",
        "--seed",
        "3",
        "--output",
        "out/metrics.json",
        "--steps",
        "5",
    ]
    foreground = command("world_model", 3, Path("out/metrics.json"), [], False)
    assert foreground[1:3] == ["-m", "minecraft_rl.world_model"]


def test_thermal_pressure_is_a_known_level_or_unavailable():
    level = thermal_pressure()
    assert level is None or 0 <= level <= 4


def test_finished_seeds_are_skipped_unless_rerun(tmp_path):
    (tmp_path / "seed1").mkdir()
    (tmp_path / "seed1" / "metrics.json").write_text("{}")
    assert pending_seeds([0, 1, 2], tmp_path, rerun=False) == [0, 2]
    assert pending_seeds([0, 1, 2], tmp_path, rerun=True) == [0, 1, 2]


class FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid

    def poll(self) -> None:
        return None


def test_duty_cycle_stops_then_resumes_every_process(monkeypatch):
    sent: list[tuple[int, int]] = []
    sleeps: list[float] = []
    monkeypatch.setattr(
        sweep.os, "killpg", lambda pid, number: sent.append((pid, number))
    )
    monkeypatch.setattr(sweep.time, "sleep", sleeps.append)
    sweep.duty_cycle_period([FakeProcess(11), FakeProcess(12)], 0.25, 2.0)
    assert sleeps == [0.5, 1.5]
    assert sent == [
        (11, signal.SIGSTOP),
        (12, signal.SIGSTOP),
        (11, signal.SIGCONT),
        (12, signal.SIGCONT),
    ]


def test_hot_system_pauses_processes_until_it_cools(monkeypatch):
    levels = iter([2, 2, 2, 0])
    sent: list[int] = []
    monkeypatch.setattr(sweep, "thermal_pressure", lambda: next(levels))
    monkeypatch.setattr(sweep.os, "killpg", lambda pid, number: sent.append(number))
    monkeypatch.setattr(sweep.time, "sleep", lambda seconds: None)
    assert sweep.wait_for_cool_system([FakeProcess(5)], max_level=0)
    assert sent == [signal.SIGSTOP, signal.SIGCONT]


def test_cool_system_sends_no_signal(monkeypatch):
    sent: list[int] = []
    monkeypatch.setattr(sweep, "thermal_pressure", lambda: 0)
    monkeypatch.setattr(sweep.os, "killpg", lambda pid, number: sent.append(number))
    assert not sweep.wait_for_cool_system([FakeProcess(5)], max_level=0)
    assert sent == []


def test_unfinished_processes_are_resumed_before_they_are_terminated(tmp_path):
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    sweep.signal_all([process], signal.SIGSTOP)
    sweep.stop_processes([process])
    assert process.returncode == -signal.SIGTERM
