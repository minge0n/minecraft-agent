import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from minecraft_rl import sweep
from minecraft_rl.macos_thermal import CpuTemperature, thermal_pressure
from minecraft_rl.sweep import (
    TASKPOLICY,
    Reading,
    Settings,
    Thermostat,
    command,
    parse_seeds,
    pending_seeds,
)


def test_seed_ranges_and_lists():
    assert parse_seeds("0-3") == [0, 1, 2, 3]
    assert parse_seeds("5") == [5]
    assert parse_seeds("0-1,7,9-10") == [0, 1, 7, 9, 10]


def test_runs_go_through_taskpolicy_background():
    run = command("world_model", 3, Path("out/metrics.json"), ["--steps", "5"])
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


def test_macos_sensors_give_plausible_values():
    level = thermal_pressure()
    assert level is None or 0 <= level <= 4
    sensor = CpuTemperature()
    if sensor.available:
        assert 0.0 < sensor.read() < 150.0


def test_finished_seeds_are_skipped_unless_rerun(tmp_path):
    (tmp_path / "seed1").mkdir()
    (tmp_path / "seed1" / "metrics.json").write_text("{}")
    assert pending_seeds([0, 1, 2], tmp_path, rerun=False) == [0, 2]
    assert pending_seeds([0, 1, 2], tmp_path, rerun=True) == [0, 1, 2]


def test_thermostat_has_a_gap_between_pause_and_resume():
    thermostat = Thermostat(
        Settings(max_temperature=60.0, resume_temperature=57.0),
        temperature=lambda: None,
        pressure=lambda: None,
    )
    assert thermostat.too_hot(Reading(60.5, 0))
    assert not thermostat.too_hot(Reading(59.0, 0))
    assert thermostat.too_hot(Reading(50.0, 1))
    assert not thermostat.cool_enough(Reading(58.0, 0))
    assert thermostat.cool_enough(Reading(57.0, 0))
    assert not thermostat.cool_enough(Reading(50.0, 1))
    assert thermostat.cool_enough(Reading(None, None))


class FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid

    def poll(self) -> None:
        return None


def fake_run(pid: int) -> sweep.SeedRun:
    return sweep.SeedRun(0, [], FakeProcess(pid), None, 0.0, 0)


def patch_signals(monkeypatch) -> list[tuple[int, int]]:
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(sweep.os, "killpg", lambda pid, n: sent.append((pid, n)))
    monkeypatch.setattr(sweep.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(sweep.macos_energy, "read", lambda pid=None: None)
    return sent


def hot_settings() -> Settings:
    return Settings(duty_cycle=1.0, max_temperature=60.0, resume_temperature=57.0)


def test_a_hot_die_does_not_pause_seeds_that_make_little_heat(monkeypatch):
    sent = patch_signals(monkeypatch)
    run = fake_run(5)
    run.recent_watts = 0.7
    settings = hot_settings()
    thermostat = Thermostat(settings, temperature=lambda: 70.0, pressure=lambda: 0)
    sweep.throttle_period([run], settings, thermostat)
    assert sent == []
    assert run.skipped_pause_events == 1
    assert run.thermostat_pause_events == 0


def test_a_hot_die_pauses_seeds_that_make_real_heat(monkeypatch):
    sent = patch_signals(monkeypatch)
    temperatures = iter([70.0, 56.0])
    run = fake_run(5)
    run.recent_watts = 6.4
    settings = hot_settings()
    thermostat = Thermostat(
        settings, temperature=lambda: next(temperatures), pressure=lambda: 0
    )
    sweep.throttle_period([run], settings, thermostat)
    assert [n for _, n in sent] == [signal.SIGSTOP, signal.SIGCONT]
    assert run.thermostat_pause_events == 1


def test_high_thermal_pressure_pauses_even_cool_seeds(monkeypatch):
    sent = patch_signals(monkeypatch)
    pressures = iter([2, 0])
    run = fake_run(5)
    run.recent_watts = 0.1
    settings = hot_settings()
    thermostat = Thermostat(
        settings, temperature=lambda: 50.0, pressure=lambda: next(pressures)
    )
    sweep.throttle_period([run], settings, thermostat)
    assert [n for _, n in sent] == [signal.SIGSTOP, signal.SIGCONT]
    assert run.skipped_pause_events == 0


def test_cool_period_stops_then_resumes_for_the_duty_cycle(monkeypatch):
    sent = patch_signals(monkeypatch)
    runs = [fake_run(11), fake_run(12)]
    thermostat = Thermostat(Settings(), temperature=lambda: 50.0, pressure=lambda: 0)
    sweep.throttle_period(runs, Settings(duty_cycle=0.5), thermostat)
    assert sent == [
        (11, signal.SIGSTOP),
        (12, signal.SIGSTOP),
        (11, signal.SIGCONT),
        (12, signal.SIGCONT),
    ]
    assert runs[0].temperatures == [50.0]
    assert runs[0].thermostat_pause_events == 0


def test_hot_period_pauses_until_the_resume_temperature(monkeypatch):
    sent = patch_signals(monkeypatch)
    temperatures = iter([62.0, 59.0, 58.0, 56.5])
    runs = [fake_run(5)]
    settings = Settings(duty_cycle=1.0, max_temperature=60.0, resume_temperature=57.0)
    thermostat = Thermostat(
        settings, temperature=lambda: next(temperatures), pressure=lambda: 0
    )
    sweep.throttle_period(runs, settings, thermostat)
    assert [n for _, n in sent] == [signal.SIGSTOP, signal.SIGCONT]
    assert runs[0].thermostat_pause_events == 1
    assert runs[0].temperatures == [62.0]


def test_full_duty_cycle_on_a_cool_chip_sends_no_signal(monkeypatch):
    sent = patch_signals(monkeypatch)
    thermostat = Thermostat(Settings(), temperature=lambda: 50.0, pressure=lambda: 0)
    sweep.throttle_period([fake_run(5)], Settings(duty_cycle=1.0), thermostat)
    assert sent == []


def test_a_pause_ends_at_its_limit_on_a_chip_that_stays_warm(monkeypatch):
    sent = patch_signals(monkeypatch)
    clock = iter(range(0, 1000, 10))
    monkeypatch.setattr(sweep.time, "monotonic", lambda: float(next(clock)))
    settings = Settings(
        max_temperature=50.0, resume_temperature=48.0, max_pause_seconds=30.0
    )
    thermostat = Thermostat(settings, temperature=lambda: 57.0, pressure=lambda: 0)
    paused = sweep.cool_down([FakeProcess(5)], thermostat, thermostat.read())
    assert [n for _, n in sent] == [signal.SIGSTOP, signal.SIGCONT]
    assert 30.0 <= paused <= 50.0


@pytest.mark.parametrize("error", [ProcessLookupError, PermissionError])
def test_a_process_that_exits_before_the_signal_is_ignored(monkeypatch, error):
    def exited(pid, number):
        raise error

    monkeypatch.setattr(sweep.os, "killpg", exited)
    sweep.signal_all([FakeProcess(9)], signal.SIGSTOP)


def test_unfinished_processes_are_resumed_before_they_are_terminated():
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    sweep.signal_all([process], signal.SIGSTOP)
    sweep.stop_processes([process])
    assert process.returncode == -signal.SIGTERM


@pytest.mark.skipif(not sweep.TASKPOLICY.exists(), reason="macOS only")
def test_the_record_keeps_the_cpu_energy_of_a_finished_seed(tmp_path):
    command = [
        str(sweep.TASKPOLICY),
        "-b",
        sys.executable,
        "-c",
        "s = 0\nfor i in range(3_000_000): s += i",
    ]
    process = subprocess.Popen(command, start_new_session=True)
    run = sweep.SeedRun(0, command, process, None, time.monotonic(), None)
    while process.poll() is None:
        run.sample_energy()
        time.sleep(0.05)
    record = sweep.sweep_record("parity", run, sweep.Settings(), tmp_path)
    assert record["cpu_energy_joules"] > 0
    assert record["mean_cpu_power_watts_while_computing"] > 0
    # Background QoS keeps the work on the efficiency cores.
    assert record["performance_core_energy_joules"] < 0.5 * record["cpu_energy_joules"]
