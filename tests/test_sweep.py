from pathlib import Path

from minecraft_rl import sweep
from minecraft_rl.sweep import TASKPOLICY, command, parse_seeds, thermal_pressure


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


def test_wait_blocks_until_the_level_drops(monkeypatch):
    levels = iter([2, 2, 0])
    monkeypatch.setattr(sweep, "thermal_pressure", lambda: next(levels))
    monkeypatch.setattr(sweep.time, "sleep", lambda seconds: None)
    sweep.wait_for_cool_system(max_level=0, poll_seconds=1.0)
    assert next(levels, "exhausted") == "exhausted"
