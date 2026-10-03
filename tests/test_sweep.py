from pathlib import Path

from minecraft_rl.sweep import TASKPOLICY, command, parse_seeds


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
