"""Tests of the client renderer selection and the Sodium Extra config."""

import json

import pytest

from minecraft_rl.minecraft_launch import (
    DEFAULT_RENDERER,
    client_game_directory,
    renderer_of,
    worker_game_directory,
    write_renderer_config,
)


def test_the_default_renderer_is_vanilla_and_others_are_named():
    assert DEFAULT_RENDERER == "vanilla"
    assert renderer_of({}) == "vanilla"
    assert renderer_of({"MCBOT_RENDERER": "sodium-extra"}) == "sodium-extra"
    with pytest.raises(ValueError, match="MCBOT_RENDERER"):
        renderer_of({"MCBOT_RENDERER": "iris"})


def test_the_config_turns_on_reduce_resolution_and_keeps_other_settings(tmp_path):
    path = tmp_path / "config" / "sodium-extra-options.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"extra_settings": {"show_fps": True}, "x": 1}))
    write_renderer_config(tmp_path, "sodium-extra")
    data = json.loads(path.read_text())
    assert data["extra_settings"] == {
        "show_fps": True,
        "reduce_resolution_on_mac": True,
    }
    assert data["x"] == 1


def test_other_renderers_write_no_config(tmp_path):
    for renderer in ("vanilla", "sodium"):
        write_renderer_config(tmp_path, renderer)
    assert not (tmp_path / "config").exists()


def test_the_config_goes_to_the_project_owned_game_directory():
    assert client_game_directory(None).parts[-3:] == (".runtime", "minecraft", "client")
    assert client_game_directory(7) == worker_game_directory(7)
