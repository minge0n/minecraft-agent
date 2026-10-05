import numpy
import pytest

from minecraft_rl.minecraft_dataset import (
    ACTION_BUTTONS,
    EpisodeMeta,
    EpisodeWriter,
    episode_directories,
    load_episode,
    load_episodes,
)
from minecraft_rl.minecraft_interface import PlayerAction, PolicyObservation
from test_minecraft_interface import SCHEMA, observation_json


def observation(**overrides) -> PolicyObservation:
    return PolicyObservation.from_json(observation_json(**overrides), SCHEMA)


def meta(steps: int, split: str = "train", episode: int = 0) -> EpisodeMeta:
    return EpisodeMeta(
        episode=episode,
        world_seed=100,
        preset="flat",
        level_id="x",
        split=split,
        policy_seed=1,
        steps=steps,
        terminated=False,
        truncated=True,
        first_tick=0,
        last_tick=steps,
    )


def written(tmp_path, steps=3, split="train", episode=0):
    writer = EpisodeWriter(observation())
    for t in range(steps):
        writer.add(
            PlayerAction(forward=True, yaw_delta=float(t), hotbar=t % 2 - 1),
            observation(food=19 - t, pitch=float(t)),
            terminated=t == steps - 1,
            step_id=t + 1,
            tick_before=t,
            tick_after=t + 1,
            client_tick=10 + t,
        )
    directory = tmp_path / split / "episodes" / f"{episode:06d}"
    writer.write(directory, meta(steps, split, episode))
    return directory


def test_an_episode_round_trips_with_aligned_time_axes(tmp_path):
    episode = load_episode(written(tmp_path))
    a = episode.arrays
    assert a["obs_ray_kind"].shape == (4, SCHEMA.rays)
    assert a["obs_inventory_item"].shape == (4, 36)
    assert a["obs_food"].tolist() == [20, 19, 18, 17]
    assert a["action_buttons"].shape == (3, len(ACTION_BUTTONS))
    assert a["action_buttons"][:, ACTION_BUTTONS.index("forward")].all()
    assert a["action_camera"][:, 0].tolist() == [0.0, 1.0, 2.0]
    assert a["action_hotbar"].tolist() == [-1, 0, -1]
    assert a["terminated"].tolist() == [False, False, True]
    assert a["client_tick"].tolist() == [10, 11, 12]
    assert a["obs_ray_kind"].dtype == numpy.int8
    assert a["obs_ray_distance"].dtype == numpy.float16


def test_only_complete_episodes_are_listed_and_splits_filter(tmp_path):
    written(tmp_path, split="train", episode=0)
    written(tmp_path, split="train", episode=1)
    (tmp_path / "train" / "episodes" / "000002").mkdir(parents=True)
    assert len(episode_directories(tmp_path / "train")) == 2
    assert [e.meta.episode for e in load_episodes(tmp_path / "train", "train")] == [
        0,
        1,
    ]
    assert load_episodes(tmp_path / "train", "eval_seed") == []


def test_a_mismatched_step_count_is_refused(tmp_path):
    directory = written(tmp_path)
    text = (directory / "episode.json").read_text().replace('"steps": 3', '"steps": 4')
    (directory / "episode.json").write_text(text)
    with pytest.raises(ValueError, match="step count"):
        load_episode(directory)
