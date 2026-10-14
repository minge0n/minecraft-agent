"""Tests of the diagnostic movement reward `reward-control-v1`."""

import numpy
import pytest

from minecraft_rl.minecraft_dataset import EpisodeMeta, EpisodeWriter, load_episode
from minecraft_rl.minecraft_interface import PlayerAction, PolicyObservation
from minecraft_rl.minecraft_reward_control import (
    CONTROL_V1,
    POSITION_KEY,
    YAW_KEY,
    ControlTask,
    control_rewards,
    episode_task,
    facing,
    target_of,
)
from test_minecraft_interface import SCHEMA, observation_json

START = numpy.array([192.5, 112.5])
# Yaw 0 faces south (+z): the target is 20 blocks further along z.
TARGET = target_of(START, 0.0, CONTROL_V1)


def path(*steps):
    """Positions from the start, each step a (dx, dz) move."""
    points = [START]
    for dx, dz in steps:
        points.append(points[-1] + numpy.array([dx, dz]))
    return numpy.stack(points)


def test_the_target_lies_ahead_along_the_start_facing():
    assert TARGET == pytest.approx([192.5, 132.5])
    assert facing(90.0) == pytest.approx((-1.0, 0.0))
    assert target_of(START, 90.0, CONTROL_V1) == pytest.approx([172.5, 112.5])


def test_forward_progress_is_positive_and_backward_negative():
    rewards = control_rewards(path((0, 1), (0, -1)), TARGET, CONTROL_V1)
    assert rewards == pytest.approx([1.0, -1.0])


def test_standing_still_and_turning_in_place_earn_zero():
    # A camera turn does not move the player: the position is unchanged.
    rewards = control_rewards(path((0, 0), (0, 0), (0, 0)), TARGET, CONTROL_V1)
    assert (rewards == 0).all()


def test_returning_to_an_earlier_position_earns_zero_in_total():
    loop = path((0, 2), (1, 0), (0, -2), (-1, 0))
    rewards = control_rewards(loop, TARGET, CONTROL_V1)
    assert rewards.sum() == pytest.approx(0.0, abs=1e-6)
    shuttle = path(*[(0, 1), (0, -1)] * 20)
    assert control_rewards(shuttle, TARGET, CONTROL_V1).sum() == pytest.approx(
        0.0, abs=1e-5
    )


def test_identical_transitions_give_identical_rewards():
    a = control_rewards(path((0.3, 0.2)), TARGET, CONTROL_V1)
    b = control_rewards(path((0.3, 0.2)), TARGET, CONTROL_V1)
    assert a.tobytes() == b.tobytes()


def test_the_reward_does_not_depend_on_earlier_visits():
    # The same transition, once after a detour and once directly.
    detour = control_rewards(path((5, 0), (-5, 0), (0, 1)), TARGET, CONTROL_V1)
    direct = control_rewards(path((0, 1)), TARGET, CONTROL_V1)
    assert detour[-1] == pytest.approx(direct[0])


def test_the_scale_multiplies_the_progress():
    task = ControlTask(scale=3.0)
    assert control_rewards(path((0, 1)), target_of(START, 0.0, task), task) == (
        pytest.approx([3.0])
    )


def test_positions_are_stored_apart_from_the_observation(tmp_path):
    first = PolicyObservation.from_json(observation_json(), SCHEMA)
    writer = EpisodeWriter(first)
    positions = path((0, 1), (0, 1))
    writer.add_privileged("position", positions[0].tolist())
    writer.add_privileged("yaw", 0.0)
    for t in range(2):
        writer.add(PlayerAction(forward=True), first, False, t, t, t + 1, t)
        writer.add_privileged("position", positions[t + 1].tolist())
        writer.add_privileged("yaw", 0.0)
    writer.write(
        tmp_path,
        EpisodeMeta(0, 100000, "normal", "x", "train", 1, 2, False, True, 0, 2),
    )
    episode = load_episode(tmp_path)
    assert not any(
        name.startswith("obs_") and "position" in name for name in episode.arrays
    )
    assert episode.arrays[POSITION_KEY].shape == (3, 2)
    assert episode.arrays[YAW_KEY].shape == (3,)
    task = episode_task(episode.arrays)
    assert task["return"] == pytest.approx(2.0)
    assert task["net_progress"] == pytest.approx(2.0)
    assert task["final_distance"] == pytest.approx(18.0)
    assert not task["reached"]


def test_a_privileged_series_needs_one_value_per_observation():
    first = PolicyObservation.from_json(observation_json(), SCHEMA)
    writer = EpisodeWriter(first)
    writer.add_privileged("position", [0.0, 0.0])
    writer.add(PlayerAction(), first, False, 0, 0, 1, 0)
    with pytest.raises(ValueError, match="one value per observation"):
        writer.arrays()
