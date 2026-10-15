"""Tests of the block-breaking diagnostic `reward-break-v1`."""

import numpy
import pytest
import torch

from minecraft_rl import runtime
from minecraft_rl.minecraft_actor import ActionSpace
from minecraft_rl.minecraft_break_train import attack_run_lengths
from minecraft_rl.minecraft_dataset import EpisodeMeta, EpisodeWriter, load_episode
from minecraft_rl.minecraft_interface import PlayerAction, PolicyObservation
from minecraft_rl.minecraft_reward_break import (
    ATTACK_FACTOR,
    BREAK_V1,
    BREAKS_KEY,
    BreakAgent,
    attack_runs,
    break_metrics,
    break_rewards,
)
from minecraft_rl.minecraft_reward_pitch import PITCH_FACTOR
from test_minecraft_interface import SCHEMA, observation_json


def test_every_real_break_pays_and_nothing_else_does():
    breaks = numpy.array([0, 0, 1, 0, 2, 0, 1])
    assert break_rewards(breaks).tolist() == [0, 0, 1, 0, 2, 0, 1]
    # The same break later in the episode pays the same: no memory.
    assert (
        break_rewards(numpy.array([1]))[0] == break_rewards(numpy.array([0, 0, 1]))[2]
    )


def test_attack_presses_without_a_break_earn_nothing():
    assert (break_rewards(numpy.zeros(20, dtype=int)) == 0).all()
    assert BREAK_V1.to_json() == {"version": "reward-break-v1", "per_break": 1.0}


def test_attack_runs_and_ticks_before_each_break():
    attack = numpy.array([0, 1, 1, 1, 0, 1, 1, 0, 1], dtype=bool)
    assert attack_runs(attack) == [3, 2, 1]
    breaks = numpy.array([0, 0, 0, 1, 0, 0, 1, 0, 0])
    m = break_metrics(breaks, attack, [7, 7])
    assert m["breaks"] == 2 and m["unique_types"] == 1
    assert m["attack_ticks_before_break"] == [3, 2]
    assert m["first_break_step"] == 3
    assert attack_run_lengths(attack).tolist() == [0, 1, 2, 3, 0, 1, 2, 0, 1]


def test_the_break_agent_chooses_only_pitch_and_attack():
    runtime.configure(0)
    agent = BreakAgent(16)
    choices = agent.act(torch.randn(256, 16), torch.Generator().manual_seed(0))
    for k, noop in enumerate(ActionSpace.NOOP):
        if k not in (PITCH_FACTOR, ATTACK_FACTOR):
            assert (choices[:, k] == noop).all()
    assert set(choices[:, ATTACK_FACTOR].tolist()) == {0, 1}
    action = ActionSpace.to_player_action(choices[0].tolist())
    assert not action.forward and not action.jump and not action.use
    assert action.yaw_delta == 0.0


def test_only_the_free_heads_learn():
    runtime.configure(0)
    agent = BreakAgent(16)
    s = torch.randn(32, 16)
    choices = agent.act(s, torch.Generator().manual_seed(0))
    log_prob, entropy, _ = agent.log_prob_entropy(s, choices)
    (log_prob.sum() + entropy.sum()).backward()
    for k, head in enumerate(agent.heads):
        learned = head.weight.grad is not None and head.weight.grad.abs().sum() > 0
        assert learned == (k in (PITCH_FACTOR, ATTACK_FACTOR))


def test_breaks_are_stored_apart_from_the_observation(tmp_path):
    first = PolicyObservation.from_json(observation_json(), SCHEMA)
    writer = EpisodeWriter(first)
    for t, n in enumerate((0, 1, 0)):
        writer.add(PlayerAction(attack=True), first, False, t, t, t + 1, t)
        writer.add_privileged_step("breaks", n)
    writer.write(
        tmp_path,
        EpisodeMeta(0, 100000, "normal", "x", "train", 1, 3, False, True, 0, 3),
    )
    episode = load_episode(tmp_path)
    assert episode.arrays[BREAKS_KEY].tolist() == [0, 1, 0]
    assert not any("break" in k for k in episode.arrays if k.startswith("obs_"))


def test_a_step_series_needs_one_value_per_step():
    first = PolicyObservation.from_json(observation_json(), SCHEMA)
    writer = EpisodeWriter(first)
    writer.add(PlayerAction(), first, False, 0, 0, 1, 0)
    writer.add_privileged_step("breaks", 0)
    writer.add_privileged_step("breaks", 0)
    with pytest.raises(ValueError, match="one value per step"):
        writer.arrays()
