import math

import pytest
import torch

from minecraft_rl import world_model
from minecraft_rl.signal_tmaze import SignalTMaze
from minecraft_rl.stochastic_world import (
    entropy_bound,
    probe_episodes,
    signal_distribution,
    signal_negative_log_likelihood,
)
from minecraft_rl.tmaze import Action, Cue, Observation

CPU = torch.device("cpu")


def walk(environment: SignalTMaze, actions: list[Action]) -> list[tuple]:
    return [environment.step(action) for action in actions]


def test_signal_appears_after_the_first_forward_step():
    environment = SignalTMaze(corridor_length=2, signal_left_probability=1.0)
    assert environment.reset(torch.Generator().manual_seed(0)) == Observation.CORRIDOR
    steps = walk(environment, [Action.LEFT, Action.FORWARD, Action.FORWARD])
    assert [s[0] for s in steps] == [
        Observation.CORRIDOR,
        Observation.CUE_LEFT,
        Observation.CORRIDOR,
    ]
    assert environment.signal_drawn


@pytest.mark.parametrize(
    ("probability", "turn", "reward"),
    [(1.0, Action.LEFT, 1.0), (1.0, Action.RIGHT, -1.0), (0.0, Action.RIGHT, 1.0)],
)
def test_turn_reward_follows_the_drawn_signal(probability, turn, reward):
    environment = SignalTMaze(corridor_length=1, signal_left_probability=probability)
    environment.reset(torch.Generator().manual_seed(0))
    steps = walk(environment, [Action.FORWARD] * 3)
    assert steps[-1][0] == Observation.JUNCTION
    assert environment.step(turn) == (Observation.ARM, reward, True, False)


def test_signal_frequency_matches_the_probability_and_the_seed():
    environment = SignalTMaze(corridor_length=1, signal_left_probability=0.7)

    def draw(seed: int) -> list[Cue]:
        generator = torch.Generator().manual_seed(seed)
        cues = []
        for _ in range(4000):
            environment.reset(generator)
            environment.step(Action.FORWARD)
            cues.append(environment.cue)
        return cues

    first = draw(0)
    assert first == draw(0)
    assert sum(c == Cue.LEFT for c in first) / len(first) == pytest.approx(
        0.7, abs=0.02
    )


def test_collected_signal_episodes_hide_the_signal_until_it_is_drawn():
    config = world_model.Config(environment="signal", corridor_length=1, max_steps=12)
    episodes = world_model.collect_episodes(
        config, 64, torch.Generator().manual_seed(0)
    )
    assert torch.all(episodes.observations[:, 0] == Observation.CORRIDOR)
    first_forward = episodes.actions[:, 0] == Action.FORWARD
    signals = episodes.next_observations[first_forward, 0]
    assert set(signals.tolist()) <= {Observation.CUE_LEFT, Observation.CUE_RIGHT}
    assert torch.equal(
        signals == Observation.CUE_RIGHT, episodes.cues[first_forward] == Cue.RIGHT
    )


def test_probe_reaches_the_junction_then_turns_left():
    config = world_model.Config(environment="signal", corridor_length=3, max_steps=12)
    probe = probe_episodes(config, 2)
    turn = config.corridor_length + 2
    assert torch.all(probe.actions[:, :turn] == Action.FORWARD)
    assert torch.all(probe.actions[:, turn] == Action.LEFT)
    assert torch.all(probe.mask[:, : turn + 1]) and not probe.mask[:, turn + 1].any()


def test_entropy_bound():
    assert entropy_bound(0.5) == pytest.approx(math.log(2))
    assert entropy_bound(1.0) == 0.0


@pytest.mark.parametrize("kind", ["gru", "rssm"])
def test_distribution_metrics_run_for_both_world_models(kind):
    config = world_model.Config(
        seed=0,
        kind=kind,
        environment="signal",
        corridor_length=1,
        max_steps=10,
        steps=30,
    )
    model, optimizer = world_model.build(config, CPU)
    data = world_model.collect_episodes(config, 64, torch.Generator().manual_seed(0))
    world_model.train(
        model, optimizer, data, config, CPU, torch.Generator().manual_seed(0), 30
    )
    distribution = signal_distribution(
        model, config, 50, torch.Generator().manual_seed(1)
    )
    likelihood = signal_negative_log_likelihood(
        model, data, torch.Generator().manual_seed(1)
    )
    fractions = (
        distribution["imagined_fraction_cue_left"]
        + distribution["imagined_fraction_cue_right"]
    )
    assert 0.0 <= fractions <= 1.0
    assert 0.0 <= distribution["one_step_probability_cue_left"] <= 1.0
    assert likelihood["episodes"] > 0
    assert likelihood["negative_log_likelihood"] >= 0.0
