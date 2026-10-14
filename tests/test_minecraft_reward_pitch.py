"""Tests of the pitch-control diagnostic `reward-pitch-v1` and its
pitch-only action restriction."""

import numpy
import pytest
import torch

from minecraft_rl import runtime
from minecraft_rl.minecraft_actor import PITCH_DEGREES, ActionSpace, Agent
from minecraft_rl.minecraft_reward import REWARD_V1
from minecraft_rl.minecraft_reward_control import CONTROL_V1
from minecraft_rl.minecraft_reward_pitch import (
    PITCH_FACTOR,
    PITCH_V1,
    PitchOnlyAgent,
    pitch_metrics,
    pitch_rewards,
)


def test_the_reward_is_zero_at_the_target_and_falls_away_from_it():
    pitch = numpy.array([0.0, 30.0, 20.0, 40.0, -90.0, 90.0])
    assert pitch_rewards(pitch) == pytest.approx(
        [0.0, -10 / 90, -10 / 90, -120 / 90, -60 / 90]
    )


def test_row_t_reads_the_next_observation_only():
    # Changing the first observation changes no reward; changing o_{t+1}
    # changes only reward t.
    pitch = numpy.array([0.0, 10.0, 20.0, 30.0])
    base = pitch_rewards(pitch)
    first = pitch.copy()
    first[0] = -80.0
    assert (pitch_rewards(first) == base).all()
    later = pitch.copy()
    later[2] = 30.0
    changed = pitch_rewards(later) != base
    assert changed.tolist() == [False, True, False]


def test_holding_the_target_earns_the_best_return():
    hold = pitch_rewards(numpy.full(11, 30.0))
    assert (hold == 0).all()
    away = pitch_rewards(numpy.full(11, 0.0))
    assert away.sum() < hold.sum()


def test_metrics_report_errors_and_the_near_share():
    m = pitch_metrics(numpy.array([0.0, 10.0, 28.0, 33.0, 40.0]))
    assert m["start_pitch"] == 0.0
    assert m["final_error"] == 10.0
    assert m["mean_error"] == pytest.approx((20 + 2 + 3 + 10) / 4)
    assert m["near_fraction"] == pytest.approx(0.5)


def test_the_reward_reads_the_observed_pitch_not_a_new_input():
    # The task has no field for positions, yaw or events: only the pitch.
    assert set(PITCH_V1.to_json()) == {"version", "target", "scale", "near"}


def test_the_pitch_only_agent_fixes_every_other_factor_at_no_op():
    runtime.configure(0)
    agent = PitchOnlyAgent(16)
    s = torch.randn(64, 16)
    choices = agent.act(s, torch.Generator().manual_seed(0))
    assert choices.shape == (64, len(ActionSpace.SIZES))
    for k, noop in enumerate(ActionSpace.NOOP):
        if k != PITCH_FACTOR:
            assert (choices[:, k] == noop).all()
    assert len(torch.unique(choices[:, PITCH_FACTOR])) > 1
    action = ActionSpace.to_player_action(choices[0].tolist())
    assert action.yaw_delta == 0.0 and not action.forward and not action.attack
    assert action.pitch_delta in PITCH_DEGREES


def test_only_the_pitch_head_learns():
    runtime.configure(0)
    agent = PitchOnlyAgent(16)
    s = torch.randn(32, 16)
    choices = agent.act(s, torch.Generator().manual_seed(0))
    log_prob, entropy, per_factor = agent.log_prob_entropy(s, choices)
    (log_prob.sum() + entropy.sum()).backward()
    for k, head in enumerate(agent.heads):
        has_grad = head.weight.grad is not None and head.weight.grad.abs().sum() > 0
        assert has_grad == (k == PITCH_FACTOR)
    assert all(
        float(h.abs().sum()) == 0 for k, h in enumerate(per_factor) if k != PITCH_FACTOR
    )


def test_a_pitch_only_checkpoint_loads_into_the_same_network():
    plain, restricted = Agent(16), PitchOnlyAgent(16)
    restricted.load_state_dict(plain.state_dict())


def test_the_earlier_reward_contracts_are_unchanged():
    assert REWARD_V1.version == "reward-v1"
    assert (REWARD_V1.block_seen, REWARD_V1.block_broken, REWARD_V1.item_acquired) == (
        0.01,
        0.05,
        0.1,
    )
    assert CONTROL_V1.to_json() == {
        "version": "reward-control-v1",
        "distance": 20.0,
        "scale": 1.0,
        "reach_radius": 1.5,
    }
