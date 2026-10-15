"""Tests of the temporal attack diagnostic: the rebuilt controller phase,
the mining-tick count and the discounted sum."""

import numpy
import pytest
import torch

from minecraft_rl.minecraft_temporal_attack import HOLD_TICKS
from minecraft_rl.minecraft_temporal_diagnostic import (
    controller_phase,
    discounted,
    mining_ticks,
)


def test_phase_matches_the_controller_for_recorded_holds():
    attack = numpy.array([0] + [1] * 40 + [0, 0] + [1] * 5, dtype=bool)
    phase = controller_phase(attack)
    assert phase[0] == 0 and phase[1] == 0
    assert phase[2] == HOLD_TICKS - 1 and phase[20] == 1
    assert phase[21] == 0 and phase[22] == HOLD_TICKS - 1
    assert phase[41] == 0 and phase[42] == 0 and phase[43] == 0
    assert phase[44] == HOLD_TICKS - 1


def test_mining_ticks_restart_when_the_target_block_changes():
    attack = numpy.array([1, 1, 1, 1, 1, 0, 1], dtype=bool)
    block = numpy.array([39, 39, 39, 115, 115, 115, 115])
    distance = numpy.array([1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6])
    assert mining_ticks(attack, block, distance).tolist() == [1, 2, 3, 1, 2, 0, 1]
    distance[2] = 2.5
    assert mining_ticks(attack, block, distance).tolist()[:3] == [1, 2, 1]


def test_discounted_sum_uses_per_tick_discount():
    rewards = torch.zeros(2, 32)
    rewards[0, 15] = 1.0
    rewards[1, 0] = 2.0
    assert discounted(rewards, 0.99).tolist() == pytest.approx([0.99**15, 2.0])
