"""Tests of the reward contract `reward-v1` (docs/decisions/reward.md)."""

import math

import pytest

from minecraft_rl.minecraft_interface import PolicyObservation
from minecraft_rl.minecraft_reward import (
    COMPONENTS,
    REWARD_V1,
    REWARD_VERSION,
    EpisodeReward,
    Registries,
    shaping_upper_bound,
)
from test_minecraft_interface import SCHEMA, observation_json, slots

# Indices: blocks 0 air, 4 grass_block, 9 oak_log; items 0 air, 3 dirt,
# 5 iron_helmet, 6 oak_log.
BLOCKS = tuple(f"minecraft:b{i}" for i in range(10))
BLOCKS = BLOCKS[:4] + ("minecraft:grass_block",) + BLOCKS[5:9] + ("minecraft:oak_log",)
ITEMS = (
    "minecraft:air",
    "minecraft:stone",
    "minecraft:stick",
    "minecraft:dirt",
    "minecraft:coal",
    "minecraft:iron_helmet",
    "minecraft:oak_log",
)
REGISTRIES = Registries(BLOCKS, ITEMS)


def observation(ray_types=None, inventory=(), armor=(0, 0, 0, 0), offhand=0):
    ray_kind = [1, 1, 0, 3, 1, 0]
    types = ray_types or [4, 4, 0, 2, 4, 0]
    payload = observation_json(
        ray_kind=ray_kind,
        ray_type=types,
        inventory_item=slots(36, inventory),
        inventory_count=slots(36, [1] * len(inventory)),
        armor_item=list(armor),
        offhand_item=offhand,
    )
    return PolicyObservation.from_json(payload, SCHEMA)


def started(first=None) -> EpisodeReward:
    reward = EpisodeReward(REWARD_V1, REGISTRIES)
    reward.reset(first or observation())
    return reward


def test_components_sum_exactly_to_the_scalar():
    reward = started()
    step = reward.step(observation(inventory=[6]), False, [9], ["minecraft:story/root"])
    assert set(step.components) == set(COMPONENTS)
    assert step.total == math.fsum(step.components[name] for name in COMPONENTS)
    assert step.total > 0


def test_first_seen_block_earns_novelty_once():
    reward = started()
    first = reward.step(observation(), False)
    assert first.components["block_seen_novelty"] == pytest.approx(0.01)
    assert first.events["blocks_seen"] == [4]
    again = reward.step(observation(), False)
    assert again.components["block_seen_novelty"] == 0.0


def test_a_different_block_type_earns_its_own_novelty():
    reward = started()
    reward.step(observation(), False)
    other = reward.step(observation(ray_types=[9, 4, 0, 2, 4, 0]), False)
    assert other.events["blocks_seen"] == [9]
    assert other.components["block_seen_novelty"] == pytest.approx(0.01)


def test_place_and_break_loops_earn_break_novelty_once():
    reward = started()
    totals = [reward.step(observation(), False, [4]).components for _ in range(5)]
    assert totals[0]["block_broken_novelty"] == pytest.approx(0.05)
    assert all(c["block_broken_novelty"] == 0.0 for c in totals[1:])


def test_acquisition_is_distinct_from_breaking():
    reward = started()
    broken = reward.step(observation(), False, [9])
    assert broken.components["block_broken_novelty"] == pytest.approx(0.05)
    assert broken.components["item_acquired_novelty"] == 0.0
    acquired = reward.step(observation(inventory=[6]), False)
    assert acquired.components["item_acquired_novelty"] == pytest.approx(0.1)
    assert acquired.components["importance"] == pytest.approx(0.2)
    assert acquired.events["important_items"] == [6]


def test_seeing_is_worth_less_than_acquiring():
    assert REWARD_V1.block_seen < REWARD_V1.block_broken < REWARD_V1.item_acquired


def test_drop_and_pick_up_again_earns_nothing_more():
    reward = started()
    first = reward.step(observation(inventory=[3]), False).total
    reward.step(observation(), False)
    again = reward.step(observation(inventory=[3]), False)
    assert first > 0
    assert again.components["item_acquired_novelty"] == 0.0


def test_items_held_at_the_start_earn_nothing():
    reward = started(observation(inventory=[3], armor=(0, 0, 0, 5)))
    step = reward.step(observation(inventory=[3], armor=(0, 0, 0, 5)), False)
    assert step.components["item_acquired_novelty"] == 0.0


def test_armor_and_offhand_count_as_held():
    reward = started()
    step = reward.step(observation(armor=(0, 0, 0, 5), offhand=4), False)
    assert step.events["items_acquired"] == [4, 5]


def test_unimportant_items_earn_novelty_only():
    reward = started()
    step = reward.step(observation(inventory=[3]), False)
    assert step.components["importance"] == 0.0
    assert step.components["item_acquired_novelty"] == pytest.approx(0.1)


def test_the_highest_matching_importance_wins():
    assert REWARD_V1.importance_of("minecraft:oak_log") == 0.2
    assert REWARD_V1.importance_of("minecraft:crimson_stem") == 0.2
    assert REWARD_V1.importance_of("minecraft:dirt") == 0.0
    assert REWARD_V1.importance_of("minecraft:ender_eye") == 8.0


def test_advancements_and_milestones_count_once():
    reward = started()
    nether = "minecraft:story/enter_the_nether"
    first = reward.step(observation(), False, (), [nether])
    assert first.components["advancement"] == 1.0
    assert first.components["progression"] == 10.0
    again = reward.step(observation(), False, (), [nether])
    assert again.components["advancement"] == 0.0
    assert again.components["progression"] == 0.0


def test_the_dragon_kill_is_the_terminal_reward():
    reward = started()
    step = reward.step(observation(), False, (), ["minecraft:end/kill_dragon"])
    assert step.components["terminal"] == 1000.0


def test_death_costs_once():
    reward = started()
    assert reward.step(observation(), True).components["survival"] == -1.0
    assert reward.step(observation(), True).components["survival"] == 0.0


def test_staying_alive_and_damage_earn_nothing():
    reward = started()
    reward.step(observation(), False)
    quiet = [reward.step(observation(), False).total for _ in range(20)]
    assert quiet == [0.0] * 20


def test_reset_starts_a_fresh_novelty_memory():
    reward = started()
    reward.step(observation(inventory=[3]), False, [4])
    reward.reset(observation())
    step = reward.step(observation(inventory=[3]), False, [4])
    assert step.components["block_broken_novelty"] == pytest.approx(0.05)
    assert step.components["item_acquired_novelty"] == pytest.approx(0.1)


def test_the_same_trajectory_gives_the_same_reward():
    trajectory = [
        (observation(), False, [4], []),
        (observation(inventory=[6]), False, [9], ["minecraft:story/root"]),
        (observation(ray_types=[9, 9, 0, 2, 4, 0], inventory=[6, 3]), True, [], []),
    ]

    def play():
        reward = started()
        steps = [reward.step(*t) for t in trajectory]
        return [s.total for s in steps], [s.components for s in steps], reward.summary()

    assert play() == play()


def test_the_summary_reports_totals_counts_and_version():
    reward = started()
    reward.step(observation(inventory=[6]), False, [9], ["minecraft:story/root"])
    summary = reward.summary()
    assert summary["version"] == REWARD_VERSION
    assert summary["total"] == pytest.approx(sum(summary["components"].values()))
    assert summary["counts"]["items_acquired"] == 1
    assert summary["counts"]["important_items"] == 1


def test_the_dragon_kill_dominates_all_shaping():
    names = tuple(f"minecraft:block_{i}" for i in range(1286))
    items = ("minecraft:air", *(name for name, _ in REWARD_V1.importance))
    items += tuple(f"minecraft:item_{i}" for i in range(1658 - len(items)))
    bound = shaping_upper_bound(REWARD_V1, Registries(names, items), advancements=125)
    assert bound < REWARD_V1.terminal


def test_the_specification_is_recorded_in_full():
    spec = REWARD_V1.to_json()
    assert spec["version"] == REWARD_VERSION
    assert spec["importance"]["minecraft:ender_eye"] == 8.0
    assert spec["progression"]["minecraft:story/enter_the_end"] == 30.0
