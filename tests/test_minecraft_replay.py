import numpy
import pytest
import torch

from minecraft_rl import runtime
from minecraft_rl.minecraft_dataset import Episode, EpisodeMeta
from minecraft_rl.minecraft_interface import (
    RAY_KIND_BLOCK,
    RAY_KIND_ENTITY,
    RAY_KIND_NONE,
)
from minecraft_rl.minecraft_replay import (
    CompactVocabulary,
    SequenceReplay,
    compact_arrays,
    unknown_fractions,
)

ROWS, COLUMNS = 5, 6
RAYS = ROWS * COLUMNS
RAW_SIZES = {"block": 50, "fluid": 4, "entity": 20, "item": 30, "effect": 6}


def synthetic_arrays(steps: int, seed: int, blocks=(3, 7, 9), items=(0, 5)) -> dict:
    """Episode arrays in the dataset format with random but valid values."""
    rng = numpy.random.default_rng(seed)
    n = steps + 1
    kind = rng.choice([RAY_KIND_NONE, RAY_KIND_BLOCK, RAY_KIND_ENTITY], (n, RAYS))
    ray_type = numpy.where(
        kind == RAY_KIND_BLOCK,
        rng.choice(blocks, (n, RAYS)),
        numpy.where(kind == RAY_KIND_ENTITY, 4, 0),
    )
    inventory = rng.choice(items, (n, 36))
    effects = numpy.zeros((n, 8), dtype=numpy.int16)
    effects[:, 0] = 2
    seconds = numpy.zeros((n, 8), dtype=numpy.int32)
    seconds[:, 0] = numpy.arange(n)[::-1] + 1
    return {
        "obs_ray_kind": kind.astype(numpy.int8),
        "obs_ray_type": ray_type.astype(numpy.int16),
        "obs_ray_distance": numpy.where(
            kind == 0, 32.0, rng.uniform(1, 20, (n, RAYS))
        ).astype(numpy.float16),
        "obs_health": rng.uniform(10, 20, n).astype(numpy.float32),
        "obs_max_health": numpy.full(n, 20.0, numpy.float32),
        "obs_absorption": numpy.zeros(n, numpy.float32),
        "obs_food": numpy.full(n, 20, numpy.int8),
        "obs_air_bubbles": numpy.full(n, 10, numpy.int8),
        "obs_armor": numpy.zeros(n, numpy.int8),
        "obs_xp_level": numpy.zeros(n, numpy.int16),
        "obs_xp_progress": numpy.zeros(n, numpy.float32),
        "obs_selected_slot": rng.integers(0, 9, n).astype(numpy.int8),
        "obs_inventory_item": inventory.astype(numpy.int16),
        "obs_inventory_count": numpy.where(inventory > 0, 3, 0).astype(numpy.uint8),
        "obs_inventory_durability": numpy.ones((n, 36), numpy.float16),
        "obs_armor_item": numpy.zeros((n, 4), numpy.int16),
        "obs_armor_durability": numpy.ones((n, 4), numpy.float16),
        "obs_offhand_item": numpy.zeros(n, numpy.int16),
        "obs_offhand_count": numpy.zeros(n, numpy.uint8),
        "obs_offhand_durability": numpy.ones(n, numpy.float16),
        "obs_effect_type": effects,
        "obs_effect_amplifier": numpy.zeros((n, 8), numpy.int16),
        "obs_effect_seconds": seconds,
        "obs_pitch": rng.uniform(-30, 30, n).astype(numpy.float32),
        "action_buttons": rng.integers(0, 2, (steps, 9)).astype(numpy.bool_),
        "action_camera": rng.uniform(-5, 5, (steps, 2)).astype(numpy.float32),
        "action_hotbar": rng.integers(-1, 9, steps).astype(numpy.int8),
        "terminated": numpy.arange(steps) == steps - 1,
        "step_id": numpy.arange(1, steps + 1),
        "tick_before": numpy.arange(steps),
        "tick_after": numpy.arange(1, steps + 1),
        "client_tick": numpy.arange(steps),
    }


def synthetic_episode(steps: int, seed: int, episode: int = 0, **kwargs) -> Episode:
    meta = EpisodeMeta(
        episode=episode,
        world_seed=seed,
        preset="normal",
        level_id="x",
        split="train",
        policy_seed=seed,
        steps=steps,
        terminated=True,
        truncated=False,
        first_tick=0,
        last_tick=steps,
    )
    return Episode(meta, synthetic_arrays(steps, seed, **kwargs))


def tag_episode(episode: Episode, tag: int) -> Episode:
    """Encode the episode index and time step into the pitch so that a window
    reveals where it came from."""
    n = episode.steps + 1
    episode.arrays["obs_pitch"] = (tag * 1000 + numpy.arange(n)).astype(numpy.float32)
    return episode


@pytest.fixture
def episodes():
    runtime.configure(0)
    return [
        tag_episode(synthetic_episode(12, seed=1, episode=0), 0),
        tag_episode(synthetic_episode(20, seed=2, episode=1), 1),
        tag_episode(synthetic_episode(5, seed=3, episode=2), 2),
    ]


def test_vocabulary_keeps_seen_ids_in_order_and_empty_slots_at_zero(episodes):
    vocabulary = CompactVocabulary.from_episodes(episodes, RAW_SIZES)
    assert vocabulary.ids["block"] == [3, 7, 9]
    assert vocabulary.ids["entity"] == [4]
    assert vocabulary.ids["item"] == [0, 5]
    assert vocabulary.ids["effect"] == [0, 2]
    assert vocabulary.ids["fluid"] == []
    assert vocabulary.size("block") == 4
    assert vocabulary.unknown("block") == 3


def test_vocabulary_is_deterministic_and_round_trips(episodes):
    first = CompactVocabulary.from_episodes(episodes, RAW_SIZES)
    second = CompactVocabulary.from_episodes(list(reversed(episodes)), RAW_SIZES)
    assert first == second
    assert first.identifier() == second.identifier()
    assert CompactVocabulary.from_json(first.to_json()) == first


def test_known_ids_map_back_to_the_raw_id(episodes):
    vocabulary = CompactVocabulary.from_episodes(episodes, RAW_SIZES)
    raw = episodes[0].arrays
    compact = compact_arrays(raw, vocabulary)
    block = raw["obs_ray_kind"] == RAY_KIND_BLOCK
    restored = numpy.array(vocabulary.ids["block"])[compact["obs_ray_type"][block]]
    assert (restored == raw["obs_ray_type"][block]).all()
    item_ids = numpy.array(vocabulary.ids["item"])
    assert (item_ids[compact["obs_inventory_item"]] == raw["obs_inventory_item"]).all()
    # Empty slots stay at index 0, so "index > 0" still means a filled slot.
    assert (
        (compact["obs_inventory_item"] == 0) == (raw["obs_inventory_item"] == 0)
    ).all()


def test_unseen_ids_map_to_the_unknown_index(episodes):
    vocabulary = CompactVocabulary.from_episodes(episodes, RAW_SIZES)
    held_out = synthetic_episode(6, seed=9, blocks=(3, 40), items=(0, 11))
    compact = compact_arrays(held_out.arrays, vocabulary)
    raw = held_out.arrays
    unseen_block = (raw["obs_ray_kind"] == RAY_KIND_BLOCK) & (raw["obs_ray_type"] == 40)
    assert unseen_block.any()
    assert (compact["obs_ray_type"][unseen_block] == vocabulary.unknown("block")).all()
    assert (compact["obs_inventory_item"][raw["obs_inventory_item"] == 11] == 2).all()
    fractions = unknown_fractions([held_out], vocabulary)
    block_rays = raw["obs_ray_kind"] == RAY_KIND_BLOCK
    expected = unseen_block.sum() / block_rays.sum()
    assert fractions["block"] == pytest.approx(expected)
    assert unknown_fractions(episodes, vocabulary)["block"] == 0.0


def test_the_train_vocabulary_is_used_for_every_split(episodes):
    vocabulary = CompactVocabulary.from_episodes(episodes, RAW_SIZES)
    held_out = synthetic_episode(6, seed=9, blocks=(40, 41))
    replay = SequenceReplay([held_out], vocabulary, 3)
    kind = replay.observations[0]["ray_kind"]
    types = replay.observations[0]["ray_type"]
    assert (types[kind == RAY_KIND_BLOCK] == vocabulary.unknown("block")).all()


def test_windows_have_the_right_shapes(episodes):
    vocabulary = CompactVocabulary.from_episodes(episodes, RAW_SIZES)
    replay = SequenceReplay(episodes, vocabulary, 4)
    assert replay.skipped_short == 0
    batch = replay.sample(6, torch.Generator().manual_seed(0))
    assert batch.observations["ray_kind"].shape == (6, 5, RAYS)
    assert batch.observations["inventory_item"].shape == (6, 5, 36)
    assert batch.observations["health"].shape == (6, 5)
    assert batch.actions.shape == (6, 4, 21)
    assert batch.continues.shape == (6, 4)


def test_short_episodes_are_skipped(episodes):
    vocabulary = CompactVocabulary.from_episodes(episodes, RAW_SIZES)
    replay = SequenceReplay(episodes, vocabulary, 8)
    assert replay.skipped_short == 1
    assert len(replay.episodes) == 2


def test_sampled_windows_never_cross_an_episode_boundary(episodes):
    vocabulary = CompactVocabulary.from_episodes(episodes, RAW_SIZES)
    replay = SequenceReplay(episodes, vocabulary, 4)
    generator = torch.Generator().manual_seed(0)
    for _ in range(50):
        batch = replay.sample(8, generator)
        pitch = batch.observations["pitch"]
        episode_tag = (pitch // 1000).long()
        assert (episode_tag == episode_tag[:, :1]).all()
        steps = pitch % 1000
        assert (steps[:, 1:] - steps[:, :-1] == 1).all()
        assert (steps[:, 0] == batch.starts).all()
        for row, e in enumerate(batch.episodes.tolist()):
            assert batch.starts[row] + 4 <= replay.episodes[e].steps


def test_continuation_is_zero_only_on_the_terminal_transition(episodes):
    vocabulary = CompactVocabulary.from_episodes(episodes, RAW_SIZES)
    replay = SequenceReplay(episodes, vocabulary, 4)
    last = replay.batch([(0, replay.episodes[0].steps - 4)])
    assert last.continues[0].tolist() == [1.0, 1.0, 1.0, 0.0]
    first = replay.batch([(0, 0)])
    assert first.continues[0].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_evaluation_windows_are_fixed_and_inside_episodes(episodes):
    vocabulary = CompactVocabulary.from_episodes(episodes, RAW_SIZES)
    replay = SequenceReplay(episodes, vocabulary, 4)
    index = replay.evaluation_windows(4)
    assert index == replay.evaluation_windows(4)
    for e, start in index:
        assert start + 4 <= replay.episodes[e].steps
