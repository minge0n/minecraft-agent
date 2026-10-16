"""Tests of the full-registry vocabulary, the factored decoder heads, the
v3 checkpoint and the migration from a compact model."""

import numpy
import pytest
import torch

from minecraft_rl import runtime
from minecraft_rl.minecraft_registry import Registry
from minecraft_rl.minecraft_registry_migration import migrate
from minecraft_rl.minecraft_replay import (
    CompactVocabulary,
    RegistryVocabulary,
    SequenceReplay,
    check_registry,
    compact_arrays,
    registry_metadata,
)
from minecraft_rl.minecraft_world_model import (
    ITEM_SLOTS,
    TYPE_CHUNK,
    FactoredDecoder,
    MinecraftRSSM,
    ModelConfig,
    element_losses,
    load_checkpoint,
    ray_class,
    save_checkpoint,
)
from minecraft_rl.minecraft_world_model_eval import categorical_outputs
from test_minecraft_replay import COLUMNS, RAW_SIZES, ROWS, synthetic_episode
from test_minecraft_world_model import SMALL

NAMES = {
    "block": tuple(f"test:block_{i}" for i in range(RAW_SIZES["block"])),
    "fluid": tuple(f"test:fluid_{i}" for i in range(RAW_SIZES["fluid"])),
    "entity": tuple(f"test:entity_{i}" for i in range(RAW_SIZES["entity"])),
    "item": ("test:air",)
    + tuple(f"test:item_{i}" for i in range(1, RAW_SIZES["item"])),
    "effect": tuple(f"test:effect_{i}" for i in range(RAW_SIZES["effect"] - 1)),
}
REGISTRY = Registry("test", NAMES)


@pytest.fixture
def setup():
    runtime.configure(0)
    episodes = [synthetic_episode(14, seed=s, episode=s) for s in range(3)]
    vocabulary = RegistryVocabulary(REGISTRY)
    torch.manual_seed(0)
    model = MinecraftRSSM(
        vocabulary.model_vocabulary(ROWS, COLUMNS, 32.0),
        ModelConfig(heads="factored", item_slot_features=8, **SMALL),
    )
    return episodes, vocabulary, model


def test_every_registered_id_is_its_own_category():
    vocabulary = RegistryVocabulary(REGISTRY)
    sizes = REGISTRY.sizes()
    assert sizes == RAW_SIZES
    for family, size in sizes.items():
        table = vocabulary.table(family)
        assert table.tolist() == list(range(size))
    v = vocabulary.model_vocabulary(ROWS, COLUMNS, 32.0)
    kinds = torch.tensor(
        [1] * sizes["block"] + [2] * sizes["fluid"] + [3] * sizes["entity"]
    )
    types = torch.cat([torch.arange(sizes[f]) for f in ("block", "fluid", "entity")])
    classes = ray_class(kinds, types, v)
    assert len(set(classes.tolist())) == len(classes)
    assert 0 not in classes.tolist()
    assert classes.max() == v.ray_classes - 1


def test_an_id_unseen_in_training_keeps_its_identity():
    train = synthetic_episode(10, seed=1, blocks=(3, 7), items=(0, 5))
    held = synthetic_episode(10, seed=2, blocks=(9,), items=(0, 11))
    compact = CompactVocabulary.from_episodes([train], RAW_SIZES)
    old = compact_arrays(held.arrays, compact)
    kind = held.arrays["obs_ray_kind"]
    assert (old["obs_ray_type"][kind == 1] == compact.unknown("block")).all()
    registry = compact_arrays(held.arrays, RegistryVocabulary(REGISTRY))
    assert (registry["obs_ray_type"][kind == 1] == 9).all()
    assert set(numpy.unique(registry["obs_inventory_item"])) == {0, 11}


def test_empty_slots_and_no_hit_stay_apart_from_real_types():
    vocabulary = RegistryVocabulary(REGISTRY)
    v = vocabulary.model_vocabulary(ROWS, COLUMNS, 32.0)
    classes = ray_class(torch.tensor([0, 1]), torch.tensor([0, 0]), v)
    assert classes.tolist() == [0, 1]
    assert REGISTRY.names["item"][0] == "test:air"
    assert REGISTRY.sizes()["effect"] == len(NAMES["effect"]) + 1


def test_the_registry_fingerprint_is_deterministic_and_detects_changes(tmp_path):
    same = Registry("test", {k: tuple(v) for k, v in NAMES.items()})
    assert same.fingerprint == REGISTRY.fingerprint
    path = tmp_path / "registry.json"
    REGISTRY.save(path)
    assert Registry.load(path).fingerprint == REGISTRY.fingerprint
    swapped = dict(NAMES)
    swapped["item"] = (NAMES["item"][0], NAMES["item"][2], NAMES["item"][1]) + NAMES[
        "item"
    ][3:]
    other = Registry("test", swapped)
    assert other.fingerprint != REGISTRY.fingerprint
    with pytest.raises(ValueError, match="registry mismatch"):
        check_registry(registry_metadata(REGISTRY), other)
    with pytest.raises(ValueError, match="registry mismatch"):
        REGISTRY.check_schema(
            {"block_types": 1, "fluid_types": 4, "entity_types": 20}
            | {"item_types": 30, "effect_types": 5}
        )


def test_an_id_outside_the_registry_is_refused():
    episode = synthetic_episode(5, seed=1)
    episode.arrays["obs_inventory_item"][0, 0] = RAW_SIZES["item"]
    with pytest.raises(ValueError, match="outside the registry"):
        compact_arrays(episode.arrays, RegistryVocabulary(REGISTRY))


def test_one_shared_item_classifier_covers_every_slot_and_item(setup):
    _, _, model = setup
    decoder = model.decoder
    assert isinstance(decoder, FactoredDecoder)
    assert decoder.item_head.out_features == RAW_SIZES["item"]
    assert decoder.item_slot_embedding.num_embeddings == ITEM_SLOTS
    prediction = decoder(torch.randn(2, 3, model.features))
    logits = decoder.item_logits(prediction["item_features"])
    assert logits["inventory_item"].shape == (2, 3, 36, RAW_SIZES["item"])
    assert logits["armor_item"].shape == (2, 3, 4, RAW_SIZES["item"])
    assert logits["offhand_item"].shape == (2, 3, RAW_SIZES["item"])
    features = prediction["item_features"]
    expected = decoder.item_head(features[0, 0, 37])
    assert torch.allclose(logits["armor_item"][0, 0, 1], expected)
    item_heads = [n for n, _ in decoder.named_modules() if n.endswith("item_head")]
    assert item_heads == ["item_head"]


def test_the_ray_decoder_covers_every_block_and_its_log_probs_normalize(setup):
    _, _, model = setup
    decoder = model.decoder
    assert decoder.type_heads["block"].out_features == RAW_SIZES["block"]
    features = torch.randn(5, 16, decoder.kind_head.in_features)
    full = decoder.ray_class_log_probs(features)
    assert full.shape[-1] == model.vocabulary.ray_classes
    assert torch.allclose(full.logsumexp(-1), torch.zeros(5, 16), atol=1e-5)
    target = torch.randint(0, model.vocabulary.ray_classes, (5, 16))
    exact = decoder.ray_log_prob(features, target)
    assert torch.allclose(
        exact, full.gather(-1, target.unsqueeze(-1)).squeeze(-1), atol=1e-5
    )
    assert torch.equal(decoder.ray_class_mode(features), full.argmax(-1))


def test_chunked_type_loss_equals_the_unchunked_loss(setup, monkeypatch):
    _, _, model = setup
    decoder = model.decoder
    features = torch.randn(3, 40, decoder.kind_head.in_features)
    target = torch.randint(1, 1 + RAW_SIZES["block"], (3, 40))
    whole = decoder.ray_log_prob(features, target)
    import minecraft_rl.minecraft_world_model as world_model

    monkeypatch.setattr(world_model, "TYPE_CHUNK", 7)
    assert TYPE_CHUNK != 7
    assert torch.allclose(decoder.ray_log_prob(features, target), whole, atol=1e-6)


def test_loss_masks_and_gradients_reach_the_rows_of_present_categories(setup):
    episodes, vocabulary, model = setup
    replay = SequenceReplay(episodes, vocabulary, 6)
    batch = replay.batch([(0, 0), (1, 2)])
    o = batch.observations
    terms = model.losses(
        o, batch.actions, batch.continues, torch.Generator().manual_seed(0)
    )
    model.zero_grad()
    terms["total"].backward()
    decoder = model.decoder
    present_items = set(o["inventory_item"].unique().tolist())
    assert (decoder.item_head.weight.grad.abs().sum(-1) > 0).all()
    embed_grad = model.encoder.item_embedding.weight.grad.abs().sum(-1)
    for item in range(RAW_SIZES["item"]):
        assert bool(embed_grad[item] > 0) == (item in present_items)
    kind, types = o["ray_kind"], o["ray_type"]
    present_blocks = set(types[kind == 1].unique().tolist())
    block_grad = decoder.type_heads["block"].weight.grad.abs().sum(-1)
    assert all(block_grad[b] > 0 for b in present_blocks)
    fluid_grad = decoder.type_heads["fluid"].weight.grad
    if not (kind == 2).any():
        assert fluid_grad is None or fluid_grad.abs().sum() == 0
    prediction = decoder(torch.randn(1, 1, model.features))
    elements = element_losses(prediction, _first(o), model.vocabulary, decoder)
    _, mask = elements["ray_distance"]
    assert torch.equal(mask, (_first(o)["ray_kind"] != 0).float().reshape(mask.shape))


def _first(o):
    return {k: v[:1, :1] for k, v in o.items()}


def test_factored_loss_equals_cross_entropy_of_its_full_logits(setup):
    episodes, vocabulary, model = setup
    batch = SequenceReplay(episodes, vocabulary, 6).batch([(0, 0)])
    o = {k: v[:, :3] for k, v in batch.observations.items()}
    prediction = model.decoder(torch.randn(1, 3, model.features))
    factored = element_losses(prediction, o, model.vocabulary, model.decoder)
    full = categorical_outputs(model, prediction)
    joint = element_losses(full, o, model.vocabulary)
    for name in (
        "ray_class",
        "inventory_item",
        "armor_item",
        "offhand_item",
        "effect_type",
    ):
        assert torch.allclose(factored[name][0], joint[name][0], atol=1e-5), name


def test_a_registry_checkpoint_round_trips_and_needs_its_registry(setup, tmp_path):
    episodes, vocabulary, model = setup
    path = tmp_path / "model.pt"
    with pytest.raises(ValueError, match="needs its registry"):
        save_checkpoint(path, model, {})
    save_checkpoint(path, model, {"registry": vocabulary.to_json()})
    loaded, extra = load_checkpoint(path)
    assert loaded.config.heads == "factored"
    check_registry(extra["registry"], REGISTRY)
    batch = SequenceReplay(episodes, vocabulary, 6).batch([(1, 1)])
    with torch.no_grad():
        assert torch.equal(
            model.encoder(batch.observations), loaded.encoder(batch.observations)
        )


def test_a_compact_checkpoint_still_loads(tmp_path):
    runtime.configure(0)
    episodes = [synthetic_episode(14, seed=s, episode=s) for s in range(2)]
    compact = CompactVocabulary.from_episodes(episodes, RAW_SIZES)
    torch.manual_seed(0)
    model = MinecraftRSSM(
        compact.model_vocabulary(ROWS, COLUMNS, 32.0), ModelConfig(**SMALL)
    )
    config = model.config.to_json()
    del config["heads"], config["item_slot_features"]
    path = tmp_path / "v2.pt"
    torch.save(
        {
            "format": "minecraft-rssm-checkpoint-v2",
            "config": config,
            "vocabulary": vars(model.vocabulary) | {},
            "state_dict": model.state_dict(),
            "extra": {"vocabulary": compact.to_json()},
        },
        path,
    )
    loaded, _ = load_checkpoint(path)
    assert loaded.config.heads == "joint"
    for (a, p), (b, q) in zip(
        model.state_dict().items(), loaded.state_dict().items(), strict=True
    ):
        assert a == b and torch.equal(p, q)


def test_migration_copies_known_rows_to_their_registry_ids(setup):
    episodes, vocabulary, new = setup
    compact = CompactVocabulary.from_episodes(episodes[:2], RAW_SIZES)
    torch.manual_seed(1)
    old = MinecraftRSSM(
        compact.model_vocabulary(ROWS, COLUMNS, 32.0), ModelConfig(**SMALL)
    )
    before = new.encoder.item_embedding.weight.clone()
    counts = migrate(old, compact, new)
    assert counts["embedding_rows"] > 0 and counts["ray_type_rows"] > 0
    for index, registry_id in enumerate(compact.ids["item"]):
        assert torch.equal(
            new.encoder.item_embedding.weight[registry_id],
            old.encoder.item_embedding.weight[index],
        )
    unseen = sorted(set(range(RAW_SIZES["item"])) - set(compact.ids["item"]))
    assert torch.equal(new.encoder.item_embedding.weight[unseen], before[unseen])
    offsets_old = old.vocabulary.kind_offsets()
    for index, registry_id in enumerate(compact.ids["block"]):
        assert torch.equal(
            new.decoder.type_heads["block"].weight[registry_id],
            old.decoder.ray_head.weight[offsets_old[1] + index],
        )
        assert torch.equal(
            new.encoder.ray_embedding.weight[1 + registry_id],
            old.encoder.ray_embedding.weight[offsets_old[1] + index],
        )
    assert torch.equal(new.cell.weight_hh, old.cell.weight_hh)
    raw = compact_arrays(episodes[0].arrays, vocabulary)
    old_ids = compact_arrays(episodes[0].arrays, compact)
    o_new = {k: torch.as_tensor(v) for k, v in raw.items() if k == "obs_inventory_item"}
    o_old = {
        k: torch.as_tensor(v) for k, v in old_ids.items() if k == "obs_inventory_item"
    }
    with torch.no_grad():
        assert torch.equal(
            new.encoder.item_embedding(o_new["obs_inventory_item"].long()),
            old.encoder.item_embedding(o_old["obs_inventory_item"].long()),
        )


def test_chunked_target_log_softmax_has_exact_gradients(monkeypatch):
    import minecraft_rl.minecraft_world_model as world_model

    monkeypatch.setattr(world_model, "TYPE_CHUNK", 3)
    torch.manual_seed(0)
    x = torch.randn(10, 4, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(6, dtype=torch.float64, requires_grad=True)
    target = torch.randint(0, 6, (10,))
    apply = world_model.TargetLogSoftmax.apply
    assert torch.autograd.gradcheck(apply, (x, weight, bias, target))
    reference = torch.log_softmax(x @ weight.T + bias, -1)
    expected = reference.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(apply(x, weight, bias, target), expected)


def test_the_pinned_registry_is_refused_when_it_differs(tmp_path):
    from minecraft_rl.minecraft_registry import PINNED_SIZES

    path = tmp_path / "registry.json"
    REGISTRY.save(path)
    with pytest.raises(ValueError, match="not the pinned"):
        Registry.pinned(path)
    assert PINNED_SIZES["effect"] == 41 and PINNED_SIZES["item"] == 1658
