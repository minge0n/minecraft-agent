import pytest
import torch

from minecraft_rl import runtime
from minecraft_rl.minecraft_interface import RAY_KIND_BLOCK, RAY_KIND_NONE
from minecraft_rl.minecraft_replay import CompactVocabulary, SequenceReplay
from minecraft_rl.minecraft_world_model import (
    LOSS_TERMS,
    MinecraftRSSM,
    ModelConfig,
    load_checkpoint,
    ray_class,
    reconstruction_losses,
    save_checkpoint,
    targets,
)
from minecraft_rl.minecraft_world_model_eval import (
    imagination,
    one_step,
    ray_frequency_baseline,
    slot_frequency_baseline,
    unknown_classes,
)
from test_minecraft_replay import COLUMNS, RAW_SIZES, ROWS, synthetic_episode

SMALL = dict(
    hidden=32,
    latent_variables=4,
    latent_classes=4,
    embed_dim=32,
    action_dim=8,
    encoder_channels=4,
    decoder_channels=4,
    patch=2,
)


@pytest.fixture
def setup():
    runtime.configure(0)
    episodes = [synthetic_episode(14, seed=s, episode=s) for s in range(3)]
    compact = CompactVocabulary.from_episodes(episodes, RAW_SIZES)
    vocabulary = compact.model_vocabulary(ROWS, COLUMNS, 32.0)
    replay = SequenceReplay(episodes, compact, 6)
    return compact, vocabulary, replay


def model_for(vocabulary, ray_layer="patch") -> MinecraftRSSM:
    torch.manual_seed(0)
    return MinecraftRSSM(vocabulary, ModelConfig(ray_layer=ray_layer, **SMALL))


def test_ray_classes_join_kind_and_type(setup):
    _, vocabulary, _ = setup
    kind = torch.tensor([RAY_KIND_NONE, RAY_KIND_BLOCK, RAY_KIND_BLOCK, 2, 3])
    type_id = torch.tensor([0, 0, 2, 0, 1])
    classes = ray_class(kind, type_id, vocabulary)
    first = vocabulary.kind_offsets()
    assert classes.tolist() == [0, 1, 3, first[2], first[3] + 1]
    assert (vocabulary.class_kinds()[classes] == kind).all()
    assert vocabulary.ray_classes == 1 + 4 + 1 + 2


@pytest.mark.parametrize("ray_layer", ["patch", "conv"])
def test_encoder_and_decoder_shapes(setup, ray_layer):
    _, vocabulary, replay = setup
    model = model_for(vocabulary, ray_layer)
    batch = replay.sample(3, torch.Generator().manual_seed(0))
    embedded = model.encoder(batch.observations)
    assert embedded.shape == (3, 7, SMALL["embed_dim"])
    s = torch.zeros(3, 7, model.features)
    out = model.decoder(s)
    rays = ROWS * COLUMNS
    assert out["ray_class"].shape == (3, 7, rays, vocabulary.ray_classes)
    assert out["ray_distance"].shape == (3, 7, rays)
    assert out["scalars"].shape == (3, 7, 9)
    assert out["selected_slot"].shape == (3, 7, 9)
    assert out["inventory_item"].shape == (3, 7, 36, vocabulary.item_types)
    assert out["inventory_count"].shape == (3, 7, 36)
    assert out["inventory_durability"].shape == (3, 7, 36)
    assert out["armor_item"].shape == (3, 7, 4, vocabulary.item_types)
    assert out["offhand_item"].shape == (3, 7, vocabulary.item_types)
    assert out["offhand_count"].shape == (3, 7)
    assert out["effect_type"].shape == (3, 7, 8, vocabulary.effect_types)
    assert out["effect_seconds"].shape == (3, 7, 8)
    terms = reconstruction_losses(out, batch.observations, vocabulary)
    assert set(terms) == set(LOSS_TERMS)
    assert all(t.shape == (3, 7) for t in terms.values())


def test_masks_exclude_fields_without_meaning(setup):
    _, vocabulary, replay = setup
    o = replay.batch([(0, 0)]).observations
    t = targets(o, vocabulary)
    no_hit = o["ray_kind"] == RAY_KIND_NONE
    assert (t["ray_distance_mask"][no_hit] == 0).all()
    assert (t["ray_distance_mask"][~no_hit] == 1).all()
    empty = o["inventory_item"] == 0
    assert (t["inventory_count_mask"][empty] == 0).all()
    assert (t["inventory_durability_mask"][empty] == 0).all()
    assert (t["inventory_count_mask"][~empty] == 1).all()
    no_effect = o["effect_type"] == 0
    assert (t["effect_seconds_mask"][no_effect] == 0).all()
    assert (t["effect_amplifier_mask"][~no_effect] == 1).all()
    assert (t["armor_durability_mask"] == 0).all()
    assert (t["offhand_count_mask"] == 0).all()


def test_masked_predictions_do_not_change_the_loss(setup):
    _, vocabulary, replay = setup
    model = model_for(vocabulary)
    o = replay.batch([(0, 0), (1, 2)]).observations
    prediction = model.decoder(torch.randn(2, 7, model.features))
    base = reconstruction_losses(prediction, o, vocabulary)
    changed = {k: v.clone() for k, v in prediction.items()}
    no_hit = o["ray_kind"] == RAY_KIND_NONE
    changed["ray_distance"][no_hit] = 1e3
    empty = o["inventory_item"] == 0
    changed["inventory_count"][empty] = -1e3
    changed["inventory_durability"][empty] = 1e3
    changed["armor_durability"][:] = 1e3
    changed["offhand_count"][:] = 1e3
    no_effect = o["effect_type"] == 0
    changed["effect_seconds"][no_effect] = 1e3
    changed["effect_amplifier"][no_effect] = 1e3
    after = reconstruction_losses(changed, o, vocabulary)
    for name in base:
        assert torch.allclose(base[name], after[name]), name
    # A valid element still counts.
    changed["ray_distance"][~no_hit] = 1e3
    assert (reconstruction_losses(changed, o, vocabulary)["ray_distance"] > 1e5).all()


def test_ray_class_loss_targets_the_joint_class(setup):
    _, vocabulary, replay = setup
    model = model_for(vocabulary)
    o = replay.batch([(0, 0)]).observations
    prediction = model.decoder(torch.zeros(1, 7, model.features))
    target = ray_class(o["ray_kind"], o["ray_type"], vocabulary)
    logits = torch.full_like(prediction["ray_class"], -30.0)
    logits.scatter_(-1, target.unsqueeze(-1), 30.0)
    prediction["ray_class"] = logits
    assert reconstruction_losses(prediction, o, vocabulary)["ray_class"].max() < 1e-6


def test_losses_and_rssm_outputs_are_finite_and_every_part_gets_gradients(setup):
    _, vocabulary, replay = setup
    model = model_for(vocabulary)
    batch = replay.sample(4, torch.Generator().manual_seed(1))
    generator = torch.Generator().manual_seed(2)
    filtered = model.filter(batch.observations, batch.actions, generator)
    for name in ("prior", "posterior", "h", "z"):
        assert torch.isfinite(filtered[name]).all()
    terms = model.losses(batch.observations, batch.actions, batch.continues, generator)
    assert all(torch.isfinite(v) for v in terms.values())
    assert terms["kl"] >= 0
    terms["total"].backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    groups = ["encoder.", "action_net.", "cell.", "prior_net.", "posterior_net."]
    groups += ["decoder.grid.", "decoder.ray_head.", "decoder.heads.", "continue_head."]
    for group in groups:
        total = sum(
            p.grad.abs().sum()
            for name, p in model.named_parameters()
            if name.startswith(group)
        )
        assert total > 0, group


def test_the_prior_depends_on_the_action(setup):
    _, vocabulary, _ = setup
    model = model_for(vocabulary)
    h = torch.zeros(1, model.hidden)
    z = torch.zeros(1, model.latent_size)
    one, other = torch.zeros(1, 21), torch.zeros(1, 21)
    other[0, 0] = 1.0
    assert not torch.allclose(
        model.prior_logits(model.transition(h, z, one)),
        model.prior_logits(model.transition(h, z, other)),
    )


def test_imagination_uses_only_the_prior(setup):
    _, vocabulary, replay = setup
    model = model_for(vocabulary)
    batch = replay.sample(2, torch.Generator().manual_seed(0))
    calls = []
    model.encoder.register_forward_hook(lambda *a: calls.append("encoder"))
    model.decoder.register_forward_hook(lambda *a: calls.append("decoder"))
    states = model.imagine(
        torch.zeros(2, model.hidden),
        torch.zeros(2, model.latent_size),
        batch.actions,
        torch.Generator().manual_seed(0),
    )
    assert states.shape == (2, 6, model.features)
    assert calls == []


def test_checkpoint_round_trip(setup, tmp_path):
    _, vocabulary, replay = setup
    model = model_for(vocabulary)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, {"updates": 3})
    loaded, extra = load_checkpoint(path)
    assert extra == {"updates": 3}
    assert loaded.config == model.config
    assert loaded.vocabulary == vocabulary
    batch = replay.sample(2, torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.equal(
            model.encoder(batch.observations), loaded.encoder(batch.observations)
        )


def train_steps(model, optimizer, replay, generator, steps):
    for _ in range(steps):
        batch = replay.sample(3, generator)
        terms = model.losses(
            batch.observations, batch.actions, batch.continues, generator
        )
        optimizer.zero_grad()
        terms["total"].backward()
        optimizer.step()
    return terms["total"].item()


def test_resumed_training_matches_uninterrupted_training(setup, tmp_path):
    _, vocabulary, replay = setup
    model = model_for(vocabulary)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(5)
    straight = train_steps(model, optimizer, replay, generator, 6)

    resumed = model_for(vocabulary)
    resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=1e-3)
    resumed_generator = torch.Generator().manual_seed(5)
    train_steps(resumed, resumed_optimizer, replay, resumed_generator, 3)
    path = tmp_path / "state.pt"
    torch.save(
        {
            "model": resumed.state_dict(),
            "optimizer": resumed_optimizer.state_dict(),
            "generator": resumed_generator.get_state(),
        },
        path,
    )
    state = torch.load(path, weights_only=True)
    restored = model_for(vocabulary)
    restored.load_state_dict(state["model"])
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
    restored_optimizer.load_state_dict(state["optimizer"])
    restored_generator = torch.Generator()
    restored_generator.set_state(state["generator"])
    result = train_steps(restored, restored_optimizer, replay, restored_generator, 3)
    assert result == straight
    for a, b in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(a, b)


def test_a_few_updates_reduce_the_loss_on_one_batch(setup):
    _, vocabulary, replay = setup
    model = model_for(vocabulary)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    batch = replay.sample(4, torch.Generator().manual_seed(0))
    generator = torch.Generator().manual_seed(0)
    losses = []
    for _ in range(30):
        terms = model.losses(
            batch.observations, batch.actions, batch.continues, generator
        )
        optimizer.zero_grad()
        terms["total"].backward()
        optimizer.step()
        losses.append(terms["reconstruction"].item())
    assert losses[-1] < 0.7 * losses[0]


def test_evaluation_reports_baselines_and_horizons(setup):
    compact, vocabulary, replay = setup
    model = model_for(vocabulary)
    unknown = unknown_classes(compact, model)
    assert unknown.sum() == 3
    assert not unknown[0]
    frequency = ray_frequency_baseline(replay, model, unknown)
    assert frequency.shape == (ROWS * COLUMNS,)
    assert not unknown[frequency].any()
    index = replay.evaluation_windows(6)
    result = one_step(
        model,
        replay,
        index,
        2,
        unknown,
        frequency,
        slot_frequency_baseline(replay),
        torch.Generator().manual_seed(0),
    )
    persistence = result["all"]["persistence"]
    assert persistence["ray_accuracy_changed"]["mean"] == 0.0
    assert persistence["pitch_mae"]["mean"] > 0
    for name in ("model", "model_shuffled_actions", "frequency"):
        assert result["all"][name]["ray_accuracy"]["count"] > 0
    assert result["latent"]["prior_entropy"] > 0
    assert result["continuation"]["terminal_transitions"] >= 0
    rollouts = imagination(
        model, replay, index, 1, unknown, torch.Generator().manual_seed(0)
    )
    assert set(rollouts) == {"horizon_1", "horizon_5"}
    assert rollouts["horizon_5"]["persistence"]["ray_accuracy"]["count"] > 0
