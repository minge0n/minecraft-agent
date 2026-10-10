import math

import pytest
import torch

from minecraft_rl import runtime
from minecraft_rl.minecraft_gradients import PLACES, component_gradients
from minecraft_rl.minecraft_interface import RAY_KIND_NONE
from minecraft_rl.minecraft_replay import CompactVocabulary, SequenceReplay
from minecraft_rl.minecraft_world_model import (
    LOSS_GROUPS,
    LOSS_TERMS,
    REDUCTIONS,
    MinecraftRSSM,
    ModelConfig,
    Objective,
    element_losses,
    normalized_scalars,
    objective_terms,
    reconstruction_losses,
    reduce_term,
    targets,
)
from test_minecraft_replay import COLUMNS, RAW_SIZES, ROWS, synthetic_episode
from test_minecraft_world_model import SMALL


@pytest.fixture
def setup():
    runtime.configure(0)
    episodes = [synthetic_episode(14, seed=s, episode=s) for s in range(3)]
    compact = CompactVocabulary.from_episodes(episodes, RAW_SIZES)
    vocabulary = compact.model_vocabulary(ROWS, COLUMNS, 32.0)
    replay = SequenceReplay(episodes, compact, 6)
    torch.manual_seed(0)
    model = MinecraftRSSM(vocabulary, ModelConfig(**SMALL))
    batch = replay.sample(3, torch.Generator().manual_seed(1))
    return model, batch


def random_prediction(model, batch):
    s = torch.randn(*batch.observations["health"].shape, model.features)
    return model.decoder(s)


@pytest.mark.parametrize("reduction", ["ray_mean", "semantic_mean"])
def test_duplicated_rays_do_not_change_a_normalized_ray_loss(reduction):
    element = torch.rand(2, 5, 30)
    once = reduce_term("ray_class", element, None, reduction)
    twice = reduce_term("ray_class", torch.cat([element, element], -1), None, reduction)
    assert torch.allclose(once, twice)
    summed = reduce_term("ray_class", torch.cat([element, element], -1), None, "sum")
    assert torch.allclose(summed, 2 * reduce_term("ray_class", element, None, "sum"))


def test_semantic_mean_divides_by_valid_elements_only():
    element = torch.tensor([[1.0, 3.0, 100.0, 100.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    assert reduce_term("ray_distance", element, mask, "semantic_mean") == 2.0
    assert reduce_term("ray_distance", element, mask, "sum") == 4.0
    empty = reduce_term(
        "ray_distance", element, torch.zeros_like(mask), "semantic_mean"
    )
    assert empty == 0.0


def test_semantic_mean_keeps_each_scalar_a_separate_component():
    element = torch.zeros(4, 9)
    element[:, 8] = 1.0
    # One unit of pitch error counts once, not one ninth of a unit.
    assert reduce_term("scalars", element, None, "semantic_mean") == 1.0


def test_group_mean_averages_the_valid_components_of_each_group(setup):
    model, batch = setup
    prediction = random_prediction(model, batch)
    o = batch.observations
    v = model.vocabulary
    semantic = objective_terms(prediction, o, v, "semantic_mean")
    group = objective_terms(prediction, o, v, "group_mean")
    elements = element_losses(prediction, o, v)
    for names in LOSS_GROUPS.values():
        components = 0
        for name in names:
            element, mask = elements[name]
            if name == "scalars":
                components += element.shape[-1]
            elif mask is None or mask.sum() > 0:
                components += 1
        expected = sum(semantic[name] for name in names) / components
        assert torch.allclose(sum(group[name] for name in names), expected)
    # The synthetic data has no armor item: the armor durability component
    # has no valid element and adds nothing.
    assert torch.equal(group["armor_durability"], torch.tensor(0.0))
    assert torch.allclose(group["ray_class"], semantic["ray_class"])
    assert torch.allclose(group["ray_distance"], semantic["ray_distance"])


def test_sum_reduction_equals_the_original_log_likelihood(setup):
    model, batch = setup
    prediction = random_prediction(model, batch)
    original = reconstruction_losses(prediction, batch.observations, model.vocabulary)
    terms = objective_terms(prediction, batch.observations, model.vocabulary, "sum")
    for name in LOSS_TERMS:
        assert torch.equal(original[name].mean(), terms[name]), name


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_masked_elements_do_not_change_any_reduction(setup, reduction):
    model, batch = setup
    o = batch.observations
    prediction = random_prediction(model, batch)
    base = objective_terms(prediction, o, model.vocabulary, reduction)
    changed = {k: v.clone() for k, v in prediction.items()}
    changed["ray_distance"][o["ray_kind"] == RAY_KIND_NONE] = 1e4
    empty = o["inventory_item"] == 0
    changed["inventory_count"][empty] = 1e4
    changed["inventory_durability"][empty] = -1e4
    changed["armor_durability"][:] = 1e4
    changed["offhand_durability"][:] = 1e4
    no_effect = o["effect_type"] == 0
    changed["effect_seconds"][no_effect] = 1e4
    after = objective_terms(changed, o, model.vocabulary, reduction)
    for name in LOSS_TERMS:
        assert torch.allclose(base[name], after[name]), name


def test_continuous_normalization_is_deterministic_and_uses_fixed_scales(setup):
    model, batch = setup
    o = {k: v.clone() for k, v in batch.observations.items()}
    o["health"][:] = 10.0
    o["food"][:] = 20.0
    o["pitch"][:] = -45.0
    first = targets(o, model.vocabulary)
    second = targets(o, model.vocabulary)
    for name, value in first.items():
        assert torch.equal(value, second[name]), name
    scalars = normalized_scalars(o)
    assert torch.allclose(scalars[..., 0], torch.tensor(0.5))
    assert torch.allclose(scalars[..., 3], torch.tensor(1.0))
    assert torch.allclose(scalars[..., 8], torch.tensor(-0.5))
    assert torch.allclose(
        first["ray_distance"], o["ray_distance"] / model.vocabulary.max_distance
    )


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_component_weights_apply_once(setup, reduction):
    model, batch = setup
    o, a, c = batch.observations, batch.actions, batch.continues

    def total(objective):
        return model.losses(o, a, c, torch.Generator().manual_seed(0), objective)

    plain = total(Objective(reduction=reduction))
    for group, names in LOSS_GROUPS.items():
        weighted = total(Objective(reduction=reduction, weights=((group, 3.0),)))
        part = sum(plain[name] for name in names)
        assert torch.allclose(weighted["total"] - plain["total"], 2.0 * part), group
    weighted = total(Objective(reduction=reduction, weights=(("continuation", 3.0),)))
    assert torch.allclose(
        weighted["total"] - plain["total"], 2.0 * plain["continuation"]
    )
    weighted = total(Objective(reduction=reduction, weights=(("kl", 0.25),)))
    # The totals are about 150 in float32, so their difference keeps ~1e-5.
    assert torch.allclose(
        plain["total"] - weighted["total"], 0.75 * plain["kl_loss"], atol=1e-4
    )
    assert torch.equal(weighted["kl"], plain["kl"])


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_the_component_scale_multiplies_reconstruction_and_continuation(
    setup, reduction
):
    model, batch = setup
    o, a, c = batch.observations, batch.actions, batch.continues

    def total(objective):
        return model.losses(o, a, c, torch.Generator().manual_seed(0), objective)

    plain = total(Objective(reduction=reduction))
    scaled = total(Objective(reduction=reduction, component_scale=4.0))
    expected = plain["kl_loss"] + 4.0 * (
        plain["reconstruction"] + plain["continuation"]
    )
    assert torch.allclose(scaled["total"], expected, rtol=1e-5)
    assert torch.equal(scaled["kl_loss"], plain["kl_loss"])


def test_semantic_mean_with_the_ray_count_keeps_the_summed_ray_term(setup):
    model, batch = setup
    prediction = random_prediction(model, batch)
    o = batch.observations
    summed = objective_terms(prediction, o, model.vocabulary, "sum")
    mean = objective_terms(prediction, o, model.vocabulary, "semantic_mean")
    rays = ROWS * COLUMNS
    assert torch.allclose(rays * mean["ray_class"], summed["ray_class"])


def test_unknown_objectives_are_rejected():
    with pytest.raises(ValueError, match="reduction"):
        Objective(reduction="median")
    with pytest.raises(ValueError, match="loss group"):
        Objective(weights=(("pitch", 2.0),))


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_losses_and_gradients_are_finite_for_every_reduction(setup, reduction):
    model, batch = setup
    o = {k: v.clone() for k, v in batch.observations.items()}
    # A window without any hit: the ray distance has no valid element.
    o["ray_kind"][:] = RAY_KIND_NONE
    o["ray_type"][:] = 0
    terms = model.losses(
        o,
        batch.actions,
        batch.continues,
        torch.Generator().manual_seed(0),
        Objective(reduction=reduction),
    )
    assert all(torch.isfinite(v) for v in terms.values())
    assert terms["ray_distance"] == 0.0
    terms["total"].backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_component_gradients_reach_the_shared_places(setup, reduction):
    model, batch = setup
    before = [p.detach().clone() for p in model.parameters()]
    objective = Objective(reduction=reduction, weights=(("kl", 0.5),))
    result = component_gradients(model, batch, objective)
    for p, q in zip(before, model.parameters(), strict=True):
        assert torch.equal(p, q)
    assert set(result) == {*LOSS_GROUPS, "pitch_in_self", "continuation", "kl"}
    for name, values in result.items():
        assert all(math.isfinite(values[place]) for place in PLACES), name
    for group in LOSS_GROUPS:
        assert result[group]["rssm_core"] > 0, group
        assert result[group]["latent_state"] > 0, group
    assert result["ray_class"]["ray_encoder"] > 0
    # The KL loss depends on prior and posterior logits, not on s = [h, z].
    assert result["kl"]["latent_state"] == 0.0
    assert result["kl"]["rssm_core"] > 0
    plain = component_gradients(model, batch, Objective(reduction=reduction))
    assert result["kl"]["rssm_core"] == pytest.approx(0.5 * plain["kl"]["rssm_core"])
    scaled = component_gradients(
        model, batch, Objective(reduction=reduction, component_scale=3.0)
    )
    for name in (*LOSS_GROUPS, "pitch_in_self", "continuation"):
        assert scaled[name]["rssm_core"] == pytest.approx(
            3.0 * plain[name]["rssm_core"], rel=1e-4
        ), name
    assert scaled["kl"]["rssm_core"] == pytest.approx(plain["kl"]["rssm_core"])


def test_ray_mean_divides_the_ray_term_by_the_ray_count(setup):
    model, batch = setup
    prediction = random_prediction(model, batch)
    o = batch.observations
    summed = objective_terms(prediction, o, model.vocabulary, "sum")
    mean = objective_terms(prediction, o, model.vocabulary, "ray_mean")
    assert torch.allclose(mean["ray_class"], summed["ray_class"] / (ROWS * COLUMNS))
    for name in LOSS_TERMS:
        if name != "ray_class":
            assert torch.equal(mean[name], summed[name]), name
    element, _ = element_losses(prediction, o, model.vocabulary)["ray_class"]
    assert element.shape[-1] == ROWS * COLUMNS
