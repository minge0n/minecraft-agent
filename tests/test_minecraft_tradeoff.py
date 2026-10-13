"""Tests of the two diagnostic objective factors: `pitch_loss_scale` and
`kl_loss_scale` (objective trade-off follow-up, Stage 3)."""

import pytest
import torch

from minecraft_rl import resumable, runtime
from minecraft_rl.minecraft_gradients import component_gradients
from minecraft_rl.minecraft_probes import STAGES, stage_probes
from minecraft_rl.minecraft_replay import CompactVocabulary, SequenceReplay
from minecraft_rl.minecraft_world_model import (
    LOSS_TERMS,
    MinecraftRSSM,
    ModelConfig,
    Objective,
    load_checkpoint,
    objective_terms,
    save_checkpoint,
)
from minecraft_rl.rssm import kl_loss
from test_minecraft_replay import COLUMNS, RAW_SIZES, ROWS, synthetic_episode
from test_minecraft_world_model import SMALL

RAYS = float(ROWS * COLUMNS)
G = Objective(reduction="group_mean", component_scale=RAYS)


def scaled(**scales) -> Objective:
    return Objective(reduction="group_mean", component_scale=RAYS, **scales)


@pytest.fixture
def setup():
    runtime.configure(0)
    episodes = [synthetic_episode(14, seed=s, episode=s) for s in range(3)]
    compact = CompactVocabulary.from_episodes(episodes, RAW_SIZES)
    vocabulary = compact.model_vocabulary(ROWS, COLUMNS, 32.0)
    replay = SequenceReplay(episodes, compact, 6)
    batch = replay.sample(3, torch.Generator().manual_seed(1))
    return vocabulary, replay, batch


def model_for(vocabulary) -> MinecraftRSSM:
    torch.manual_seed(0)
    return MinecraftRSSM(vocabulary, ModelConfig(**SMALL))


def losses(model, batch, objective):
    return model.losses(
        batch.observations,
        batch.actions,
        batch.continues,
        torch.Generator().manual_seed(0),
        objective,
    )


def test_both_scales_default_to_one_and_g_is_unchanged(setup):
    vocabulary, _, batch = setup
    model = model_for(vocabulary)
    plain = losses(model, batch, G)
    explicit = losses(model, batch, scaled(pitch_loss_scale=1.0, kl_loss_scale=1.0))
    assert G.pitch_loss_scale == 1.0 and G.kl_loss_scale == 1.0
    for name, value in plain.items():
        assert torch.equal(value, explicit[name]), name


def test_g_trains_bit_identically_with_explicit_unit_scales(setup):
    vocabulary, replay, _ = setup
    results = []
    for objective in (G, scaled(pitch_loss_scale=1.0, kl_loss_scale=1.0)):
        model = model_for(vocabulary)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-5)
        generator = torch.Generator().manual_seed(3)
        for _ in range(3):
            batch = replay.sample(3, generator)
            terms = model.losses(
                batch.observations, batch.actions, batch.continues, generator, objective
            )
            optimizer.zero_grad()
            terms["total"].backward()
            optimizer.step()
        results.append([p.detach().clone() for p in model.parameters()])
    for a, b in zip(*results, strict=True):
        assert torch.equal(a, b)


def test_the_pitch_scale_changes_only_the_pitch_component(setup):
    vocabulary, _, batch = setup
    model = model_for(vocabulary)
    s = torch.randn(*batch.observations["health"].shape, model.features)
    prediction = model.decoder(s)
    o = batch.observations
    plain = objective_terms(prediction, o, vocabulary, "group_mean")
    pitched = objective_terms(prediction, o, vocabulary, "group_mean", 128.0)
    assert torch.allclose(pitched["pitch_component"], 128.0 * plain["pitch_component"])
    assert torch.allclose(
        pitched["scalars"] - plain["scalars"], 127.0 * plain["pitch_component"]
    )
    for name in LOSS_TERMS:
        if name != "scalars":
            assert torch.equal(pitched[name], plain[name]), name


def test_the_pitch_scale_moves_the_total_by_the_pitch_part_only(setup):
    vocabulary, _, batch = setup
    model = model_for(vocabulary)
    plain = losses(model, batch, G)
    pitched = losses(model, batch, scaled(pitch_loss_scale=256.0))
    assert torch.equal(pitched["kl_loss"], plain["kl_loss"])
    assert torch.equal(pitched["continuation"], plain["continuation"])
    expected = 255.0 * RAYS * plain["pitch_component"]
    assert torch.allclose(pitched["total"] - plain["total"], expected, rtol=1e-4)


def test_the_kl_scale_changes_only_the_final_kl_contribution(setup):
    vocabulary, _, batch = setup
    model = model_for(vocabulary)
    plain = losses(model, batch, G)
    half = losses(model, batch, scaled(kl_loss_scale=0.5))
    for name in (*LOSS_TERMS, "reconstruction", "continuation", "kl", "kl_loss"):
        assert torch.equal(half[name], plain[name]), name
    assert torch.allclose(
        plain["total"] - half["total"], 0.5 * plain["kl_loss"], rtol=1e-5, atol=1e-3
    )


def test_the_kl_scale_multiplies_the_weight_of_kl():
    objective = Objective(weights=(("kl", 0.25),), kl_loss_scale=0.5)
    assert objective.kl_coefficient == 0.125
    assert G.kl_coefficient == 1.0


def test_free_nats_work_the_same_under_the_kl_scale(setup):
    vocabulary, _, batch = setup
    model = model_for(vocabulary)
    plain = losses(model, batch, G)
    half = losses(model, batch, scaled(kl_loss_scale=0.5))
    assert torch.equal(half["kl_below_free_nats"], plain["kl_below_free_nats"])
    assert torch.equal(half["kl_loss"], plain["kl_loss"])
    torch.manual_seed(0)
    posterior = torch.randn(4, 3, 5, requires_grad=True)
    prior = (posterior + 0.01 * torch.randn(4, 3, 5)).detach()
    regularizer = kl_loss(posterior, prior, 1.0, 1.0, 1.0)
    (scaled(kl_loss_scale=0.5).kl_coefficient * regularizer.sum()).backward()
    # Below free nats the posterior part is constant, and the prior part has
    # no gradient on the posterior logits, under any KL scale.
    assert torch.count_nonzero(posterior.grad) == 0


def test_the_gradient_diagnostic_reports_the_scaled_kl(setup):
    vocabulary, _, batch = setup
    model = model_for(vocabulary)
    plain = component_gradients(model, batch, G)
    half = component_gradients(model, batch, scaled(kl_loss_scale=0.5))
    assert half["kl"]["rssm_core"] == pytest.approx(0.5 * plain["kl"]["rssm_core"])
    pitched = component_gradients(model, batch, scaled(pitch_loss_scale=128.0))
    assert pitched["pitch_in_self"]["rssm_core"] == pytest.approx(
        128.0 * plain["pitch_in_self"]["rssm_core"], rel=1e-4
    )
    assert pitched["ray_class"]["rssm_core"] == pytest.approx(
        plain["ray_class"]["rssm_core"]
    )


@pytest.mark.parametrize(
    "objective",
    [
        G,
        scaled(pitch_loss_scale=128.0),
        scaled(pitch_loss_scale=256.0),
        scaled(kl_loss_scale=0.5),
    ],
    ids=["G", "P128", "P256", "K0.5"],
)
def test_losses_and_gradients_stay_finite_and_reach_every_parameter(setup, objective):
    vocabulary, _, batch = setup
    model = model_for(vocabulary)
    terms = losses(model, batch, objective)
    assert all(torch.isfinite(v) for v in terms.values())
    terms["total"].backward()
    # Each decoder head reads one field, and a field without a valid target
    # in the synthetic batch gives its head no gradient. Every other part
    # must get one.
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        if not name.startswith("decoder.heads."):
            assert parameter.grad.abs().sum() > 0, name


def test_the_objective_round_trips_and_old_records_mean_unit_scales():
    objective = scaled(pitch_loss_scale=256.0, kl_loss_scale=0.5)
    assert Objective.from_json(objective.to_json()) == objective
    old = {"reduction": "group_mean", "weights": {}, "component_scale": RAYS}
    assert Objective.from_json(old) == G


def test_invalid_scales_are_rejected():
    for value in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="pitch_loss_scale"):
            Objective(pitch_loss_scale=value)
        with pytest.raises(ValueError, match="kl_loss_scale"):
            Objective(kl_loss_scale=value)


def test_a_checkpoint_keeps_both_scales(setup, tmp_path):
    vocabulary, _, _ = setup
    model = model_for(vocabulary)
    objective = scaled(pitch_loss_scale=128.0, kl_loss_scale=0.5)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, {"objective": objective.to_json()})
    _, extra = load_checkpoint(path)
    assert Objective.from_json(extra["objective"]) == objective


def test_a_resumed_run_refuses_a_state_of_other_scales(tmp_path):
    path = tmp_path / "state.pt"

    def identity(objective):
        return {"experiment": "minecraft_world_model", "objective": objective.to_json()}

    first = resumable.Session(path, 0.0, identity(scaled(pitch_loss_scale=256.0)))
    with pytest.raises(resumable.Incomplete):
        first.boundary(lambda: {"update": 7})
    same = resumable.Session(path, None, identity(scaled(pitch_loss_scale=256.0)))
    assert same.load()["update"] == 7
    other = resumable.Session(path, None, identity(scaled(kl_loss_scale=0.5)))
    with pytest.raises(ValueError):
        other.load()


def test_the_probes_include_the_rssm_state(setup):
    vocabulary, replay, _ = setup
    model = model_for(vocabulary)
    fit = replay.sample(4, torch.Generator().manual_seed(0))
    test = replay.sample(4, torch.Generator().manual_seed(1))
    result = stage_probes(model, fit, test)
    assert {"h", "prior_z"} <= set(STAGES)
    assert set(result) == {*STAGES, "mean_predictor"}
