import math

import pytest
import torch

from minecraft_rl import actor_critic, world_model
from minecraft_rl.imagination import rollout_errors
from minecraft_rl.rssm import (
    RSSM,
    categorical_entropy,
    categorical_kl,
    categorical_kl_per_variable,
    first_cue_states,
    kl_loss,
    sample_one_hot,
)
from minecraft_rl.tmaze import Observation

CPU = torch.device("cpu")
FAST = world_model.Config(
    seed=0,
    kind="rssm",
    corridor_length=1,
    max_steps=12,
    training_episodes=256,
    evaluation_episodes=128,
    learning_rate=0.01,
    steps=800,
)
LATENT = FAST.latent_variables * FAST.latent_classes


def episodes(count: int = 16, seed: int = 0) -> world_model.Episodes:
    return world_model.collect_episodes(
        FAST, count, torch.Generator().manual_seed(seed)
    )


@pytest.fixture(scope="module")
def trained() -> RSSM:
    model, optimizer = world_model.build(FAST, CPU)
    world_model.train(
        model,
        optimizer,
        episodes(FAST.training_episodes),
        FAST,
        CPU,
        torch.Generator().manual_seed(0),
        FAST.steps,
    )
    return model


def test_categorical_kl_matches_hand_computation():
    q = torch.tensor([[[0.9, 0.1]]])
    p = torch.tensor([[[0.5, 0.5]]])
    expected = 0.9 * math.log(0.9 / 0.5) + 0.1 * math.log(0.1 / 0.5)
    assert categorical_kl(q.log(), p.log()).item() == pytest.approx(expected)
    assert categorical_kl(p.log(), p.log()).item() == pytest.approx(0.0, abs=1e-7)
    uniform = torch.zeros(3, 8, 4)
    assert torch.allclose(
        categorical_entropy(uniform), torch.full((3,), 8 * math.log(4))
    )


def test_kl_loss_parts_and_free_nats_route_gradients():
    posterior = torch.randn(2, 8, 4, requires_grad=True)
    prior = torch.randn(2, 8, 4, requires_grad=True)
    plain = categorical_kl(posterior, prior)

    combined = kl_loss(posterior, prior, 1.0, 1.0, 0.0)
    assert torch.allclose(combined, 2 * plain)
    combined.sum().backward()
    plain_posterior, plain_prior = torch.autograd.grad(
        categorical_kl(posterior, prior).sum(), (posterior, prior)
    )
    assert torch.allclose(posterior.grad, plain_posterior, atol=1e-6)
    assert torch.allclose(prior.grad, plain_prior, atol=1e-6)

    posterior.grad, prior.grad = None, None
    kl_loss(posterior, prior, 1.0, 0.0, 0.0).sum().backward()
    assert torch.all(posterior.grad == 0)
    assert prior.grad.abs().sum() > 0

    posterior.grad, prior.grad = None, None
    kl_loss(posterior, prior, 0.0, 1.0, plain.max().item() + 1.0).sum().backward()
    assert torch.all(posterior.grad == 0)


def test_one_hot_samples_follow_the_probabilities_and_pass_gradients():
    logits = torch.log(torch.tensor([[0.7, 0.2, 0.1]])).expand(20000, 1, 3)
    samples = sample_one_hot(logits, torch.Generator().manual_seed(0), False)
    assert torch.all(samples.sum(-1) == 1)
    frequencies = samples.mean((0, 1))
    assert torch.allclose(frequencies, torch.tensor([0.7, 0.2, 0.1]), atol=0.02)

    trainable = torch.zeros(1, 2, 3, requires_grad=True)
    sample = sample_one_hot(trainable, torch.Generator().manual_seed(0), True)
    assert torch.all((sample == 0) | (sample == 1))
    (sample * torch.tensor([1.0, 2.0, 3.0])).sum().backward()
    assert trainable.grad.abs().sum() > 0


def test_filter_and_policy_state_shapes():
    model, _ = world_model.build(FAST, CPU)
    data = episodes()
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        filtered = model.filter(data.observations, data.actions[:, :-1], generator)
        states = model.policy_states(data, generator)
    batch, time = data.observations.shape
    assert filtered["h"].shape == (batch, time, FAST.hidden)
    assert filtered["z"].shape == (batch, time, LATENT)
    assert filtered["prior"].shape == (batch, time, 8, 4)
    assert torch.all(filtered["h"][:, 0] == 0)
    assert torch.all(filtered["z"].unflatten(-1, (8, 4)).sum(-1) == 1)
    assert states.shape == (batch, time, FAST.hidden + LATENT)
    assert model.policy_state_size == world_model.policy_state_size(FAST)


def test_observations_reach_the_recurrent_state_only_through_the_latent():
    model, _ = world_model.build(FAST, CPU)
    data = episodes()
    altered = world_model.Episodes(
        *(getattr(data, f).clone() for f in world_model.Episodes.__dataclass_fields__)
    )
    altered.observations[:, 0] = Observation.ARM
    with torch.no_grad():
        original = model.filter(
            data.observations, data.actions[:, :-1], torch.Generator().manual_seed(0)
        )
        changed = model.filter(
            altered.observations,
            altered.actions[:, :-1],
            torch.Generator().manual_seed(0),
        )
    assert torch.equal(original["h"][:, 0], changed["h"][:, 0])
    assert not torch.equal(original["posterior"][:, 0], changed["posterior"][:, 0])
    assert torch.equal(original["prior"][:, 0], changed["prior"][:, 0])


def test_padding_does_not_contribute_to_the_loss():
    model, _ = world_model.build(FAST, CPU)
    data = episodes()
    before = model.training_losses(data, torch.Generator().manual_seed(0))
    padding = ~data.mask
    data.next_observations[padding] = Observation.ARM
    data.rewards[padding] = 5.0
    data.continues[padding] = 1.0
    after = model.training_losses(data, torch.Generator().manual_seed(0))
    assert torch.equal(before["total"], after["total"])


def test_every_loss_term_trains_the_parameters_it_should():
    model, _ = world_model.build(FAST, CPU)
    terms = model.training_losses(episodes(), torch.Generator().manual_seed(0))
    expected = {
        "reconstruction": ("posterior_net", "observation_head"),
        "reward": ("reward_head", "cell", "posterior_net"),
        "continuation": ("continue_head",),
        "kl": ("prior_net", "posterior_net"),
    }
    for term, modules in expected.items():
        model.zero_grad()
        terms = model.training_losses(episodes(), torch.Generator().manual_seed(0))
        terms[term].backward()
        for module in modules:
            gradient = sum(
                p.grad.abs().sum() for p in getattr(model, module).parameters()
            )
            assert gradient > 0, (term, module)
    model.zero_grad()
    terms = model.training_losses(episodes(), torch.Generator().manual_seed(0))
    terms["continuation"].backward()
    assert all(
        p.grad is None or torch.all(p.grad == 0) for p in model.prior_net.parameters()
    )


def test_trained_rssm_predicts_the_cue_dependent_reward(trained):
    evaluation = episodes(FAST.evaluation_episodes, seed=1)
    result = world_model.evaluate(
        trained, evaluation, CPU, torch.Generator().manual_seed(0)
    )
    assert result["observation_accuracy"] > 0.99
    assert result["continuation_accuracy"] > 0.99
    assert result["junction_turn"]["reward_sign_accuracy"] > 0.95


def test_trained_latent_carries_the_cue_and_is_not_collapsed(trained):
    latent = trained.diagnostics(
        episodes(FAST.evaluation_episodes, seed=1), torch.Generator().manual_seed(0)
    )
    assert latent["kl_cue_step"] > math.log(2) * 0.8
    assert latent["cue_coding_variables"] >= 1
    assert latent["prior_entropy_cue_step"] > math.log(2) * 0.8
    assert latent["kl_later_steps"] < latent["kl_cue_step"]
    assert latent["posterior_entropy_mean"] < latent["maximum_entropy"] / 2


def test_latent_imagination_never_decodes_observations(trained):
    agent, _ = actor_critic.build(actor_critic.Config(model=FAST), CPU)
    starts = actor_critic.start_states(
        trained, episodes(), torch.Generator().manual_seed(0)
    )
    calls = []
    hook = trained.observation_head.register_forward_hook(lambda *_: calls.append(1))
    trained.imagine_step(
        starts[:4],
        torch.zeros(4, dtype=torch.long),
        torch.Generator().manual_seed(0),
    )
    hook.remove()
    imagined = actor_critic.imagine(
        trained, agent, starts[:4], 5, torch.Generator().manual_seed(0)
    )
    assert imagined.features.shape == (4, 6, FAST.hidden + LATENT)
    assert calls == [1]


def test_open_loop_errors_are_finite_for_the_rssm(trained):
    errors = rollout_errors(
        trained, episodes(32, seed=1), (1, 5), torch.Generator().manual_seed(0)
    )
    assert errors["1"]["observation_accuracy"] > 0.95
    assert 0.0 <= errors["5"]["return_absolute_error"] < 0.5


def test_checkpoint_round_trip_and_resume(tmp_path):
    config = world_model.Config(
        seed=0, kind="rssm", corridor_length=1, max_steps=12, steps=20
    )
    data = world_model.collect_episodes(config, 32, torch.Generator().manual_seed(0))
    uninterrupted, optimizer = world_model.build(config, CPU)
    generator = torch.Generator().manual_seed(0)
    world_model.train(uninterrupted, optimizer, data, config, CPU, generator, 20)

    generator = torch.Generator().manual_seed(0)
    resumed, optimizer = world_model.build(config, CPU)
    world_model.train(resumed, optimizer, data, config, CPU, generator, 10)
    path = tmp_path / "rssm.pt"
    world_model.save_checkpoint(path, config, 10, resumed, optimizer)
    restored_config, step, restored, restored_optimizer = world_model.load_checkpoint(
        path, CPU
    )
    assert isinstance(restored, RSSM)
    world_model.train(restored, restored_optimizer, data, config, CPU, generator, 10)
    assert restored_config == config
    assert step == 10
    for a, b in zip(uninterrupted.parameters(), restored.parameters(), strict=True):
        assert torch.equal(a, b)


def test_first_cue_state_is_the_first_cue_observation():
    observations = torch.tensor(
        [
            [Observation.CUE_LEFT, Observation.CORRIDOR, Observation.CUE_LEFT],
            [Observation.CORRIDOR, Observation.CUE_RIGHT, Observation.CORRIDOR],
            [Observation.CORRIDOR, Observation.CORRIDOR, Observation.CORRIDOR],
        ]
    )
    valid = torch.ones_like(observations, dtype=torch.bool)
    expected = torch.tensor(
        [[True, False, False], [False, True, False], [False, False, False]]
    )
    assert torch.equal(first_cue_states(observations, valid), expected)


def test_per_variable_kl_sums_to_the_total_kl():
    q, p = torch.randn(5, 8, 4), torch.randn(5, 8, 4)
    assert torch.allclose(
        categorical_kl_per_variable(q, p).sum(-1), categorical_kl(q, p)
    )


def test_diagnostics_report_the_free_nats_effect_and_latent_use(trained):
    latent = trained.diagnostics(
        episodes(FAST.evaluation_episodes, seed=1), torch.Generator().manual_seed(0)
    )
    assert (
        latent["kl_posterior_part_effective"]
        >= max(FAST.free_nats, latent["kl_mean"]) - 1e-6
    )
    assert latent["kl_loss_effective"] >= latent["kl_mean"]
    assert len(latent["kl_per_variable_cue_step"]) == FAST.latent_variables
    assert sum(latent["kl_per_variable_cue_step"]) == pytest.approx(
        latent["kl_cue_step"], rel=1e-4
    )
    assert 1 <= latent["active_variables"] <= FAST.latent_variables
    assert 1.0 <= latent["class_perplexity_mean"] <= FAST.latent_classes
    assert 0.0 <= latent["prior_posterior_agreement"] <= 1.0
    intervention = latent["cue_from_prior"]
    assert intervention["reward_sign_accuracy_posterior"] > 0.95
    assert intervention["reward_sign_accuracy_cue_from_prior"] < 0.9


def test_diagnostics_do_not_change_the_callers_random_numbers(trained):
    data = episodes(32, seed=2)
    first = torch.Generator().manual_seed(5)
    trained.diagnostics(data, first)
    second = torch.Generator().manual_seed(5)
    observations, _ = trained._states(data)
    trained.filter(observations, data.actions, second)
    trained.training_losses(data, second)
    assert torch.equal(torch.rand(3, generator=first), torch.rand(3, generator=second))
