import torch

from minecraft_rl import world_model
from minecraft_rl.imagination import rollout_errors
from minecraft_rl.tmaze import Action, Observation

CPU = torch.device("cpu")
FAST = world_model.Config(
    seed=0,
    corridor_length=1,
    max_steps=12,
    training_episodes=256,
    evaluation_episodes=128,
    learning_rate=0.01,
    steps=600,
)


def episodes(count: int = 32, seed: int = 0) -> world_model.Episodes:
    return world_model.collect_episodes(
        FAST, count, torch.Generator().manual_seed(seed)
    )


def test_first_imagined_step_is_the_teacher_forced_prediction():
    model, _ = world_model.build(FAST, CPU)
    data = episodes()
    with torch.no_grad():
        teacher_forced = model(data.observations, data.actions)
        for start in (0, 3, 7):
            imagined = model.open_loop(data, start, 1, None)
            assert torch.allclose(
                imagined.reward[:, 0], teacher_forced.reward[:, start], atol=1e-6
            )
            assert torch.allclose(
                imagined.observation_logits[:, 0],
                teacher_forced.observation_logits[:, start],
                atol=1e-6,
            )


def test_imagination_never_reads_real_observations_after_the_start():
    model, _ = world_model.build(FAST, CPU)
    data = episodes()
    start = 2
    altered = world_model.Episodes(
        *(getattr(data, f).clone() for f in world_model.Episodes.__dataclass_fields__)
    )
    altered.observations[:, start + 1 :] = Observation.ARM
    altered.next_observations[:, start:] = Observation.ARM
    with torch.no_grad():
        original = model.open_loop(data, start, 6, None)
        changed = model.open_loop(altered, start, 6, None)
    assert torch.equal(original.reward, changed.reward)
    assert torch.equal(original.observation_logits, changed.observation_logits)


def test_imagination_follows_the_real_actions():
    model, _ = world_model.build(FAST, CPU)
    data = episodes()
    altered = world_model.Episodes(
        *(getattr(data, f).clone() for f in world_model.Episodes.__dataclass_fields__)
    )
    altered.actions[:, 2] = (altered.actions[:, 2] + 1) % len(Action)
    with torch.no_grad():
        original = model.open_loop(data, 0, 4, None)
        changed = model.open_loop(altered, 0, 4, None)
    assert torch.equal(original.reward[:, :2], changed.reward[:, :2])
    assert not torch.equal(original.reward[:, 2:], changed.reward[:, 2:])


def test_horizon_is_clipped_at_the_padded_length():
    model, _ = world_model.build(FAST, CPU)
    with torch.no_grad():
        imagined = model.open_loop(episodes(), FAST.max_steps - 3, 20, None)
    assert imagined.reward.shape == (32, 3)


def test_pairs_count_every_start_whose_episode_reaches_the_horizon():
    model, _ = world_model.build(FAST, CPU)
    data = episodes()
    errors = rollout_errors(model, data, (1, 5, 10))
    lengths = data.mask.sum(1)
    for k in (1, 5, 10):
        assert errors[str(k)]["pairs"] == int((lengths - k + 1).clamp(min=0).sum())
        assert 0.0 <= errors[str(k)]["trajectory_accuracy"]
        assert (
            errors[str(k)]["trajectory_accuracy"]
            <= errors[str(k)]["observation_accuracy"]
        )


def test_trained_world_model_imagines_correctly_and_control_does_not():
    training = episodes(FAST.training_episodes, seed=0)
    evaluation = episodes(FAST.evaluation_episodes, seed=1)
    results = {}
    for memory in (True, False):
        model, optimizer = world_model.build(FAST, CPU, memory)
        generator = torch.Generator().manual_seed(0)
        world_model.train(model, optimizer, training, FAST, CPU, generator, FAST.steps)
        results[memory] = rollout_errors(model, evaluation, (1, 5, 10))

    for k in ("1", "5", "10"):
        assert results[True][k]["trajectory_accuracy"] == 1.0
        assert results[True][k]["junction_turn_sign_accuracy"] == 1.0
        assert results[True][k]["return_absolute_error"] < 0.1
        assert results[False][k]["junction_turn_sign_accuracy"] < 0.75
    assert (
        results[False]["10"]["return_absolute_error"]
        > 5 * results[True]["10"]["return_absolute_error"]
    )
