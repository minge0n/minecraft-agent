import pytest
import torch

from minecraft_rl.tmaze import Action, Cue, Observation, TMaze
from minecraft_rl.world_model import (
    CHECKPOINT_FORMAT,
    Config,
    build,
    collect_episodes,
    evaluate,
    load_checkpoint,
    losses,
    save_checkpoint,
    train,
)

CPU = torch.device("cpu")
FAST = Config(
    seed=0,
    corridor_length=1,
    max_steps=12,
    training_episodes=256,
    evaluation_episodes=128,
    learning_rate=0.01,
    steps=600,
)


def walk(environment: TMaze, actions: list[Action]) -> list[tuple]:
    return [environment.step(action) for action in actions]


def test_cue_is_visible_only_at_the_start():
    environment = TMaze(corridor_length=2)
    assert environment.reset(Cue.RIGHT) == Observation.CUE_RIGHT
    steps = walk(environment, [Action.FORWARD] * 3)
    assert [s[0] for s in steps] == [
        Observation.CORRIDOR,
        Observation.CORRIDOR,
        Observation.JUNCTION,
    ]
    assert environment.reset(Cue.LEFT) == Observation.CUE_LEFT


def test_turning_in_the_corridor_bumps_into_the_wall():
    environment = TMaze(corridor_length=2)
    environment.reset(Cue.LEFT)
    observation, reward, terminated, truncated = environment.step(Action.LEFT)
    assert (observation, reward, terminated, truncated) == (
        Observation.CUE_LEFT,
        0.0,
        False,
        False,
    )
    assert environment.position == 0


@pytest.mark.parametrize(
    ("cue", "turn", "reward"),
    [
        (Cue.LEFT, Action.LEFT, 1.0),
        (Cue.LEFT, Action.RIGHT, -1.0),
        (Cue.RIGHT, Action.RIGHT, 1.0),
        (Cue.RIGHT, Action.LEFT, -1.0),
    ],
)
def test_junction_turn_is_rewarded_by_the_cue(cue, turn, reward):
    environment = TMaze(corridor_length=1)
    environment.reset(cue)
    walk(environment, [Action.FORWARD, Action.FORWARD])
    assert environment.step(turn) == (Observation.ARM, reward, True, False)
    with pytest.raises(RuntimeError):
        environment.step(Action.FORWARD)


def test_forward_at_the_junction_stays_and_episode_truncates():
    environment = TMaze(corridor_length=1, max_steps=4)
    environment.reset(Cue.LEFT)
    steps = walk(environment, [Action.FORWARD] * 4)
    assert [s[0] for s in steps] == [Observation.CORRIDOR] + [Observation.JUNCTION] * 3
    assert [s[3] for s in steps] == [False, False, False, True]


def test_collected_episodes_are_consistent_transitions():
    episodes = collect_episodes(FAST, 64, torch.Generator().manual_seed(0))
    lengths = episodes.mask.sum(1)
    assert episodes.observations.shape == (64, FAST.max_steps)
    assert torch.all(lengths >= 1)
    assert torch.all(episodes.mask[:, 0])
    for episode in range(64):
        length = int(lengths[episode])
        assert not episodes.mask[episode, length:].any()
        first = episodes.observations[episode, 0]
        assert first == Observation(int(episodes.cues[episode]))
        assert torch.equal(
            episodes.observations[episode, 1:length],
            episodes.next_observations[episode, : length - 1],
        )
        terminal = episodes.continues[episode, :length] == 0
        assert terminal.sum() <= 1
        if terminal.any():
            assert terminal[-1]
            assert episodes.rewards[episode, length - 1] in (-1.0, 1.0)


def test_weighted_behavior_policy_draws_only_allowed_actions():
    episodes = collect_episodes(
        FAST, 8, torch.Generator().manual_seed(0), action_weights=(1.0, 0.0, 0.0)
    )
    assert torch.all(episodes.actions[episodes.mask] == Action.FORWARD)
    assert torch.all(episodes.mask.sum(1) == FAST.max_steps)
    assert torch.all(episodes.continues[episodes.mask] == 1.0)


def test_padding_does_not_contribute_to_the_loss():
    episodes = collect_episodes(FAST, 16, torch.Generator().manual_seed(0))
    model, _ = build(FAST, CPU)
    before = losses(model, episodes)["total"]
    padding = ~episodes.mask
    episodes.next_observations[padding] = Observation.ARM
    episodes.rewards[padding] = 5.0
    episodes.continues[padding] = 1.0
    assert torch.equal(before, losses(model, episodes)["total"])


def test_world_model_learns_what_needs_memory_and_control_cannot():
    training = collect_episodes(
        FAST, FAST.training_episodes, torch.Generator().manual_seed(0)
    )
    evaluation = collect_episodes(
        FAST, FAST.evaluation_episodes, torch.Generator().manual_seed(1)
    )
    results = {}
    for memory in (True, False):
        model, optimizer = build(FAST, CPU, memory)
        initial = evaluate(model, evaluation, CPU)
        generator = torch.Generator().manual_seed(0)
        train(model, optimizer, training, FAST, CPU, generator, FAST.steps)
        results[memory] = (initial, evaluate(model, evaluation, CPU))

    initial, final = results[True]
    assert final["loss"]["total"] < 0.1 * initial["loss"]["total"]
    assert final["observation_accuracy"] == 1.0
    assert final["junction_turn"]["reward_sign_accuracy"] == 1.0
    assert final["junction_turn"]["reward_mse"] < 0.05

    _, control = results[False]
    assert control["junction_turn"]["reward_mse"] > 0.9
    assert control["junction_turn"]["reward_sign_accuracy"] < 0.7


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path):
    config = Config(seed=0, corridor_length=1, max_steps=12, steps=40)
    episodes = collect_episodes(config, 32, torch.Generator().manual_seed(0))
    uninterrupted, optimizer = build(config, CPU)
    generator = torch.Generator().manual_seed(0)
    train(uninterrupted, optimizer, episodes, config, CPU, generator, 40)

    generator = torch.Generator().manual_seed(0)
    resumed, optimizer = build(config, CPU)
    train(resumed, optimizer, episodes, config, CPU, generator, 20)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, config, 20, resumed, optimizer)
    restored_config, step, restored, restored_optimizer = load_checkpoint(path, CPU)
    train(restored, restored_optimizer, episodes, config, CPU, generator, 20)

    assert restored_config == config
    assert step == 20
    for a, b in zip(uninterrupted.parameters(), restored.parameters(), strict=True):
        assert torch.equal(a, b)


def test_load_rejects_a_foreign_checkpoint(tmp_path):
    path = tmp_path / "other.pt"
    torch.save({"format": "cue-recall-gru-v1"}, path)
    with pytest.raises(ValueError, match=CHECKPOINT_FORMAT):
        load_checkpoint(path, CPU)
