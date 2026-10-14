"""Tests of the first Minecraft actor pipeline: action space, reward head,
imagination, actor-critic gradients, the one-step filter and reward replay."""

import json

import numpy
import pytest
import torch

from minecraft_rl import runtime
from minecraft_rl.minecraft_actor import (
    CANONICAL_OBJECTIVE,
    ActionSpace,
    ActorConfig,
    Agent,
    Filter,
    RewardHead,
    imagine,
    load_world_model,
    losses,
    reward_metrics,
    train_actor,
)
from minecraft_rl.minecraft_dataset import EpisodeMeta, EpisodeWriter, load_episode
from minecraft_rl.minecraft_interface import PlayerAction, PolicyObservation
from minecraft_rl.minecraft_replay import CompactVocabulary, SequenceReplay
from minecraft_rl.minecraft_world_model import (
    ACTION_SIZE,
    MinecraftRSSM,
    ModelConfig,
    Objective,
    save_checkpoint,
)
from test_minecraft_interface import SCHEMA, observation_json
from test_minecraft_replay import COLUMNS, RAW_SIZES, ROWS, synthetic_episode
from test_minecraft_world_model import SMALL


@pytest.fixture
def setup():
    runtime.configure(0)
    episodes = [synthetic_episode(14, seed=s, episode=s) for s in range(3)]
    compact = CompactVocabulary.from_episodes(episodes, RAW_SIZES)
    vocabulary = compact.model_vocabulary(ROWS, COLUMNS, 32.0)
    torch.manual_seed(0)
    model = MinecraftRSSM(vocabulary, ModelConfig(**SMALL))
    replay = SequenceReplay(episodes, compact, 6)
    return model, replay, compact


def test_every_choice_maps_to_a_valid_player_action():
    generator = torch.Generator().manual_seed(0)
    for _ in range(200):
        choice = [
            int(torch.randint(size, (1,), generator=generator))
            for size in ActionSpace.SIZES
        ]
        action = ActionSpace.to_player_action(choice)
        assert isinstance(action, PlayerAction)
        assert not (action.forward and action.back) and not (
            action.left and action.right
        )
        assert not action.sprint or action.forward
        assert action.abstract.is_none
    noop = ActionSpace.to_player_action(ActionSpace.NOOP)
    assert noop.physical() == PlayerAction().physical()


def test_the_world_model_action_matches_the_dataset_encoding():
    choice = torch.tensor([[1, 1, 0, 1, 1, 0, 6, 0, 3]])
    vector = ActionSpace.world_model_action(choice)
    assert vector.shape == (1, ACTION_SIZE)
    action = ActionSpace.to_player_action(choice[0].tolist())
    assert action.forward and action.jump and action.sprint and action.attack
    assert (
        action.yaw_delta == 30.0 and action.pitch_delta == -10.0 and action.hotbar == 2
    )
    # Buttons in ACTION_BUTTONS order, then camera / 45, then the hotbar one-hot.
    assert vector[0, :9].tolist() == [1, 0, 0, 0, 1, 0, 1, 1, 0]
    assert vector[0, 9:11].tolist() == pytest.approx([30 / 45, -10 / 45])
    assert vector[0, 11 + 2] == 1.0


def test_imagination_shapes_and_finite_values(setup):
    model, replay, _ = setup
    agent, head = Agent(model.features, 32), RewardHead(model.features, 32)
    h = torch.zeros(5, model.hidden)
    z = torch.zeros(5, model.latent_size)
    imagined = imagine(model, head, agent, h, z, 4, torch.Generator().manual_seed(0))
    assert imagined.features.shape == (5, 5, model.features)
    assert imagined.choices.shape == (5, 4, len(ActionSpace.SIZES))
    for k, size in enumerate(ActionSpace.SIZES):
        assert int(imagined.choices[..., k].max()) < size
    assert (
        torch.isfinite(imagined.rewards).all()
        and torch.isfinite(imagined.continues).all()
    )


def test_actor_and_critic_get_gradients_and_the_world_model_stays_frozen(setup):
    model, replay, _ = setup
    for p in model.parameters():
        p.requires_grad_(False)
    before = [p.clone() for p in model.parameters()]
    agent, head = Agent(model.features, 32), RewardHead(model.features, 32)
    imagined = imagine(
        model,
        head,
        agent,
        torch.zeros(4, model.hidden),
        torch.zeros(4, model.latent_size),
        3,
        torch.Generator().manual_seed(0),
    )
    terms = losses(agent, imagined, ActorConfig(horizon=3))
    (terms["actor"] + terms["critic"]).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in agent.heads.parameters()
    )
    assert all(p.grad is not None for p in agent.critic.parameters())
    optimizer = torch.optim.Adam(agent.parameters())
    starts = (torch.zeros(8, model.hidden), torch.zeros(8, model.latent_size))
    history = train_actor(
        model,
        head,
        agent,
        optimizer,
        starts,
        ActorConfig(horizon=3, start_states=4),
        torch.Generator().manual_seed(1),
        2,
    )
    assert len(history) == 2 and all(numpy.isfinite(h["actor"]) for h in history)
    for a, b in zip(before, model.parameters(), strict=True):
        assert torch.equal(a, b)


def test_the_filter_follows_the_batch_posterior(setup):
    model, replay, compact = setup
    model.mode_latents = True
    batch = replay.batch([(0, 0)])
    filtered = model.filter(
        batch.observations, batch.actions, torch.Generator().manual_seed(0)
    )
    stepper = Filter(model, torch.Generator().manual_seed(0))
    # The filter takes dataset arrays; rebuild them from the replay tensors.
    names = {"air": "air_bubbles"}
    raw = {
        f"obs_{names.get(k, k)}": v.numpy() for k, v in replay.observations[0].items()
    }
    state = stepper.reset({k: v[0] for k, v in raw.items()})
    assert torch.allclose(state[0, : model.hidden], filtered["h"][0, 0])
    assert torch.allclose(state[0, model.hidden :], filtered["z"][0, 0])
    choice = torch.tensor([1, 0, 0, 0, 1, 0, 3, 2, 0])
    batch_action = ActionSpace.world_model_action(choice.unsqueeze(0))
    batch.actions[0, 0] = batch_action[0]
    refiltered = model.filter(
        batch.observations, batch.actions, torch.Generator().manual_seed(0)
    )
    state = stepper.step(choice, {k: v[1] for k, v in raw.items()})
    assert torch.allclose(state[0, : model.hidden], refiltered["h"][0, 1], atol=1e-6)
    model.mode_latents = False


def test_reward_metrics_separate_zero_and_positive_steps():
    target = torch.tensor([0.0, 0.0, 0.0, 0.1])
    predicted = torch.tensor([0.0, 0.01, 0.0, 0.08])
    out = reward_metrics(predicted, target)
    assert out["positive"]["count"] == 1
    assert out["positive"]["model_mae"] == pytest.approx(0.02)
    assert out["positive"]["zero_baseline_mae"] == pytest.approx(0.1)
    assert out["zero"]["model_mae"] == pytest.approx(0.01 / 3)
    assert out["positive_above_zero_rate"] == 1.0


def test_a_non_canonical_checkpoint_is_refused(setup, tmp_path):
    model, _, compact = setup
    path = tmp_path / "model.pt"
    save_checkpoint(
        path,
        model,
        {"vocabulary": compact.to_json(), "objective": Objective().to_json()},
    )
    with pytest.raises(ValueError, match="canonical"):
        load_world_model(path)
    assert CANONICAL_OBJECTIVE.pitch_loss_scale == 128.0


def test_reward_replay_round_trips_with_versions(tmp_path):
    first = PolicyObservation.from_json(observation_json(), SCHEMA)
    writer = EpisodeWriter(first)
    for t in range(3):
        writer.add(
            PlayerAction(forward=True),
            first,
            t == 2,
            t,
            t,
            t + 1,
            t,
            reward=0.1 * t,
            components=[0.1 * t, 0, 0, 0, 0, 0, 0, 0],
            events={"t": t},
        )
    meta = EpisodeMeta(
        0,
        100000,
        "normal",
        "x",
        "train",
        1,
        3,
        True,
        False,
        0,
        3,
        action_schema="player-action-v3",
        reward_version="reward-v1",
    )
    writer.write(tmp_path, meta)
    episode = load_episode(tmp_path)
    assert episode.meta.reward_version == "reward-v1"
    assert episode.arrays["reward"].tolist() == pytest.approx([0.0, 0.1, 0.2])
    assert episode.arrays["reward_components"].sum(1).tolist() == pytest.approx(
        [0.0, 0.1, 0.2]
    )
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert [e["t"] for e in events] == [0, 1, 2]


def test_an_episode_without_reward_still_loads(tmp_path):
    first = PolicyObservation.from_json(observation_json(), SCHEMA)
    writer = EpisodeWriter(first)
    writer.add(PlayerAction(), first, False, 0, 0, 1, 0)
    writer.write(
        tmp_path, EpisodeMeta(0, 1, "flat", "x", "train", 1, 1, False, True, 0, 1)
    )
    episode = load_episode(tmp_path)
    assert "reward" not in episode.arrays
