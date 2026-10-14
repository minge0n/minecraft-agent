"""Tests of reward prediction with full episode context: reward timing,
episode boundaries, recurrent state carry and reset, trainable parameter
groups, reward training and resume."""

import numpy
import pytest
import torch

from minecraft_rl import runtime
from minecraft_rl.minecraft_actor import RewardHead
from minecraft_rl.minecraft_dataset import Episode
from minecraft_rl.minecraft_replay import CompactVocabulary
from minecraft_rl.minecraft_reward import COMPONENTS
from minecraft_rl.minecraft_reward_model import (
    TRAINABLE_MODULES,
    ChunkSampler,
    RecurrentTrainer,
    RewardTraining,
    average_precision,
    chronological_states,
    episode_data,
    event_labels,
    filter_episode,
    history_labels,
    probe_auc,
    report_on,
    reward_report,
    set_trainable,
    train_frozen_head,
    window_states,
)
from minecraft_rl.minecraft_world_model import (
    MinecraftRSSM,
    ModelConfig,
    Objective,
)
from test_minecraft_replay import COLUMNS, RAW_SIZES, ROWS, synthetic_episode
from test_minecraft_world_model import SMALL

STEPS = 40
REWARD_AT = (0, 9, 25)


def rewarded(episode: Episode, at=REWARD_AT) -> Episode:
    reward = numpy.zeros(episode.steps, dtype=numpy.float32)
    components = numpy.zeros((episode.steps, len(COMPONENTS)), dtype=numpy.float32)
    for t in at:
        reward[t] = 0.01
        components[t, COMPONENTS.index("block_seen_novelty")] = 0.01
    episode.arrays["reward"] = reward
    episode.arrays["reward_components"] = components
    return episode


@pytest.fixture
def setup():
    runtime.configure(0)
    episodes = [rewarded(synthetic_episode(STEPS, seed=s, episode=s)) for s in range(2)]
    compact = CompactVocabulary.from_episodes(episodes, RAW_SIZES)
    torch.manual_seed(0)
    model = MinecraftRSSM(
        compact.model_vocabulary(ROWS, COLUMNS, 32.0), ModelConfig(**SMALL)
    )
    model.eval()
    return model, episode_data(model, episodes, compact)


def test_reward_t_is_paired_with_the_state_after_action_t(setup):
    model, data = setup
    states = chronological_states(model, data[:1], seed=0)
    assert states[0].shape[0] == data[0].steps + 1

    class Probe(RewardHead):
        """Returns the index of the state it reads, coded in its first value."""

        def forward(self, s):
            return s[..., 0]

    head = Probe(model.features)
    coded = torch.arange(data[0].steps + 1).float().unsqueeze(-1)
    report = report_on(head, [coded.expand(-1, model.features)], data[:1])
    # reward[t] is predicted from s_{t+1}, so the mean of the predicted
    # indices is the mean of 1..T.
    assert report["predicted_reward_per_step"] == pytest.approx((data[0].steps + 1) / 2)


def test_the_chronological_filter_carries_h_and_resets_per_episode(setup):
    model, data = setup
    # h_0 is zero at every episode start.
    for states in chronological_states(model, data, seed=0):
        assert torch.equal(states[0, : model.hidden], torch.zeros(model.hidden))
    # With deterministic latents, the first window matches the carried filter
    # and later windows differ from it: restarting h drops the history.
    model.mode_latents = True
    try:
        exact = filter_episode(model, data[0], None)
        restarted = window_states(model, data[0], 8, seed=0)
        assert torch.allclose(restarted[:7], exact[1:8])
        assert not torch.allclose(restarted[8:15], exact[9:16])
    finally:
        model.mode_latents = False


def test_event_labels_split_first_seen_later_seen_and_repeats(setup):
    episode = rewarded(synthetic_episode(STEPS, seed=0), at=(0, 5))
    labels = event_labels(episode)
    assert labels["seen_first_step"].tolist().count(True) == 1
    assert labels["seen_later"][5] and not labels["seen_later"][0]
    assert labels["positive"].sum() == 2
    assert (labels["zero_repeat_view"] | labels["zero_other"]).sum() == STEPS - 2
    assert not (labels["zero_repeat_view"] & labels["positive"]).any()


def test_the_report_counts_precision_recall_and_false_positives():
    target = torch.tensor([0.0, 0.0, 0.01, 0.0, 0.05])
    predicted = torch.tensor([0.0, 0.03, 0.01, 0.0, 0.04])
    labels = {
        "positive": target > 0,
        "zero": target == 0,
        **{
            name: torch.zeros(5, dtype=torch.bool)
            for name in (
                "seen_first_step",
                "seen_later",
                "broken",
                "acquired",
                "zero_repeat_view",
                "zero_other",
            )
        },
    }
    out = reward_report(predicted, target, labels)
    assert out["positive_count"] == 2
    assert out["at_0.005"]["precision"] == pytest.approx(2 / 3)
    assert out["at_0.005"]["recall"] == 1.0
    assert out["at_0.005"]["false_positive_rate"] == pytest.approx(1 / 3)
    assert out["at_0.02"]["recall"] == 0.5
    assert out["positive_zero_baseline_mae"] == pytest.approx(0.03)
    # Ranking: 0.04 (pos), 0.03 (neg), 0.01 (pos): AP = (1 + 2/3) / 2.
    assert average_precision(predicted, labels["positive"]) == pytest.approx(5 / 6)


def test_positive_chunks_contain_their_rewarded_step(setup):
    _, data = setup
    config = RewardTraining(chunk=8, batch=64, positive_chunk_share=1.0)
    sampler = ChunkSampler(data, config)
    for e, start in sampler.sample(torch.Generator().manual_seed(0)):
        window = data[e].rewards[start : start + 8]
        assert window.shape[0] == 8 and (window > 0).any()


def test_a_frozen_head_overfits_a_known_episode(setup):
    model, data = setup
    states = chronological_states(model, data[:1], seed=0)
    config = RewardTraining(chunk=8, batch=8)
    head, _ = train_frozen_head(states, data[:1], model.features, config, 400, seed=0)
    out = report_on(head, states, data[:1])
    assert out["average_precision"] == 1.0
    assert out["at_0.005"]["false_positive_rate"] == 0.0


def test_only_the_recurrent_dynamics_are_trainable(setup):
    model, _ = setup
    set_trainable(model)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable
    assert all(name.split(".")[0] in TRAINABLE_MODULES for name in trainable)
    for frozen in ("encoder", "decoder", "continue_head"):
        assert not any(p.requires_grad for p in getattr(model, frozen).parameters())


def test_recurrent_training_changes_only_the_dynamics_and_resumes(setup):
    model, data = setup
    before = {name: p.clone() for name, p in model.state_dict().items()}
    config = RewardTraining(chunk=8, batch=4)
    objective = Objective(reduction="group_mean", component_scale=8.0)
    trainer = RecurrentTrainer(model, objective, data, config, 1e-3, 2, seed=0)
    for _ in range(3):
        record = trainer.step()
        assert all(numpy.isfinite(v) for v in record.values())
    after = model.state_dict()
    for name, value in before.items():
        moved = not torch.equal(value, after[name])
        assert moved == (name.split(".")[0] in TRAINABLE_MODULES), name
    saved = trainer.state()
    expected = trainer.step()

    torch.manual_seed(5)
    other = MinecraftRSSM(model.vocabulary, model.config)
    resumed = RecurrentTrainer(other, objective, data, config, 1e-3, 2, seed=1)
    resumed.load(saved)
    record = resumed.step()
    assert record["total"] == pytest.approx(expected["total"], rel=1e-5)


def test_history_labels_mark_types_seen_before_and_out_of_view_now():
    episode = synthetic_episode(4, seed=0)
    kinds = numpy.zeros((5, 2), dtype=numpy.int8)
    types = numpy.zeros((5, 2), dtype=numpy.int16)
    kinds[0, 0], types[0, 0] = 1, 7  # type 7 in view at t = 0 only
    kinds[3, 0], types[3, 0] = 1, 7  # and again at t = 3
    episode.arrays["obs_ray_kind"], episode.arrays["obs_ray_type"] = kinds, types
    labels = history_labels(episode, [7, 9])
    assert labels[:, 0].tolist() == [False, True, True, False, True]
    assert not labels[:, 1].any()


def test_a_probe_separates_a_linear_signal_and_not_noise():
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(400, 4, generator=generator)
    y = x[:, 0] > 0.5
    assert probe_auc(x[:200], y[:200], x[200:], y[200:]) > 0.95
    noise = torch.rand(400, generator=generator) > 0.7
    assert abs(probe_auc(x[:200], noise[:200], x[200:], noise[200:]) - 0.5) < 0.15
