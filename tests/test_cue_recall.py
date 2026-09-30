import pytest
import torch

from minecraft_rl.cue_recall import (
    CHECKPOINT_FORMAT,
    CUE_A,
    CUE_B,
    DISTRACTOR,
    QUERY,
    Config,
    CueRecallGRU,
    build,
    cue_gradient,
    cue_recall_batch,
    evaluate,
    load_checkpoint,
    sample_batch,
    save_checkpoint,
    state_separation,
    train,
)

CPU = torch.device("cpu")
FAST = Config(seed=0, steps=200)


def trained(config: Config = FAST) -> CueRecallGRU:
    model, optimizer = build(config, CPU)
    train(
        model,
        optimizer,
        config,
        CPU,
        torch.Generator().manual_seed(config.seed),
        config.steps,
    )
    return model


def test_sequences_differ_only_in_the_cue():
    tokens, targets = cue_recall_batch(torch.tensor([CUE_A, CUE_B]), delay=3)
    assert tokens.shape == (2, 5)
    assert tokens[:, 0].tolist() == [CUE_A, CUE_B]
    assert torch.equal(tokens[0, 1:], tokens[1, 1:])
    assert tokens[0, 1:-1].tolist() == [DISTRACTOR] * 3
    assert tokens[0, -1] == QUERY
    assert targets.tolist() == [CUE_A, CUE_B]


def test_sampled_batches_stay_within_the_training_delays():
    generator = torch.Generator().manual_seed(0)
    lengths = {sample_batch(generator, FAST)[0].shape[1] for _ in range(200)}
    assert lengths == set(range(FAST.min_delay + 2, FAST.max_delay + 3))


def test_no_memory_control_cannot_tell_the_cues_apart():
    control, _ = build(FAST, CPU, memory=False)
    tokens, _ = cue_recall_batch(torch.tensor([CUE_A, CUE_B]), delay=4)
    with torch.no_grad():
        logits = control(tokens)
    assert torch.equal(logits[0], logits[1])


def test_gru_learns_to_recall_the_cue_beyond_the_training_delays():
    model = trained()
    for delay in (1, FAST.max_delay, 2 * FAST.max_delay):
        assert evaluate(model, delay, CPU)["accuracy"] == 1.0


def test_trained_prediction_depends_on_the_first_input():
    model = trained()
    assert cue_gradient(model, FAST.max_delay, CPU) > 1e-3
    distances = state_separation(model, FAST.max_delay, CPU)
    assert len(distances) == FAST.max_delay + 2
    assert min(distances) > 0.1


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path):
    config = Config(seed=0, steps=40)
    uninterrupted, optimizer = build(config, CPU)
    train(uninterrupted, optimizer, config, CPU, torch.Generator().manual_seed(0), 40)

    generator = torch.Generator().manual_seed(0)
    resumed, optimizer = build(config, CPU)
    train(resumed, optimizer, config, CPU, generator, 20)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, config, 20, resumed, optimizer)
    restored_config, step, restored, restored_optimizer = load_checkpoint(path, CPU)
    train(restored, restored_optimizer, config, CPU, generator, 20)

    assert restored_config == config
    assert step == 20
    for a, b in zip(uninterrupted.parameters(), restored.parameters(), strict=True):
        assert torch.equal(a, b)


def test_load_rejects_a_foreign_checkpoint(tmp_path):
    path = tmp_path / "other.pt"
    torch.save({"format": "parity-mlp-v1"}, path)
    with pytest.raises(ValueError, match=CHECKPOINT_FORMAT):
        load_checkpoint(path, CPU)
