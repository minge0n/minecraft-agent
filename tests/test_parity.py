import pytest
import torch

from minecraft_rl.parity import (
    CHECKPOINT_FORMAT,
    Config,
    ParityMLP,
    build,
    evaluate,
    gradients_match_finite_differences,
    load_checkpoint,
    parameter_changes,
    parity_dataset,
    save_checkpoint,
    select_device,
    train,
)

CPU = torch.device("cpu")


def test_dataset_holds_every_pattern_with_its_parity():
    inputs, labels = parity_dataset(4)
    assert inputs.shape == (16, 4)
    assert labels.shape == (16,)
    assert set(inputs.unique().tolist()) == {-1.0, 1.0}
    bits = (inputs > 0).long()
    assert torch.equal(labels, bits.sum(dim=1) % 2)
    assert len({tuple(row) for row in bits.tolist()}) == 16


def test_model_maps_inputs_to_two_logits():
    model = ParityMLP(bits=4, hidden=16)
    assert model(torch.zeros(5, 4)).shape == (5, 2)
    assert sum(p.numel() for p in model.parameters()) == 4 * 16 + 16 + 16 * 2 + 2


def test_initialization_is_reproducible_from_the_seed():
    first, _ = build(Config(seed=3), CPU)
    second, _ = build(Config(seed=3), CPU)
    other, _ = build(Config(seed=4), CPU)
    for a, b, c in zip(
        first.parameters(), second.parameters(), other.parameters(), strict=True
    ):
        assert torch.equal(a, b)
        assert not torch.equal(a, c)


def test_autograd_gradients_match_finite_differences():
    model, _ = build(Config(seed=0), CPU)
    assert gradients_match_finite_differences(model, 4)


def test_one_step_changes_every_parameter():
    config = Config(seed=0)
    inputs, labels = parity_dataset(config.bits)
    model, optimizer = build(config, CPU)
    changes = parameter_changes(model, optimizer, inputs, labels)
    assert set(changes) == {name for name, _ in model.named_parameters()}
    for change in changes.values():
        assert change["gradient_norm"] > 0.0
        assert change["update_norm"] > 0.0


def test_training_learns_parity():
    config = Config(seed=0)
    inputs, labels = parity_dataset(config.bits)
    model, optimizer = build(config, CPU)
    before = evaluate(model, inputs, labels)
    train(model, optimizer, inputs, labels, config.steps)
    after = evaluate(model, inputs, labels)
    assert after["accuracy"] == 1.0
    assert after["loss"] < 0.1 * before["loss"]


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path):
    config = Config(seed=0, steps=40)
    inputs, labels = parity_dataset(config.bits)

    uninterrupted, optimizer = build(config, CPU)
    train(uninterrupted, optimizer, inputs, labels, 40)

    resumed, optimizer = build(config, CPU)
    train(resumed, optimizer, inputs, labels, 20)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, config, 20, resumed, optimizer)
    restored_config, step, restored, restored_optimizer = load_checkpoint(path, CPU)
    train(restored, restored_optimizer, inputs, labels, 20)

    assert restored_config == config
    assert step == 20
    for a, b in zip(uninterrupted.parameters(), restored.parameters(), strict=True):
        assert torch.equal(a, b)


def test_load_rejects_a_foreign_checkpoint(tmp_path):
    path = tmp_path / "other.pt"
    torch.save({"format": "something-else"}, path)
    with pytest.raises(ValueError, match=CHECKPOINT_FORMAT):
        load_checkpoint(path, CPU)


def test_device_selection():
    assert select_device("cpu") == CPU
    assert select_device("auto").type in {"cpu", "cuda", "mps"}
    with pytest.raises(ValueError):
        select_device("tpu")
    if not torch.cuda.is_available():
        with pytest.raises(ValueError):
            select_device("cuda")
