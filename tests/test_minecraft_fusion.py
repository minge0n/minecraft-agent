import platform

import pytest
import torch

from minecraft_rl import macos_qos, runtime
from minecraft_rl.minecraft_probes import STAGES, probe, saturation, stage_probes
from minecraft_rl.minecraft_replay import CompactVocabulary, SequenceReplay
from minecraft_rl.minecraft_world_model import (
    CHECKPOINT_FORMAT,
    FUSIONS,
    MinecraftRSSM,
    ModelConfig,
    Objective,
    load_checkpoint,
    save_checkpoint,
)
from minecraft_rl.minecraft_world_model_eval import (
    action_pitch,
    camera_of,
    imagination,
    one_step,
    ray_frequency_baseline,
    slot_frequency_baseline,
    unknown_classes,
)
from test_minecraft_replay import COLUMNS, RAW_SIZES, ROWS, synthetic_episode
from test_minecraft_world_model import SMALL

GROUP = Objective(reduction="group_mean", component_scale=float(ROWS * COLUMNS))


@pytest.fixture
def setup():
    runtime.configure(0)
    episodes = [synthetic_episode(14, seed=s, episode=s) for s in range(3)]
    compact = CompactVocabulary.from_episodes(episodes, RAW_SIZES)
    vocabulary = compact.model_vocabulary(ROWS, COLUMNS, 32.0)
    replay = SequenceReplay(episodes, compact, 6)
    return compact, vocabulary, replay


def model_for(vocabulary, fusion: str) -> MinecraftRSSM:
    torch.manual_seed(0)
    return MinecraftRSSM(vocabulary, ModelConfig(fusion=fusion, self_skip=6, **SMALL))


@pytest.mark.parametrize("fusion", FUSIONS)
def test_encoder_output_size_matches_the_posterior_input(setup, fusion):
    _, vocabulary, replay = setup
    model = model_for(vocabulary, fusion)
    batch = replay.sample(3, torch.Generator().manual_seed(0))
    embedded = model.encoder(batch.observations)
    assert embedded.shape == (3, 7, model.encoder.outputs)
    grid, embed = model.encoder.grid.outputs, SMALL["embed_dim"]
    expected = {"shared": embed, "self_skip": embed + 6, "late": grid + embed}
    assert model.encoder.outputs == expected[fusion]
    first = model.posterior_net[0]
    assert first.in_features == model.hidden + model.encoder.outputs
    taps = model.encoder.taps(batch.observations)
    assert torch.allclose(taps["encoder_output"].reshape(3, 7, -1), embedded)


@pytest.mark.parametrize("fusion", ["self_skip", "late"])
def test_the_self_state_reaches_the_posterior_beside_the_shared_layer(setup, fusion):
    _, vocabulary, replay = setup
    model = model_for(vocabulary, fusion)
    o = replay.batch([(0, 0)]).observations
    changed = {k: v.clone() for k, v in o.items()}
    changed["pitch"] = changed["pitch"] + 30.0
    h = torch.zeros(7, model.hidden)
    before = model.posterior_logits(h, model.encoder(o)[0])
    after = model.posterior_logits(h, model.encoder(changed)[0])
    assert not torch.allclose(before, after)
    # The self-state part of e does not pass the shared layer.
    taps = model.encoder.taps(o)
    self_part = taps["encoder_output"][:, -taps["self_output"].shape[-1] :]
    if fusion == "late":
        assert torch.equal(self_part, taps["self_output"])
    else:
        assert torch.allclose(
            taps["encoder_output"][:, -6:], model.encoder.skip(taps["self_output"])
        )


@pytest.mark.parametrize("fusion", FUSIONS)
def test_every_parameter_of_each_fusion_gets_a_finite_gradient(setup, fusion):
    _, vocabulary, replay = setup
    model = model_for(vocabulary, fusion)
    batch = replay.sample(4, torch.Generator().manual_seed(1))
    terms = model.losses(
        batch.observations,
        batch.actions,
        batch.continues,
        torch.Generator().manual_seed(2),
        GROUP,
    )
    assert all(torch.isfinite(v) for v in terms.values())
    terms["total"].backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        # The synthetic data has no armor or offhand item, so the masks give
        # their decoder heads no signal. Every part of the observation path
        # must learn.
        if name.startswith(("encoder.", "posterior_net.")):
            assert parameter.grad.abs().sum() > 0, name


def test_the_late_fusion_has_no_shared_layer(setup):
    _, vocabulary, _ = setup
    late = model_for(vocabulary, "late")
    assert late.encoder.out is None
    assert not any(
        name.startswith("encoder.out.") for name, _ in late.named_parameters()
    )


@pytest.mark.parametrize("fusion", FUSIONS)
def test_imagination_never_calls_the_encoder(setup, fusion):
    _, vocabulary, replay = setup
    model = model_for(vocabulary, fusion)
    batch = replay.sample(2, torch.Generator().manual_seed(0))
    calls = []
    model.encoder.register_forward_hook(lambda *a: calls.append("encoder"))
    states = model.imagine(
        torch.zeros(2, model.hidden),
        torch.zeros(2, model.latent_size),
        batch.actions,
        torch.Generator().manual_seed(0),
    )
    assert states.shape == (2, 6, model.features)
    assert calls == []


@pytest.mark.parametrize("fusion", FUSIONS)
def test_a_checkpoint_keeps_its_fusion(setup, tmp_path, fusion):
    _, vocabulary, replay = setup
    model = model_for(vocabulary, fusion)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, {})
    assert torch.load(path, weights_only=True)["format"] == CHECKPOINT_FORMAT
    loaded, _ = load_checkpoint(path)
    assert loaded.config.fusion == fusion
    o = replay.batch([(1, 2)]).observations
    with torch.no_grad():
        assert torch.equal(model.encoder(o), loaded.encoder(o))


def test_a_v1_checkpoint_loads_as_the_shared_encoder(setup, tmp_path):
    _, vocabulary, _ = setup
    model = model_for(vocabulary, "shared")
    config = model.config.to_json()
    del config["fusion"], config["self_skip"]
    del config["heads"], config["item_slot_features"]
    data = {
        "format": "minecraft-rssm-checkpoint-v1",
        "config": config,
        "vocabulary": vars(vocabulary) | {},
        "state_dict": model.state_dict(),
        "extra": {},
    }
    path = tmp_path / "old.pt"
    torch.save(data, path)
    loaded, _ = load_checkpoint(path)
    assert loaded.config.fusion == "shared"
    for (a, p), (b, q) in zip(
        model.state_dict().items(), loaded.state_dict().items(), strict=True
    ):
        assert a == b and torch.equal(p, q)


def test_a_checkpoint_never_falls_back_to_a_default_fusion(setup, tmp_path):
    _, vocabulary, _ = setup
    model = model_for(vocabulary, "late")
    config = model.config.to_json()
    del config["fusion"]
    path = tmp_path / "broken.pt"
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "config": config,
            "vocabulary": vars(vocabulary) | {},
            "state_dict": model.state_dict(),
            "extra": {},
        },
        path,
    )
    with pytest.raises(ValueError, match="needs a fusion"):
        load_checkpoint(path)
    shared = model_for(vocabulary, "shared")
    late_weights = tmp_path / "mismatch.pt"
    config = shared.config.to_json()
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "config": config,
            "vocabulary": vars(vocabulary) | {},
            "state_dict": model.state_dict(),
            "extra": {},
        },
        late_weights,
    )
    with pytest.raises(RuntimeError):
        load_checkpoint(late_weights)


def test_an_unknown_fusion_is_rejected(setup):
    _, vocabulary, _ = setup
    with pytest.raises(ValueError, match="fusion"):
        MinecraftRSSM(vocabulary, ModelConfig(fusion="bridge", **SMALL))


@pytest.mark.parametrize("fusion", FUSIONS)
def test_stage_probes_cover_every_stage_and_change_nothing(setup, fusion):
    _, vocabulary, replay = setup
    model = model_for(vocabulary, fusion)
    before = [p.detach().clone() for p in model.parameters()]
    fit = replay.batch([(0, 0), (1, 0), (2, 0)])
    test = replay.batch([(0, 8), (1, 8), (2, 8)])
    result = stage_probes(model, fit, test)
    assert set(result) == {*STAGES, "mean_predictor"}
    assert all(v["mae"] >= 0 for v in result.values())
    for p, q in zip(before, model.parameters(), strict=True):
        assert torch.equal(p, q)
    shares = saturation(model, test)
    if fusion == "late":
        assert shares["fusion_output_below_minus_0_95"] is None
    else:
        assert 0.0 <= shares["fusion_output_below_minus_0_95"] <= 1.0


def test_a_probe_recovers_a_linear_target_and_not_noise():
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(200, 5, generator=generator)
    y = 3.0 * x[:, 0] - x[:, 2] + 1.0
    exact = probe(x[:150], y[:150], x[150:], y[150:])
    assert exact["mae"] < 0.05
    noise = torch.randn(200, generator=generator)
    assert probe(x[:150], noise[:150], x[150:], noise[150:])["mae"] > 0.5


def test_the_action_pitch_rule_follows_the_environment():
    pitch = torch.tensor([10.0, 88.0, -89.0])
    camera = torch.tensor(
        [
            [[0.0, 5.0], [0.0, -2.0]],
            [[3.0, 5.0], [0.0, -1.0]],
            [[0.0, -4.0], [0.0, 3.0]],
        ]
    )
    expected = torch.tensor([[15.0, 13.0], [90.0, 89.0], [-90.0, -87.0]])
    assert torch.equal(action_pitch(pitch, camera), expected)


def test_camera_of_reads_degrees_from_an_action_vector(setup):
    _, _, replay = setup
    batch = replay.batch([(0, 0)])
    raw = torch.as_tensor(replay.episodes[0].arrays["action_camera"][:6])
    assert torch.allclose(camera_of(batch.actions[0]), raw.float(), atol=1e-5)


def test_evaluation_reports_the_pitch_baselines_and_camera_windows(setup):
    compact, vocabulary, replay = setup
    model = model_for(vocabulary, "late")
    unknown = unknown_classes(compact, model)
    index = replay.evaluation_windows(6)
    result = one_step(
        model,
        replay,
        index,
        2,
        unknown,
        ray_frequency_baseline(replay, model, unknown),
        slot_frequency_baseline(replay),
        torch.Generator().manual_seed(0),
    )
    # The synthetic pitch does not follow the camera rule, so the rule is not
    # exact here; on real data its error is below 1e-6 degrees.
    assert result["all"]["action_pitch"]["pitch_mae"]["count"] > 0
    assert "action_pitch_shuffled" in result["all"]
    rollouts = imagination(
        model, replay, index, 1, unknown, torch.Generator().manual_seed(0)
    )
    assert rollouts["windows"]["all"] == len(index)
    assert "action_pitch" in rollouts["horizon_5"]
    assert "camera_horizon_5" in rollouts or rollouts["windows"]["camera"] == 0


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS policy")
def test_the_background_policy_is_set_for_the_whole_process():
    import subprocess
    import sys

    code = (
        "from minecraft_rl import macos_qos; "
        "print(macos_qos.background_policy(), macos_qos.run_in_background())"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == ["False", "True"]


def test_the_background_policy_is_a_no_op_off_macos(monkeypatch):
    monkeypatch.setattr(macos_qos.sys, "platform", "linux")
    assert macos_qos.run_in_background() is None
    assert macos_qos.background_policy() is None
