"""Tests of the temporal attack controller, its imagination, the life
lifecycle without a time limit, and life segments."""

import numpy
import pytest
import torch

from minecraft_rl import runtime
from minecraft_rl.actor_critic import lambda_returns
from minecraft_rl.minecraft_actor import (
    ActionSpace,
    ActorConfig,
    Imagined,
    RewardHead,
    losses,
)
from minecraft_rl.minecraft_dataset import (
    join_segments,
    load_episodes,
)
from minecraft_rl.minecraft_interface import PolicyObservation
from minecraft_rl.minecraft_life import LifeConfig, play_life
from minecraft_rl.minecraft_replay import CompactVocabulary, SequenceReplay
from minecraft_rl.minecraft_reward_break import ATTACK_FACTOR
from minecraft_rl.minecraft_reward_pitch import PITCH_FACTOR
from minecraft_rl.minecraft_temporal_attack import (
    HOLD_ATTACK_20,
    HOLD_TICKS,
    RELEASE_ATTACK,
    AttackController,
    TemporalBreakAgent,
    imagine_temporal,
    temporal_step,
)
from minecraft_rl.minecraft_temporal_train import controller_state
from minecraft_rl.minecraft_world_model import MinecraftRSSM, ModelConfig
from test_minecraft_interface import SCHEMA, observation_json
from test_minecraft_replay import COLUMNS, RAW_SIZES, ROWS, synthetic_episode
from test_minecraft_reward import REGISTRIES
from test_minecraft_world_model import SMALL


def run(commands):
    """Physical attack per tick for a list of commands offered every tick."""
    controller = AttackController.start(1)
    out, deciding = [], []
    for command in commands:
        deciding.append(bool(controller.deciding[0]))
        out.append(bool(controller.advance(torch.tensor([command]))[0]))
    return out, deciding


def test_a_hold_lasts_exactly_20_ticks_and_ignores_commands_inside():
    attack, deciding = run([HOLD_ATTACK_20] + [RELEASE_ATTACK] * 25)
    assert attack == [True] * HOLD_TICKS + [False] * 6
    assert deciding == [True] + [False] * 19 + [True] * 6


def test_release_is_one_tick_and_the_next_tick_decides_again():
    attack, deciding = run([RELEASE_ATTACK, HOLD_ATTACK_20, RELEASE_ATTACK])
    assert attack == [False, True, True]
    assert deciding == [True, True, False]


def test_two_consecutive_holds_give_40_attack_ticks_with_two_decisions():
    attack, deciding = run([HOLD_ATTACK_20] * 41 + [RELEASE_ATTACK])
    assert attack[:40] == [True] * 40
    assert sum(deciding[:40]) == 2 and deciding[0] and deciding[20]
    assert deciding[40] and attack[40]


def test_controller_state_of_data_starts():
    run_lengths = torch.tensor([0, 1, 5, 19, 20, 21])
    assert controller_state(run_lengths).tolist() == [0, 19, 15, 1, 0, 19]


@pytest.fixture
def world():
    runtime.configure(0)
    episodes = [synthetic_episode(14, seed=s, episode=s) for s in range(2)]
    compact = CompactVocabulary.from_episodes(episodes, RAW_SIZES)
    torch.manual_seed(0)
    model = MinecraftRSSM(
        compact.model_vocabulary(ROWS, COLUMNS, 32.0), ModelConfig(**SMALL)
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    head = RewardHead(model.features, 16)
    for p in head.parameters():
        p.requires_grad_(False)
    return model, head


def test_pitch_is_chosen_every_tick_and_attack_only_at_decisions(world):
    model, _ = world
    agent = TemporalBreakAgent(model.features, 32)
    s = torch.zeros(64, model.features)
    controller = AttackController(torch.tensor([0] * 32 + [7] * 32))
    x, choices, deciding = temporal_step(
        agent, s, controller, torch.Generator().manual_seed(0)
    )
    assert deciding.tolist() == [True] * 32 + [False] * 32
    assert (choices[32:, ATTACK_FACTOR] == 1).all()
    assert len(set(choices[:, PITCH_FACTOR].tolist())) > 1
    for k, noop in enumerate(ActionSpace.NOOP):
        if k not in (PITCH_FACTOR, ATTACK_FACTOR):
            assert (choices[:, k] == noop).all()
    assert controller.remaining[32:].tolist() == [6] * 32


def test_imagination_feeds_the_physical_attack_of_every_tick(world, monkeypatch):
    model, head = world
    agent = TemporalBreakAgent(model.features, 32)
    with torch.no_grad():
        agent.heads[ATTACK_FACTOR].bias.copy_(torch.tensor([-20.0, 20.0]))
    fed = []
    transition = model.transition

    def recording(h, z, action):
        fed.append(action.clone())
        return transition(h, z, action)

    monkeypatch.setattr(model, "transition", recording)
    horizon = 32
    imagined = imagine_temporal(
        model,
        head,
        agent,
        torch.zeros(3, model.hidden),
        torch.zeros(3, model.latent_size),
        horizon,
        torch.Generator().manual_seed(0),
    )
    assert len(fed) == horizon
    attack = imagined.choices[..., ATTACK_FACTOR]
    assert (attack == 1).all()
    for t, action in enumerate(fed):
        expected = ActionSpace.world_model_action(imagined.choices[:, t])
        assert torch.equal(action, expected)
    deciding = agent.deciding(imagined.features[:, :-1])
    assert deciding[0].nonzero().squeeze(1).tolist() == [0, 20]


def test_real_and_imagined_steps_share_one_controller_rule(world):
    model, head = world
    agent = TemporalBreakAgent(model.features, 32)
    start = torch.zeros(1, model.hidden), torch.zeros(1, model.latent_size)
    imagined = imagine_temporal(
        model, head, agent, *start, 30, torch.Generator().manual_seed(4)
    )
    controller = AttackController.start(1)
    generator = torch.Generator().manual_seed(4)
    h, z = start
    for t in range(30):
        s = torch.cat([h, z], -1)
        x, choice, _ = temporal_step(agent, s, controller, generator)
        assert torch.equal(x, imagined.features[:, t])
        assert torch.equal(choice, imagined.choices[:, t])
        h = model.transition(h, z, ActionSpace.world_model_action(choice))
        z = model.sample(model.prior_logits(h), generator, straight_through=False)
        head.reward(torch.cat([h, z], -1))
        torch.sigmoid(model.continue_head(torch.cat([h, z], -1)))


def test_rollouts_start_in_the_given_controller_state(world):
    model, head = world
    agent = TemporalBreakAgent(model.features, 32)
    imagined = imagine_temporal(
        model,
        head,
        agent,
        torch.zeros(2, model.hidden),
        torch.zeros(2, model.latent_size),
        5,
        torch.Generator().manual_seed(0),
        torch.tensor([3, 0]),
    )
    assert (imagined.choices[0, :3, ATTACK_FACTOR] == 1).all()
    assert agent.deciding(imagined.features[0, :5]).tolist() == [
        False,
        False,
        False,
        True,
        bool(imagined.choices[0, 3, ATTACK_FACTOR] == RELEASE_ATTACK),
    ]


def imagined_with(agent, features, choices, rewards):
    n, horizon = rewards.shape
    return Imagined(features, choices, rewards, torch.ones(n, horizon))


def decision_features(features: int, phases: list[int]) -> torch.Tensor:
    s = torch.zeros(1, len(phases) + 1, features)
    phase = torch.tensor(phases + [0]).float() / HOLD_TICKS
    return torch.cat([s, phase.view(1, -1, 1)], -1)


def test_only_decision_ticks_carry_command_log_probability_and_entropy(world):
    model, _ = world
    agent = TemporalBreakAgent(model.features, 32)
    x = decision_features(model.features, [0, 19, 18, 0])[:, :-1]
    choices = torch.as_tensor(ActionSpace.NOOP).expand(1, 4, -1).clone()
    choices[..., ATTACK_FACTOR] = torch.tensor([1, 1, 1, 0])
    _, entropy, per_factor = agent.log_prob_entropy(x, choices)
    assert per_factor[ATTACK_FACTOR][0, 1:3].tolist() == [0.0, 0.0]
    assert (per_factor[ATTACK_FACTOR][0, [0, 3]] > 0).all()
    assert (per_factor[PITCH_FACTOR][0] > 0).all()
    assert torch.allclose(entropy, per_factor[ATTACK_FACTOR] + per_factor[PITCH_FACTOR])
    log_prob, _, _ = agent.log_prob_entropy(x, choices)
    log_prob[0, 1].backward()
    assert agent.heads[ATTACK_FACTOR].weight.grad.abs().sum() == 0


def test_a_delayed_reward_reaches_the_hold_decision(world):
    model, _ = world
    agent = TemporalBreakAgent(model.features, 32)
    horizon = 20
    x = decision_features(model.features, [0] + list(range(19, 0, -1)))
    choices = torch.as_tensor(ActionSpace.NOOP).expand(1, horizon, -1).clone()
    choices[..., ATTACK_FACTOR] = HOLD_ATTACK_20
    rewards = torch.zeros(1, horizon)
    rewards[0, 15] = 1.0
    terms = losses(
        agent, imagined_with(agent, x, choices, rewards), ActorConfig(entropy=0.0)
    )
    terms["actor"].backward()
    grad = agent.heads[ATTACK_FACTOR].bias.grad
    assert grad[HOLD_ATTACK_20] < 0 < grad[RELEASE_ATTACK]


def test_returns_discount_every_tick_inside_a_hold():
    rewards = torch.zeros(1, 20)
    rewards[0, 15] = 1.0
    values = torch.zeros(1, 21)
    returns = lambda_returns(rewards, torch.ones(1, 20), values, 0.99, 1.0)
    assert returns[0, 0] == pytest.approx(0.99**15)
    assert returns[0, 15] == pytest.approx(1.0)


class FakeProbe:
    def __init__(self, client):
        self.client = client

    def player(self):
        return {"server": {"x": 0.0, "z": 0.0, "yaw": 0.0}}

    def registry(self):
        return {"block": list(REGISTRIES.block), "item": list(REGISTRIES.item)}


class FakeInfo:
    def __init__(self, tick):
        self.step_id = tick
        self.tick_before = tick
        self.tick_after = tick + 1
        self.client_tick = tick


class FakeEvents:
    def __init__(self, broken):
        self.blocks_broken = broken
        self.advancements = ()


class FakeResult:
    def __init__(self, observation, tick, terminated, broken):
        self.observation = observation
        self.terminated = terminated
        self.info = FakeInfo(tick)
        self.events = FakeEvents(broken)


class FakeReset:
    level_id = "x"
    game_time = 1000


class FakeClient:
    """A Minecraft stand-in: one tick per step, a death at `death_tick`,
    a break every 30 ticks. Counts resets."""

    def __init__(self, death_tick=None):
        self.tick = 1000
        self.death_tick = death_tick
        self.resets = 0
        self.observation = PolicyObservation.from_json(observation_json(), SCHEMA)

    def reset(self, seed, preset, fixed_spawn):
        self.resets += 1
        self.tick = 1000
        return self.observation, FakeReset()

    def observe(self):
        return self.observation

    def step(self, action):
        tick = self.tick
        self.tick += 1
        steps = self.tick - 1000
        dead = self.death_tick is not None and steps >= self.death_tick
        broken = (4,) if steps % 30 == 0 else ()
        return FakeResult(self.observation, tick, dead, broken)

    def privileged(self):
        return FakeProbe(self)


class CountingPolicy:
    """A temporal-controller policy stand-in that holds attack forever and
    counts its resets and steps."""

    def __init__(self):
        self.resets = 0
        self.steps = 0
        self.controller = AttackController.start(1)
        self.decision = None
        self.value = 0.0

    def reset(self, observation):
        self.resets += 1
        self.controller = AttackController.start(1)

    def act(self, observation):
        deciding = bool(self.controller.deciding[0])
        attack = bool(self.controller.advance(torch.tensor([HOLD_ATTACK_20]))[0])
        self.decision = HOLD_ATTACK_20 if deciding else None
        choice = list(ActionSpace.NOOP)
        choice[ATTACK_FACTOR] = int(attack)
        return ActionSpace.to_player_action(choice), choice

    def observe(self, choice, observation):
        self.steps += 1


def play(tmp_path, client, policy, **config):
    return play_life(
        client,
        policy,
        0,
        LifeConfig(**config),
        REGISTRIES,
        tmp_path,
        None,
        None,
        {"policy": "test", "policy_seed": 1},
    )


def test_a_life_continues_past_the_old_step_limit_and_checkpoints(tmp_path):
    client, policy = FakeClient(death_tick=1500), CountingPolicy()
    summary = play(tmp_path, client, policy, checkpoints=(200, 1000), segment_steps=400)
    assert summary["ticks"] == 1500 and summary["stop_reason"] == "death"
    assert summary["terminated"] and not summary["truncated"]
    assert client.resets == 1 and policy.resets == 1
    assert policy.steps == 1500
    assert summary["checkpoints"]["200"]["ticks"] == 200
    assert summary["checkpoints"]["1000"]["blocks_broken"] == 1000 // 30
    assert summary["final"]["attack_fraction"] == 1.0


def test_a_checkpoint_does_not_restart_the_controller(tmp_path):
    client, policy = FakeClient(death_tick=230), CountingPolicy()
    summary = play(tmp_path, client, policy, checkpoints=(10,), segment_steps=1000)
    assert summary["final"]["attack_runs"] == 1
    assert summary["final"]["decisions"] == 12


def test_death_terminates_and_the_next_life_resets(tmp_path):
    client = FakeClient(death_tick=50)
    policy = CountingPolicy()
    first = play(tmp_path, client, policy)
    policy.controller.remaining[:] = 5
    second = play_life(
        client,
        policy,
        1,
        LifeConfig(),
        REGISTRIES,
        tmp_path,
        None,
        None,
        {"policy": "test", "policy_seed": 2},
    )
    assert first["ticks"] == second["ticks"] == 50
    assert client.resets == 2 and policy.resets == 2
    lives = join_segments(load_episodes(tmp_path / "eval"))
    assert sorted(e.meta.episode for e in lives) == [0, 1]
    assert all(e.meta.terminated and e.arrays["terminated"][-1] for e in lives)


def test_an_external_stop_is_truncated_not_death(tmp_path):
    client = FakeClient(death_tick=None)
    summary = play(
        tmp_path, client, CountingPolicy(), tick_budget=120, segment_steps=60
    )
    assert [s["steps"] for s in summary["segments"]] == [60, 60]
    assert summary["truncated"] and not summary["terminated"]
    assert summary["stop_reason"] == "tick_budget" and not summary["complete_life"]
    (life,) = join_segments(load_episodes(tmp_path / "eval"))
    assert life.meta.truncated and not life.meta.terminated
    assert not life.arrays["terminated"].any()
    replay_continues = 1.0 - torch.as_tensor(life.arrays["terminated"]).float()
    assert (replay_continues == 1).all()


def test_segments_join_into_one_life_without_reset(tmp_path):
    client = FakeClient(death_tick=1000)
    summary = play(tmp_path, client, CountingPolicy(), segment_steps=300)
    assert [s["steps"] for s in summary["segments"]] == [300, 300, 300, 100]
    stored = load_episodes(tmp_path / "eval")
    assert len(stored) == 4
    assert [e.meta.terminated for e in stored] == [False, False, False, True]
    assert [e.meta.first_step for e in stored] == [0, 300, 600, 900]
    (life,) = join_segments(stored)
    assert life.steps == 1000 and life.arrays["obs_ray_kind"].shape[0] == 1001
    assert (numpy.diff(life.arrays["tick_before"]) == 1).all()
    assert life.arrays["privileged_breaks"].sum() == 1000 // 30


def test_replay_windows_never_cross_lives():
    runtime.configure(0)
    a = synthetic_episode(10, seed=1, episode=0)
    b = synthetic_episode(10, seed=2, episode=1)
    a.arrays["obs_pitch"] = numpy.arange(11, dtype=numpy.float32)
    b.arrays["obs_pitch"] = 100 + numpy.arange(11, dtype=numpy.float32)
    compact = CompactVocabulary.from_episodes([a, b], RAW_SIZES)
    replay = SequenceReplay([a, b], compact, 6)
    batch = replay.sample(200, torch.Generator().manual_seed(0))
    pitch = batch.observations["pitch"]
    assert ((pitch < 100).all(1) | (pitch >= 100).all(1)).all()
    assert (torch.diff(pitch, dim=1) == 1).all()


class FakeRecorder:
    """Counts video segments; a stand-in for `WorkerRecorder`."""

    def __init__(self):
        self.run = None
        self.episode = 0
        self.started = 0
        self.stopped = 0

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

        class Encoded:
            status = "complete"
            video_bytes = 10
            directory = "x"

        return {}, Encoded()


def test_video_segments_split_without_resetting_minecraft(tmp_path):
    client, policy, recorder = (
        FakeClient(death_tick=700),
        CountingPolicy(),
        FakeRecorder(),
    )
    summary = play_life(
        client,
        policy,
        0,
        LifeConfig(segment_steps=250),
        REGISTRIES,
        tmp_path,
        recorder,
        None,
        {"policy": "test", "policy_seed": 1},
    )
    assert recorder.started == recorder.stopped == 3
    assert [v["segment"] for v in summary["videos"]] == [0, 1, 2]
    assert client.resets == 1 and policy.resets == 1 and policy.steps == 700


def test_a_step_that_skips_a_tick_stops_the_life(tmp_path):
    client = FakeClient(death_tick=None)
    step = client.step

    def skipping(action):
        result = step(action)
        if client.tick - 1000 == 5:
            result.info.tick_after += 1
        return result

    client.step = skipping
    with pytest.raises(RuntimeError, match="exactly one tick"):
        play(tmp_path, client, CountingPolicy(), tick_budget=50)
