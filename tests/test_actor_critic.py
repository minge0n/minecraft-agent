import pytest
import torch

from minecraft_rl import actor_critic, world_model
from minecraft_rl.tmaze import Observation

CPU = torch.device("cpu")
FAST = actor_critic.Config(
    model=world_model.Config(
        seed=0,
        corridor_length=1,
        max_steps=12,
        training_episodes=256,
        evaluation_episodes=128,
        learning_rate=0.01,
        steps=600,
    ),
    horizon=8,
    steps=150,
    start_states=64,
    evaluation_episodes=64,
    learning_rate=0.003,
)


@pytest.fixture(scope="module")
def frozen():
    model, episodes, _ = actor_critic.train_world_model(FAST, CPU)
    return model, actor_critic.start_states(model, episodes)


def test_lambda_returns_match_hand_computation():
    rewards = torch.tensor([[1.0, 0.0, 2.0]])
    continues = torch.tensor([[1.0, 1.0, 1.0]])
    values = torch.tensor([[0.5, 0.25, 1.0, 4.0]])
    monte_carlo = actor_critic.lambda_returns(rewards, continues, values, 0.5, 1.0)
    assert torch.allclose(
        monte_carlo[0, 0], torch.tensor(1 + 0.5 * 0 + 0.25 * 2 + 0.125 * 4)
    )
    one_step = actor_critic.lambda_returns(rewards, continues, values, 0.5, 0.0)
    assert torch.allclose(one_step, torch.tensor([[1.125, 0.5, 4.0]]))


def test_lambda_returns_stop_at_a_terminal_transition():
    rewards = torch.tensor([[1.0, 5.0]])
    continues = torch.tensor([[0.0, 1.0]])
    values = torch.tensor([[0.0, 7.0, 9.0]])
    returns = actor_critic.lambda_returns(rewards, continues, values, 0.9, 0.95)
    assert returns[0, 0] == 1.0


def test_no_memory_agent_ignores_the_recurrent_state():
    agent, _ = actor_critic.build(FAST, CPU, memory=False)
    observations = torch.tensor([Observation.JUNCTION] * 2)
    states = torch.stack([torch.zeros(32), torch.ones(32)])
    with torch.no_grad():
        logits = agent.actor(agent.features(states, observations))
    assert torch.equal(logits[0], logits[1])


def test_imagined_rollout_shapes(frozen):
    model, starts = frozen
    agent, _ = actor_critic.build(FAST, CPU)
    batch = actor_critic.StartStates(starts.previous[:5], starts.observations[:5])
    imagined = actor_critic.imagine(
        model, agent, batch, 7, torch.Generator().manual_seed(0)
    )
    assert imagined.features.shape == (5, 8, 32 + len(Observation))
    assert imagined.actions.shape == imagined.rewards.shape == (5, 7)
    assert torch.all((imagined.continues >= 0) & (imagined.continues <= 1))


def test_start_states_cover_every_valid_step(frozen):
    model, starts = frozen
    episodes = world_model.collect_episodes(
        FAST.model, 4, torch.Generator().manual_seed(3)
    )
    states = actor_critic.start_states(model, episodes)
    assert states.observations.shape[0] == int(episodes.mask.sum())
    first = episodes.mask[:, 0].sum()
    assert torch.all(states.previous[: int(first)][0] == 0)


def test_actor_critic_learns_in_imagination_and_beats_random(frozen):
    model, starts = frozen
    parameters_before = [p.clone() for p in model.parameters()]
    success = {}
    for memory in (True, False):
        agent, optimizer = actor_critic.build(FAST, CPU, memory)
        history = actor_critic.train(
            model,
            agent,
            optimizer,
            starts,
            FAST,
            torch.Generator().manual_seed(0),
            FAST.steps,
        )
        success[memory] = actor_critic.run_in_environment(
            model, agent, FAST, 64, torch.Generator().manual_seed(1), greedy=True
        )["success_rate"]
        if memory:
            assert history[-1]["imagined_return"] > history[0]["imagined_return"]
    random_policy = actor_critic.run_in_environment(
        model, None, FAST, 64, torch.Generator().manual_seed(1), greedy=False
    )
    assert success[True] == 1.0
    assert success[True] > random_policy["success_rate"] + 0.3
    assert success[False] <= 0.5
    for before, after in zip(parameters_before, model.parameters(), strict=True):
        assert torch.equal(before, after)


def test_checkpoint_resume_matches_uninterrupted_training(frozen, tmp_path):
    model, starts = frozen
    uninterrupted, optimizer = actor_critic.build(FAST, CPU)
    generator = torch.Generator().manual_seed(0)
    actor_critic.train(model, uninterrupted, optimizer, starts, FAST, generator, 10)

    generator = torch.Generator().manual_seed(0)
    resumed, optimizer = actor_critic.build(FAST, CPU)
    actor_critic.train(model, resumed, optimizer, starts, FAST, generator, 5)
    path = tmp_path / "agent.pt"
    actor_critic.save_checkpoint(path, FAST, 5, resumed, optimizer)
    config, step, restored, restored_optimizer = actor_critic.load_checkpoint(path, CPU)
    actor_critic.train(model, restored, restored_optimizer, starts, FAST, generator, 5)

    assert config == FAST
    assert step == 5
    for a, b in zip(uninterrupted.parameters(), restored.parameters(), strict=True):
        assert torch.equal(a, b)


def test_load_rejects_a_foreign_checkpoint(tmp_path):
    path = tmp_path / "other.pt"
    torch.save({"format": world_model.CHECKPOINT_FORMAT}, path)
    with pytest.raises(ValueError, match=actor_critic.CHECKPOINT_FORMAT):
        actor_critic.load_checkpoint(path, CPU)
