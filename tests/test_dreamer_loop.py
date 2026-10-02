import json

import torch

from minecraft_rl import actor_critic, dreamer_loop, world_model
from minecraft_rl.tmaze import Observation

CPU = torch.device("cpu")
FAST = dreamer_loop.Config(
    agent=actor_critic.Config(
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
        start_states=64,
        evaluation_episodes=32,
        learning_rate=0.003,
        entropy=0.03,
    ),
    iterations=4,
    initial_episodes=32,
    episodes_per_iteration=16,
    world_model_updates=300,
    actor_critic_updates=150,
)


def test_policy_collected_episodes_are_consistent_transitions():
    model, _ = world_model.build(FAST.agent.model, CPU)
    agent, _ = actor_critic.build(FAST.agent, CPU)
    episodes = dreamer_loop.collect_with_agent(
        model, agent, FAST.agent, 16, torch.Generator().manual_seed(0)
    )
    lengths = episodes.mask.sum(1)
    for episode in range(16):
        length = int(lengths[episode])
        assert length >= 1
        assert not episodes.mask[episode, length:].any()
        assert episodes.observations[episode, 0] == Observation(
            int(episodes.cues[episode])
        )
        assert torch.equal(
            episodes.observations[episode, 1:length],
            episodes.next_observations[episode, : length - 1],
        )
        terminal = episodes.continues[episode, :length] == 0
        assert terminal.sum() <= 1
        if terminal.any():
            assert terminal[-1]


def test_integrated_loop_learns_from_its_own_experience(tmp_path):
    output = tmp_path / "metrics.json"
    result = dreamer_loop.run(FAST, CPU, output)
    iterations = result["iterations"]

    assert [it["replay_episodes"] for it in iterations] == [48, 64, 80, 96]
    assert all(
        it["model_error_on_new_policy_data"]["imagined"]["1"]["pairs"] > 0
        for it in iterations
    )
    assert iterations[-1]["evaluation"]["greedy"]["success_rate"] == 1.0
    assert iterations[-1]["evaluation"]["sampled"]["success_rate"] > 0.8
    assert (
        abs(
            iterations[-1]["evaluation"]["greedy"][
                "imagined_minus_real_discounted_return"
            ]
        )
        < 0.2
    )
    assert json.loads(output.read_text())["stage"] == result["stage"]

    _, _, model, _ = world_model.load_checkpoint(tmp_path / "world_model.pt", CPU)
    _, _, agent, _ = actor_critic.load_checkpoint(tmp_path / "agent.pt", CPU)
    reloaded = actor_critic.run_in_environment(
        model,
        agent,
        FAST.agent,
        32,
        torch.Generator().manual_seed(FAST.agent.model.seed),
        greedy=True,
    )
    assert reloaded["success_rate"] == 1.0
