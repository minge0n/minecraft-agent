"""Stage 2F integrated Dreamer-style toy agent on the T-maze.

See docs/stage2f.md. One loop alternates three phases: train the world model on
all real episodes collected so far, train actor and critic in the world model's
imagination, then collect new real episodes with the current policy and add them
to the replay data. Before the new episodes are trained on, the world model's
one-step and open-loop error on them is measured: that is the model error under
the policy's own behavior.
"""

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from minecraft_rl import actor_critic, resumable, runtime, world_model
from minecraft_rl.devices import DEVICES, select_device
from minecraft_rl.imagination import rollout_errors
from minecraft_rl.provenance import git_commit
from minecraft_rl.tmaze import Action, Cue, Observation, TMaze
from minecraft_rl.world_model import DynamicsModel, Episodes

COLLECTION_SEED_OFFSET = 5_000_000
LATENT_SEED_OFFSET = 6_000_000
EXPLORATION_ENTROPY = 0.03


@dataclass(frozen=True)
class Config:
    agent: actor_critic.Config = field(
        default_factory=lambda: actor_critic.Config(entropy=EXPLORATION_ENTROPY)
    )
    iterations: int = 12
    initial_episodes: int = 64
    episodes_per_iteration: int = 32
    world_model_updates: int = 1000
    actor_critic_updates: int = 200
    horizons: tuple[int, ...] = (1, 5, 10, 20)


def collect_with_agent(
    model: DynamicsModel,
    agent: actor_critic.Agent,
    config: actor_critic.Config,
    count: int,
    generator: torch.Generator,
) -> Episodes:
    """Run `count` real episodes with random cues, sampling actions from the
    actor on the world model's policy state built from the real observations."""
    model_config = config.model
    device = next(model.parameters()).device
    shape = (count, model_config.max_steps)
    observations = torch.zeros(shape, dtype=torch.long)
    actions = torch.zeros(shape, dtype=torch.long)
    next_observations = torch.zeros(shape, dtype=torch.long)
    rewards = torch.zeros(shape)
    continues = torch.zeros(shape)
    mask = torch.zeros(shape, dtype=torch.bool)
    cues = torch.randint(0, len(Cue), (count,), generator=generator)
    environments = [
        TMaze(model_config.corridor_length, model_config.max_steps)
        for _ in range(count)
    ]
    current = torch.tensor(
        [env.reset(Cue(int(c))) for env, c in zip(environments, cues, strict=True)]
    )
    with torch.no_grad():
        states = model.initial_policy_states(current.to(device), generator)
    running = torch.ones(count, dtype=torch.bool)
    for t in range(model_config.max_steps):
        with torch.no_grad():
            chosen = agent.act(states, generator, greedy=False).cpu()
        following = current.clone()
        for i in torch.nonzero(running).squeeze(-1).tolist():
            observation, reward, terminated, truncated = environments[i].step(
                Action(int(chosen[i]))
            )
            observations[i, t] = current[i]
            actions[i, t] = chosen[i]
            next_observations[i, t] = observation
            rewards[i, t] = reward
            continues[i, t] = 0.0 if terminated else 1.0
            mask[i, t] = True
            following[i] = observation
            if terminated or truncated:
                running[i] = False
        current = following
        if not running.any():
            break
        with torch.no_grad():
            states = model.observe_step(
                states, chosen.to(device), current.to(device), generator
            )
    return Episodes(
        observations, actions, next_observations, rewards, continues, mask, cues
    )


def concatenate(first: Episodes, second: Episodes) -> Episodes:
    return Episodes(
        *(
            torch.cat([getattr(first, f), getattr(second, f)])
            for f in Episodes.__dataclass_fields__
        )
    )


def run(
    config: Config,
    device: torch.device,
    output: Path,
    session: resumable.Session | None = None,
) -> dict[str, Any]:
    """The integrated loop. With a `session`, the run can stop after any
    iteration and continue in a later process with identical results."""
    session = session or resumable.Session(None, None, {})
    agent_config = config.agent
    model_config = agent_config.model
    seed = model_config.seed
    seeds = {
        "torch_initialization": seed,
        "initial_random_data": seed,
        "world_model_minibatches": seed,
        "imagination_sampling": seed,
        "policy_data_collection": seed + COLLECTION_SEED_OFFSET,
        "latent_sampling": seed + LATENT_SEED_OFFSET,
        "real_evaluation": seed + actor_critic.EVALUATION_SEED_OFFSET,
    }
    saved = session.load()
    model, model_optimizer = world_model.build(model_config, device)
    agent, agent_optimizer = actor_critic.build(agent_config, device)
    if saved is None:
        replay = world_model.collect_episodes(
            model_config, config.initial_episodes, torch.Generator().manual_seed(seed)
        )
        model_generator = torch.Generator().manual_seed(seed)
        agent_generator = torch.Generator().manual_seed(seed)
        latent_generator = torch.Generator().manual_seed(seed + LATENT_SEED_OFFSET)
        collection_generator = torch.Generator().manual_seed(
            seeds["policy_data_collection"]
        )
        iterations: list[dict[str, Any]] = []
    else:
        replay = Episodes(**saved["replay"])
        model.load_state_dict(saved["model"])
        model_optimizer.load_state_dict(saved["model_optimizer"])
        agent.load_state_dict(saved["agent"])
        agent_optimizer.load_state_dict(saved["agent_optimizer"])
        model_generator = resumable.restore_generator(saved["model_generator"])
        agent_generator = resumable.restore_generator(saved["agent_generator"])
        latent_generator = resumable.restore_generator(saved["latent_generator"])
        collection_generator = resumable.restore_generator(
            saved["collection_generator"]
        )
        iterations = saved["iterations"]
        session.restore_global_random_state()

    def state() -> dict[str, Any]:
        return {
            "replay": resumable.tensors_of(replay),
            "model": model.state_dict(),
            "model_optimizer": model_optimizer.state_dict(),
            "agent": agent.state_dict(),
            "agent_optimizer": agent_optimizer.state_dict(),
            "model_generator": model_generator.get_state(),
            "agent_generator": agent_generator.get_state(),
            "latent_generator": latent_generator.get_state(),
            "collection_generator": collection_generator.get_state(),
            "iterations": iterations,
        }

    for iteration in range(len(iterations), config.iterations):
        model_terms: list[dict[str, float]] = []
        model_losses = world_model.train(
            model,
            model_optimizer,
            replay,
            model_config,
            device,
            model_generator,
            config.world_model_updates,
            model_terms,
        )
        starts = actor_critic.start_states(model, replay.to(device), latent_generator)
        agent_history = actor_critic.train(
            model,
            agent,
            agent_optimizer,
            starts,
            agent_config,
            agent_generator,
            config.actor_critic_updates,
        )
        evaluation = actor_critic.evaluate(
            model, agent, agent_config, seeds["real_evaluation"]
        )

        collected = collect_with_agent(
            model,
            agent,
            agent_config,
            config.episodes_per_iteration,
            collection_generator,
        )
        on_policy = collected.to(device)
        model_error_on_new_data = {
            "one_step": world_model.evaluate(
                model, on_policy, device, latent_generator
            ),
            "imagined": rollout_errors(
                model, on_policy, config.horizons, latent_generator
            ),
            "latent": model.diagnostics(on_policy, latent_generator),
        }
        replay = concatenate(replay, collected)

        iterations.append(
            {
                "iteration": iteration,
                "world_model_final_loss": model_losses[-1],
                "world_model_training_terms": world_model.mean_terms(model_terms),
                "actor_critic_final": agent_history[-1],
                "evaluation": evaluation,
                "collected": world_model.dataset_summary(collected),
                "model_error_on_new_policy_data": model_error_on_new_data,
                "replay_episodes": int(replay.mask.shape[0]),
                "real_transitions_total": int(replay.mask.sum()),
            }
        )
        if iteration + 1 < config.iterations:
            session.boundary(state)

    world_model.save_checkpoint(
        output.parent / "world_model.pt",
        model_config,
        config.iterations * config.world_model_updates,
        model,
        model_optimizer,
    )
    actor_critic.save_checkpoint(
        output.parent / "agent.pt",
        agent_config,
        config.iterations * config.actor_critic_updates,
        agent,
        agent_optimizer,
    )
    result: dict[str, Any] = {
        "stage": "2f-integrated-dreamer-style-tmaze",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "config": asdict(config),
        "seeds": seeds,
        "device": str(device),
        "runtime": runtime.metadata(device),
        "environment": {
            "name": "TMaze",
            "observations": {o.name: int(o) for o in Observation},
            "actions": {a.name: int(a) for a in Action},
            "corridor_length": model_config.corridor_length,
            "max_steps": model_config.max_steps,
        },
        "parameters": {
            "world_model": world_model.parameter_count(model),
            "agent": world_model.parameter_count(agent),
        },
        "iterations": iterations,
        "final": iterations[-1],
        "checkpoints": {
            "world_model": str(output.parent / "world_model.pt"),
            "agent": str(output.parent / "agent.pt"),
        },
        "duration_seconds": session.elapsed_seconds,
        "sessions": session.sessions,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    session.finish()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=world_model.Config.seed)
    parser.add_argument("--iterations", type=int, default=Config.iterations)
    parser.add_argument("--entropy", type=float, default=EXPLORATION_ENTROPY)
    world_model.add_world_model_arguments(parser)
    resumable.add_arguments(parser)
    parser.add_argument("--device", default="cpu", choices=DEVICES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runtime.configure(args.seed)
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    config = Config(
        agent=actor_critic.Config(
            model=world_model.Config(
                seed=args.seed, **world_model.world_model_options(args)
            ),
            entropy=args.entropy,
        ),
        iterations=args.iterations,
    )
    output = (
        args.output
        or Path("runs")
        / datetime.now(UTC).strftime("stage2f-%Y%m%dT%H%M%S%fZ")
        / "metrics.json"
    )
    session = resumable.session_from(
        args, {"experiment": "dreamer_loop", "config": asdict(config)}
    )
    try:
        result = run(config, select_device(args.device), output, session)
    except resumable.Incomplete as stopped:
        print(stopped)
        raise SystemExit(resumable.INCOMPLETE_EXIT_CODE) from None

    runtime_info = result["runtime"]
    print(
        f"Device {result['device']}, torch {runtime_info['versions']['torch']}, "
        f"threads {runtime_info['intra_op_threads']}+{runtime_info['inter_op_threads']}"
    )
    print(
        f"World model {config.agent.model.kind} "
        f"{result['parameters']['world_model']} parameters, "
        f"agent {result['parameters']['agent']}"
    )
    longest = str(max(config.horizons))
    print(
        "iter  real steps  greedy success (left/right)  sampled  "
        "imagined-real return | on new policy data: turn sign acc, "
        f"k={longest} return err"
    )
    for at in result["iterations"]:
        greedy = at["evaluation"]["greedy"]
        sampled = at["evaluation"]["sampled"]
        error = at["model_error_on_new_policy_data"]
        imagined = error["imagined"][longest]
        turn = error["one_step"]["junction_turn"]
        print(
            f"  {at['iteration']:>2}  {at['real_transitions_total']:>9}  "
            f"{greedy['success_rate']:>6.0%} "
            f"({greedy['success_rate_by_cue']['left']:.0%}/"
            f"{greedy['success_rate_by_cue']['right']:.0%})"
            f"          {sampled['success_rate']:>5.0%}   "
            f"{greedy['imagined_minus_real_discounted_return']:+.3f}"
            f"            | {turn['reward_sign_accuracy']:.0%}"
            f"  {imagined['return_absolute_error']:.3f}"
        )
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
