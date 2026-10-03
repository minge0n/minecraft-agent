"""Stage 2E actor-critic trained in imagination on the T-maze.

See docs/stage2e.md. A Stage 2C world model is trained on random-policy episodes
and frozen. An actor and a critic are then trained only on trajectories the world
model imagines from real start states, with lambda-returns, a REINFORCE actor
gradient and an entropy bonus. They are evaluated only in the real T-maze, next to
a uniform random policy and an agent whose actor and critic see the current
observation without the recurrent state.
"""

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

from minecraft_rl import resumable, runtime, world_model
from minecraft_rl.devices import DEVICES, select_device
from minecraft_rl.provenance import git_commit
from minecraft_rl.tmaze import Action, Cue, Observation, TMaze
from minecraft_rl.world_model import ACTIONS, OBSERVATIONS, DynamicsModel

CHECKPOINT_FORMAT = "tmaze-actor-critic-v1"
EVALUATION_SEED_OFFSET = 4_000_000
LATENT_SEED_OFFSET = 6_000_000


@dataclass(frozen=True)
class Config:
    model: world_model.Config = field(default_factory=world_model.Config)
    hidden: int = 64
    horizon: int = 15
    discount: float = 0.95
    return_lambda: float = 0.95
    entropy: float = 0.01
    learning_rate: float = 0.001
    steps: int = 1000
    start_states: int = 256
    evaluation_episodes: int = 256
    report_steps: tuple[int, ...] = (0, 50, 100, 200, 400, 700, 1000)


@dataclass(frozen=True)
class Imagined:
    """A rollout of horizon T from n start states: features of s_0 .. s_T
    (n, T + 1, F), actions a_0 .. a_{T-1} (n, T), predicted rewards and
    continuation probabilities of each transition (n, T)."""

    features: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    continues: torch.Tensor


class Agent(nn.Module):
    """Actor pi(a | s) and critic v(s) on the world model's policy state s_t.

    With `memory` the agent sees the whole policy state: [h_{t-1}, onehot(o_t)]
    for the GRU world model, [h_t, z_t] for the RSSM. Without it, only
    onehot(o_t), the last entries of the GRU policy state (the no-memory control).
    """

    def __init__(self, state_size: int, hidden: int, memory: bool) -> None:
        super().__init__()
        self.memory = memory
        inputs = state_size if memory else OBSERVATIONS
        self.actor = nn.Sequential(
            nn.Linear(inputs, hidden), nn.Tanh(), nn.Linear(hidden, ACTIONS)
        )
        self.critic = nn.Sequential(
            nn.Linear(inputs, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def features(self, states: torch.Tensor) -> torch.Tensor:
        return states if self.memory else states[..., -OBSERVATIONS:]

    def act(
        self,
        states: torch.Tensor,
        generator: torch.Generator | None,
        greedy: bool,
    ) -> torch.Tensor:
        logits = self.actor(self.features(states))
        if greedy:
            return logits.argmax(-1)
        probabilities = torch.softmax(logits, -1).cpu()
        action = torch.multinomial(probabilities, 1, generator=generator)
        return action.squeeze(-1).to(logits.device)


def build(
    config: Config, device: torch.device, memory: bool = True
) -> tuple[Agent, torch.optim.Adam]:
    if not memory and config.model.kind != "gru":
        raise ValueError("the no-memory agent control needs the GRU world model")
    torch.manual_seed(config.model.seed)
    state_size = world_model.policy_state_size(config.model)
    agent = Agent(state_size, config.hidden, memory).to(device)
    return agent, torch.optim.Adam(agent.parameters(), lr=config.learning_rate)


def start_states(
    model: DynamicsModel,
    episodes: world_model.Episodes,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """The policy state of every valid real step of the episodes (n, F)."""
    with torch.no_grad():
        states = model.policy_states(episodes, generator)
    return states[episodes.mask]


def imagine(
    model: DynamicsModel,
    agent: Agent,
    starts: torch.Tensor,
    horizon: int,
    generator: torch.Generator | None,
    greedy: bool = False,
) -> Imagined:
    """Roll the frozen world model forward with actions from the actor.

    Each step samples a_t from pi(. | s_t) (or takes its argmax) and lets the
    world model produce s_{t+1}, the reward and the continuation probability:
    the GRU feeds back its most likely predicted observation, the RSSM samples
    its prior latent. Everything is computed without gradients; the losses
    re-evaluate actor and critic on the returned features.
    """
    state = starts
    features, actions, rewards, continues = [], [], [], []
    with torch.no_grad():
        for _ in range(horizon):
            action = agent.act(state, generator, greedy)
            following, reward, continuation = model.imagine_step(
                state, action, generator
            )
            features.append(agent.features(state))
            actions.append(action)
            rewards.append(reward)
            continues.append(continuation)
            state = following
        features.append(agent.features(state))
    return Imagined(
        torch.stack(features, 1),
        torch.stack(actions, 1),
        torch.stack(rewards, 1),
        torch.stack(continues, 1),
    )


def lambda_returns(
    rewards: torch.Tensor,
    continues: torch.Tensor,
    values: torch.Tensor,
    discount: float,
    return_lambda: float,
) -> torch.Tensor:
    """R_t = r_t + gamma c_t ((1 - lambda) v_{t+1} + lambda R_{t+1}), R_T = v_T.

    rewards and continues (n, T), values (n, T + 1); returns (n, T).
    """
    following = values[:, -1]
    returns = []
    for t in reversed(range(rewards.shape[1])):
        following = rewards[:, t] + discount * continues[:, t] * (
            (1 - return_lambda) * values[:, t + 1] + return_lambda * following
        )
        returns.append(following)
    return torch.stack(returns[::-1], 1)


def losses(agent: Agent, imagined: Imagined, config: Config) -> dict[str, torch.Tensor]:
    """Critic regression to lambda-returns and REINFORCE actor loss with an entropy
    bonus, both weighted by the imagined probability of still being in the
    episode."""
    values = agent.critic(imagined.features).squeeze(-1)
    with torch.no_grad():
        returns = lambda_returns(
            imagined.rewards,
            imagined.continues,
            values,
            config.discount,
            config.return_lambda,
        )
        alive = torch.cat(
            [
                torch.ones_like(imagined.continues[:, :1]),
                imagined.continues.cumprod(1)[:, :-1],
            ],
            dim=1,
        )
        advantages = returns - values[:, :-1]
    critic = (alive * 0.5 * (values[:, :-1] - returns) ** 2).mean()
    log_probabilities = torch.log_softmax(agent.actor(imagined.features[:, :-1]), -1)
    chosen = log_probabilities.gather(-1, imagined.actions.unsqueeze(-1)).squeeze(-1)
    entropy = -(log_probabilities.exp() * log_probabilities).sum(-1)
    actor = -(alive * (chosen * advantages + config.entropy * entropy)).mean()
    return {
        "actor": actor,
        "critic": critic,
        "entropy": (alive * entropy).sum() / alive.sum(),
        "imagined_return": returns[:, 0].mean(),
    }


def train(
    model: DynamicsModel,
    agent: Agent,
    optimizer: torch.optim.Optimizer,
    starts: torch.Tensor,
    config: Config,
    generator: torch.Generator,
    steps: int,
) -> list[dict[str, float]]:
    history = []
    count = starts.shape[0]
    for _ in range(steps):
        index = torch.randint(0, count, (config.start_states,), generator=generator)
        batch = starts[index.to(starts.device)]
        imagined = imagine(model, agent, batch, config.horizon, generator)
        terms = losses(agent, imagined, config)
        optimizer.zero_grad()
        (terms["actor"] + terms["critic"]).backward()
        optimizer.step()
        history.append({name: value.item() for name, value in terms.items()})
    return history


def run_in_environment(
    model: DynamicsModel,
    agent: Agent | None,
    config: Config,
    episodes: int,
    generator: torch.Generator,
    greedy: bool,
) -> dict[str, Any]:
    """Run episodes in the real T-maze, cues alternating left/right. The agent
    acts on the world model's policy state built from real observations; `None`
    is the uniform random policy."""
    device = next(model.parameters()).device
    environments = [
        TMaze(config.model.corridor_length, config.model.max_steps)
        for _ in range(episodes)
    ]
    cues = [Cue(i % len(Cue)) for i in range(episodes)]
    observations = torch.tensor(
        [env.reset(cue) for env, cue in zip(environments, cues, strict=True)],
        device=device,
    )
    with torch.no_grad():
        states = model.initial_policy_states(observations, generator)
    returns = torch.zeros(episodes)
    discounted = torch.zeros(episodes)
    lengths = torch.zeros(episodes, dtype=torch.long)
    outcome = torch.zeros(episodes)
    running = torch.ones(episodes, dtype=torch.bool)
    for t in range(config.model.max_steps):
        with torch.no_grad():
            if agent is None:
                actions = torch.randint(0, ACTIONS, (episodes,), generator=generator)
            else:
                actions = agent.act(states, generator, greedy).cpu()
        next_observations = observations.clone()
        for i in torch.nonzero(running).squeeze(-1).tolist():
            observation, reward, terminated, truncated = environments[i].step(
                Action(int(actions[i]))
            )
            next_observations[i] = observation
            returns[i] += reward
            discounted[i] += config.discount**t * reward
            lengths[i] += 1
            if terminated:
                outcome[i] = reward
            if terminated or truncated:
                running[i] = False
        observations = next_observations
        if not running.any():
            break
        with torch.no_grad():
            states = model.observe_step(
                states, actions.to(device), observations, generator
            )
    cue_right = torch.tensor([cue == Cue.RIGHT for cue in cues])
    return {
        "episodes": episodes,
        "success_rate": (outcome > 0).float().mean().item(),
        "wrong_turn_rate": (outcome < 0).float().mean().item(),
        "truncated_rate": (outcome == 0).float().mean().item(),
        "mean_return": returns.mean().item(),
        "return_standard_deviation": returns.std(unbiased=False).item(),
        "mean_discounted_return": discounted.mean().item(),
        "mean_length": lengths.float().mean().item(),
        "success_rate_by_cue": {
            "left": (outcome[~cue_right] > 0).float().mean().item(),
            "right": (outcome[cue_right] > 0).float().mean().item(),
        },
        "discounted_return_by_cue": {
            "left": discounted[~cue_right].mean().item(),
            "right": discounted[cue_right].mean().item(),
        },
    }


def imagined_from_episode_start(
    model: DynamicsModel, agent: Agent, config: Config, greedy: bool
) -> dict[str, float]:
    """Discounted return the world model imagines for the agent from each cue's
    real start state, over the episode step limit, for comparison with the real
    discounted return from the same start. A stochastic world model or a sampling
    policy is averaged over 128 rollouts."""
    device = next(model.parameters()).device
    cues = torch.tensor([Observation.CUE_LEFT, Observation.CUE_RIGHT], device=device)
    generator = torch.Generator().manual_seed(config.model.seed)
    with torch.no_grad():
        starts = model.initial_policy_states(cues, generator)
    rollouts = [
        imagine(model, agent, starts, config.model.max_steps, generator, greedy)
        for _ in range(1 if greedy and model.deterministic else 128)
    ]
    totals = []
    for rollout in rollouts:
        alive = torch.cat(
            [
                torch.ones_like(rollout.continues[:, :1]),
                rollout.continues.cumprod(1)[:, :-1],
            ],
            dim=1,
        )
        discounts = config.discount ** torch.arange(
            rollout.rewards.shape[1], device=device
        )
        totals.append((alive * discounts * rollout.rewards).sum(1))
    per_cue = torch.stack(totals).mean(0).cpu()
    return {"left": per_cue[0].item(), "right": per_cue[1].item()}


def evaluate(
    model: DynamicsModel, agent: Agent, config: Config, seed: int
) -> dict[str, Any]:
    """Real-environment results with greedy and sampled actions, each next to the
    discounted return the world model imagines for the same policy from the same
    cue start states.

    A stochastic world model is also evaluated with `mode_latents` set: every
    latent takes its most likely class instead of a sample, in the real
    environment and in imagination. This diagnostic separates latent sampling
    noise from a weak policy. Training never uses it."""
    result = evaluate_actions(model, agent, config, seed)
    if not model.deterministic:
        model.mode_latents = True
        try:
            result["latent_mode"] = evaluate_actions(model, agent, config, seed)
        finally:
            model.mode_latents = False
    return result


def evaluate_actions(
    model: DynamicsModel, agent: Agent, config: Config, seed: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode, greedy in (("greedy", True), ("sampled", False)):
        real = run_in_environment(
            model,
            agent,
            config,
            config.evaluation_episodes,
            torch.Generator().manual_seed(seed),
            greedy,
        )
        imagined = imagined_from_episode_start(model, agent, config, greedy)
        gap = [
            imagined[cue] - real["discounted_return_by_cue"][cue] for cue in imagined
        ]
        result[mode] = real | {
            "imagined_discounted_return_by_cue": imagined,
            "imagined_minus_real_discounted_return": sum(gap) / len(gap),
        }
    return result


def save_checkpoint(
    path: Path,
    config: Config,
    step: int,
    agent: Agent,
    optimizer: torch.optim.Optimizer,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "config": asdict(config),
            "memory": agent.memory,
            "step": step,
            "agent": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def config_from_dict(raw: dict[str, Any]) -> Config:
    model = raw["model"]
    return Config(
        **raw
        | {
            "model": world_model.Config(
                **model | {"report_steps": tuple(model["report_steps"])}
            ),
            "report_steps": tuple(raw["report_steps"]),
        }
    )


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[Config, int, Agent, torch.optim.Adam]:
    data = torch.load(path, map_location=device, weights_only=True)
    if data.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    config = config_from_dict(data["config"])
    agent, optimizer = build(config, device, data["memory"])
    agent.load_state_dict(data["agent"])
    optimizer.load_state_dict(data["optimizer"])
    return config, data["step"], agent, optimizer


def train_world_model(
    config: Config, device: torch.device
) -> tuple[DynamicsModel, world_model.Episodes, dict[str, Any]]:
    """The world model `config.model.kind` names, trained on uniform-random-policy
    data, then frozen."""
    model_config = config.model
    episodes = world_model.collect_episodes(
        model_config,
        model_config.training_episodes,
        torch.Generator().manual_seed(model_config.seed),
    )
    model, optimizer = world_model.build(model_config, device)
    world_model.train(
        model,
        optimizer,
        episodes,
        model_config,
        device,
        torch.Generator().manual_seed(model_config.seed),
        model_config.steps,
    )
    model.requires_grad_(False)
    model.eval()
    evaluation = world_model.collect_episodes(
        model_config,
        model_config.evaluation_episodes,
        torch.Generator().manual_seed(
            model_config.seed + world_model.EVALUATION_DATA_SEED_OFFSET
        ),
    )
    latent = torch.Generator().manual_seed(model_config.seed + LATENT_SEED_OFFSET)
    one_step = world_model.evaluate(model, evaluation, device, latent)
    return (
        model,
        episodes,
        one_step | {"latent": model.diagnostics(evaluation.to(device), latent)},
    )


def train_with_curve(
    model: DynamicsModel,
    starts: torch.Tensor,
    memory: bool,
    config: Config,
    device: torch.device,
    evaluation_seed: int,
    session: resumable.Session | None = None,
    saved: dict[str, Any] | None = None,
    state: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> tuple[Agent, torch.optim.Adam, list[dict[str, float]], dict[str, Any]]:
    """Train the agent, evaluating at each report step. With a `session`, the
    run can stop after a report step: `state(agent_state)` gives the full run
    state to save, and `saved` is the agent state of a stopped run."""
    agent, optimizer = build(config, device, memory)
    generator = torch.Generator().manual_seed(config.model.seed)
    history: list[dict[str, float]] = []
    curve: dict[str, Any] = {}
    if saved is not None:
        agent.load_state_dict(saved["agent"])
        optimizer.load_state_dict(saved["optimizer"])
        generator = resumable.restore_generator(saved["generator"])
        history = saved["history"]
        curve = saved["curve"]
        if session is not None:
            session.restore_global_random_state()
    report = sorted({0, *config.report_steps, config.steps})
    for step in report:
        if step > config.steps:
            break
        if str(step) in curve:
            continue
        history += train(
            model, agent, optimizer, starts, config, generator, step - len(history)
        )
        curve[str(step)] = evaluate(model, agent, config, evaluation_seed)
        if session is not None and state is not None:
            session.boundary(
                lambda trained=history: state(
                    {
                        "agent": agent.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "generator": generator.get_state(),
                        "history": trained,
                        "curve": curve,
                    }
                )
            )
    return agent, optimizer, history, curve


def run(
    config: Config,
    device: torch.device,
    output: Path,
    session: resumable.Session | None = None,
) -> dict[str, Any]:
    """Train the frozen world model, then each agent variant. With a
    `session`, the run can stop after the world model or after any report step
    of an agent and continue in a later process with identical results."""
    session = session or resumable.Session(None, None, {})
    seeds = {
        "torch_initialization": config.model.seed,
        "world_model_data": config.model.seed,
        "world_model_minibatches": config.model.seed,
        "world_model_evaluation_data": config.model.seed
        + world_model.EVALUATION_DATA_SEED_OFFSET,
        "imagination_sampling": config.model.seed,
        "latent_sampling": config.model.seed + LATENT_SEED_OFFSET,
        "real_evaluation": config.model.seed + EVALUATION_SEED_OFFSET,
    }
    saved = session.load()
    if saved is None:
        model, episodes, model_evaluation = train_world_model(config, device)
        agents: dict[str, Any] = {}
        in_progress: dict[str, Any] | None = None
    else:
        model, _ = world_model.build(config.model, device)
        model.load_state_dict(saved["world_model"])
        model.requires_grad_(False)
        model.eval()
        episodes = world_model.Episodes(**saved["episodes"])
        model_evaluation = saved["model_evaluation"]
        agents = saved["agents"]
        in_progress = saved["in_progress"]
        session.restore_global_random_state()

    def state(agent_state: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "world_model": model.state_dict(),
            "episodes": resumable.tensors_of(episodes),
            "model_evaluation": model_evaluation,
            "agents": agents,
            "in_progress": agent_state,
        }

    if saved is None:
        session.boundary(lambda: state(None))
    starts = start_states(
        model,
        episodes.to(device),
        torch.Generator().manual_seed(seeds["latent_sampling"]),
    )
    evaluation_seed = seeds["real_evaluation"]
    random_policy = run_in_environment(
        model,
        None,
        config,
        config.evaluation_episodes,
        torch.Generator().manual_seed(evaluation_seed),
        greedy=False,
    )
    variants = (("recurrent_state", True), ("no_memory", False))
    if config.model.kind != "gru":
        variants = variants[:1]
    for name, memory in variants:
        if name in agents:
            continue
        resume = in_progress if in_progress and in_progress["name"] == name else None
        agent, optimizer, history, curve = train_with_curve(
            model,
            starts,
            memory,
            config,
            device,
            evaluation_seed,
            session,
            resume,
            lambda agent_state, name=name: state(agent_state | {"name": name}),
        )
        checkpoint = output.parent / f"{name}.pt"
        save_checkpoint(checkpoint, config, config.steps, agent, optimizer)
        agents[name] = {
            "memory": memory,
            "parameters": world_model.parameter_count(agent),
            "parameter_shapes": {n: list(p.shape) for n, p in agent.named_parameters()},
            "training_at_steps": {
                str(s): history[s] for s in config.report_steps if s < len(history)
            },
            "final_training": history[-1],
            "evaluation_curve": curve,
            "evaluation": curve[str(config.steps)],
            "checkpoint": str(checkpoint),
        }
        in_progress = None

    result: dict[str, Any] = {
        "stage": "2e-actor-critic-in-imagination-tmaze",
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
            "corridor_length": config.model.corridor_length,
            "max_steps": config.model.max_steps,
        },
        "world_model": {
            "parameters": world_model.parameter_count(model),
            "training_data": world_model.dataset_summary(episodes),
            "held_out_one_step": model_evaluation,
            "frozen_during_actor_critic": True,
        },
        "start_states": int(starts.shape[0]),
        "random_policy": random_policy,
        "agents": agents,
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
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--horizon", type=int, default=Config.horizon)
    world_model.add_world_model_arguments(parser)
    resumable.add_arguments(parser)
    parser.add_argument("--device", default="cpu", choices=DEVICES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runtime.configure(args.seed)
    if args.steps < 1 or args.horizon < 1:
        parser.error("--steps and --horizon must be positive")
    config = Config(
        model=world_model.Config(
            seed=args.seed, **world_model.world_model_options(args)
        ),
        steps=args.steps,
        horizon=args.horizon,
    )
    output = (
        args.output
        or Path("runs")
        / datetime.now(UTC).strftime("stage2e-%Y%m%dT%H%M%S%fZ")
        / "metrics.json"
    )
    session = resumable.session_from(
        args, {"experiment": "actor_critic", "config": asdict(config)}
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
    turn = result["world_model"]["held_out_one_step"]["junction_turn"]
    print(
        f"Frozen world model: {result['world_model']['parameters']} parameters, "
        f"held-out turn reward sign accuracy {turn['reward_sign_accuracy']:.0%}"
    )
    random_policy = result["random_policy"]
    print(
        f"Random policy in the real maze: success {random_policy['success_rate']:.0%}, "
        f"wrong turn {random_policy['wrong_turn_rate']:.0%}, "
        f"return {random_policy['mean_return']:+.3f}"
    )
    for name, agent in result["agents"].items():
        print(f"Agent {name} ({agent['parameters']} parameters), by training step:")
        print(
            "   step  imagined R0  entropy | real greedy: success wrong  return"
            " | sampled: success  imagined-real"
        )
        for step, at in agent["evaluation_curve"].items():
            training = agent["training_at_steps"].get(step, agent["final_training"])
            greedy, sampled = at["greedy"], at["sampled"]
            print(
                f"  {step:>5}  {training['imagined_return']:+.3f}      "
                f"{training['entropy']:.3f}  |  {greedy['success_rate']:>6.0%} "
                f"{greedy['wrong_turn_rate']:>5.0%}  {greedy['mean_return']:+.3f}"
                f" |  {sampled['success_rate']:>6.0%}   "
                f"{sampled['imagined_minus_real_discounted_return']:+.3f}"
            )
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
