"""Train the temporal attack actor in imagination
(docs/stage4-temporal-attack.md).

Reuses the continued block-breaking world model and its `reward-break-v1`
reward head (`runs/stage4/break-continued`), both frozen, and the training
episodes of `runs/stage4/break-random2`. Stages, each skipped when its
result exists in `--output`:

1. `probe.json`: imagined reward over 32 ticks from in-reach start states
   that look down, for four fixed physical attack sequences.
2. `agent-0.pt`, `agent-<updates>.pt`, `actor.json`: the existing
   actor-critic losses with the attack controller, horizon 32.

Start states carry a controller state. The data was not collected with the
controller, so a state after k consecutive attack ticks gets
`remaining = (-k) mod 20`: the state is read as a hold that started at a
decision tick. A state after a release is a decision tick.
"""

import argparse
import json
from pathlib import Path

import torch

from minecraft_rl import macos_qos, runtime
from minecraft_rl.minecraft_actor import (
    ActionSpace,
    ActorConfig,
    RewardHead,
    load_world_model,
)
from minecraft_rl.minecraft_break_train import break_data, write
from minecraft_rl.minecraft_dataset import load_episodes
from minecraft_rl.minecraft_replay import CompactVocabulary
from minecraft_rl.minecraft_reward_break import ATTACK_FACTOR
from minecraft_rl.minecraft_reward_model import chronological_states
from minecraft_rl.minecraft_temporal_attack import (
    HOLD_TICKS,
    TemporalBreakAgent,
    command_statistics,
    imagine_temporal,
    train_temporal_actor,
)
from minecraft_rl.provenance import git_commit

RUN = Path("runs/stage4/break-continued")
HORIZON = 32
PROBE_SEQUENCES = {
    "no attack": [0] * HORIZON,
    "short attack (5 ticks)": [1] * 5 + [0] * (HORIZON - 5),
    "one 20-tick hold": [1] * 20 + [0] * (HORIZON - 20),
    "two 20-tick holds": [1] * HORIZON,
}


def controller_state(attack_run: torch.Tensor) -> torch.Tensor:
    return (-attack_run) % HOLD_TICKS


def start_states(model, data, seed: int, rows: int, columns: int):
    states = chronological_states(model, data, seed=seed)
    h, z, remaining, pitch, center = [], [], [], [], []
    for d, s in zip(data, states, strict=True):
        run = torch.cat([torch.zeros(1, dtype=torch.long), d.labels["attack_run"]])
        h.append(s[:, : model.hidden])
        z.append(s[:, model.hidden :])
        remaining.append(controller_state(run))
        pitch.append(d.observations["pitch"])
        rays = d.observations["ray_distance"]
        center.append(rays[:, (rows // 2) * columns + columns // 2])
    return (
        torch.cat(h),
        torch.cat(z),
        torch.cat(remaining),
        torch.cat(pitch),
        torch.cat(center),
    )


@torch.no_grad()
def probe(model, head, h, z, generator) -> dict:
    out = {}
    for name, attack in PROBE_SEQUENCES.items():
        hh, zz = h.clone(), z.clone()
        rewards = []
        for held in attack:
            choice = torch.as_tensor(ActionSpace.NOOP).expand(h.shape[0], -1).clone()
            choice[:, ATTACK_FACTOR] = held
            hh = model.transition(hh, zz, ActionSpace.world_model_action(choice))
            zz = model.sample(model.prior_logits(hh), generator, straight_through=False)
            rewards.append(head.reward(torch.cat([hh, zz], -1)))
        r = torch.stack(rewards, 1)
        out[name] = {
            "attack_ticks": sum(attack),
            "predicted_reward_32": float(r.sum(1).mean()),
            "predicted_reward_first_20": float(r[:, :20].sum(1).mean()),
            "share_reaching_0.3": float((r.max(1).values >= 0.3).float().mean()),
            "per_tick": [round(float(x), 4) for x in r.mean(0)],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument(
        "--dataset", type=Path, default=Path("runs/stage4/break-random2")
    )
    parser.add_argument("--test-episodes", type=int, default=6)
    parser.add_argument("--actor-updates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    macos_qos.run_in_background()
    runtime.configure(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    model, extra = load_world_model(args.run / "world_model.pt")
    compact = CompactVocabulary.from_json(extra["vocabulary"])
    saved = torch.load(args.run / "reward_head.pt", weights_only=True)
    head = RewardHead(model.features)
    head.load_state_dict(saved["reward_head"])
    head.scale = saved["scale"]
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    episodes = sorted(
        load_episodes(args.dataset / "train"), key=lambda e: e.meta.episode
    )
    train = break_data(model, episodes[: -args.test_episodes], compact)
    h, z, remaining, pitch, center = start_states(
        model, train, 3, model.vocabulary.rows, model.vocabulary.columns
    )
    probe_path = args.output / "probe.json"
    if not probe_path.exists():
        in_reach = (pitch >= 60) & (center <= 4.5) & (remaining == 0)
        index = torch.nonzero(in_reach).squeeze(1)
        index = index[
            torch.randperm(len(index), generator=torch.Generator().manual_seed(1))
        ][:200]
        result = {
            "start_states": int(len(index)),
            "selection": "pitch >= 60, center ray <= 4.5 blocks, decision tick",
            "horizon": HORIZON,
            "sequences": probe(
                model, head, h[index], z[index], torch.Generator().manual_seed(2)
            ),
        }
        write(probe_path, result)
        print(
            json.dumps(
                {k: v["predicted_reward_32"] for k, v in result["sequences"].items()}
            ),
            flush=True,
        )
    final = args.output / f"agent-{args.actor_updates}.pt"
    if final.exists():
        return
    config = ActorConfig(horizon=HORIZON, seed=args.seed)
    torch.manual_seed(args.seed)
    agent = TemporalBreakAgent(model.features)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate)
    record = {"config": config.to_json(), "updates": 0}
    torch.save({"agent": agent.state_dict()} | record, args.output / "agent-0.pt")
    generator = torch.Generator().manual_seed(args.seed + 10)
    before = command_statistics(
        agent,
        imagine_temporal(
            model, head, agent, h[:512], z[:512], HORIZON, generator, remaining[:512]
        ),
    )
    history = train_temporal_actor(
        model,
        head,
        agent,
        optimizer,
        (h, z, remaining),
        config,
        generator,
        args.actor_updates,
    )
    record["updates"] = args.actor_updates
    torch.save({"agent": agent.state_dict()} | record, final)
    recent = history[-50:]

    def mean(key: str) -> float:
        values = [x[key] for x in recent if x[key] is not None]
        return sum(values) / len(values)

    write(
        args.output / "actor.json",
        {
            "git_commit": git_commit(),
            "config": config.to_json(),
            "world_model": str(args.run / "world_model.pt"),
            "reward_head": str(args.run / "reward_head.pt"),
            "action_space": "pitch every tick; RELEASE_ATTACK or HOLD_ATTACK_20 "
            "at decision ticks; every other factor at no-op",
            "start_states": int(h.shape[0]),
            "start_decision_fraction": float((remaining == 0).float().mean()),
            "untrained": before,
            "imagined_return_first_10": sum(x["imagined_return"] for x in history[:10])
            / 10,
            "last_50": {
                key: mean(key)
                for key in (
                    "imagined_return",
                    "imagined_reward_per_step",
                    "critic_value",
                    "hold_fraction_of_decisions",
                    "command_entropy",
                    "pitch_entropy",
                    "attack_fraction",
                    "decision_fraction",
                )
            },
            "history": history[::10],
        },
    )
    print(json.dumps({"last_50": mean("imagined_return")}), flush=True)


if __name__ == "__main__":
    main()
