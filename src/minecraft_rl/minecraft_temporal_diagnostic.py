"""Diagnose the gap between imagined and real return of the temporal attack
actor (docs/stage4-temporal-diagnostic.md).

Reads the recorded lives of `runs/stage4/temporal-eval` and the frozen
continued world model, reward head and trained actor of
`runs/stage4/break-continued` and `runs/stage4/temporal`. Nothing here
trains or changes a model. Every comparison starts from the same real
posterior state and uses the same recorded physical actions, horizon and
discount, so the three error sources stay apart:

1. `reward_head`: the reward head on real posterior states against the real
   breaks, per block type.
2. `open_loop`: from a real posterior state, the RSSM imagines the recorded
   actions for up to 32 ticks. Its reward and its center ray are compared
   with the matching real trajectory at horizons 1, 5, 10, 20 and 32.
3. `returns`: imagined and real reward sums, discounted sums, the critic
   bootstrap value and the lambda return over the same 32 ticks, the
   actor's own imagined return from the same states, and the critic value
   against the real discounted return of the rest of the life.

The controller phase of every recorded tick is rebuilt from the recorded
attack runs: each run starts at a decision tick, so a state after k
consecutive attack ticks has `(-k) mod 20` forced hold ticks left. The
recorded runs are all multiples of 20 except the run cut by the tick budget,
which the script checks.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy
import torch

from minecraft_rl import macos_qos, runtime
from minecraft_rl.actor_critic import lambda_returns
from minecraft_rl.minecraft_actor import RewardHead, load_world_model
from minecraft_rl.minecraft_break_train import attack_run_lengths
from minecraft_rl.minecraft_dataset import ACTION_BUTTONS, join_segments, load_episodes
from minecraft_rl.minecraft_interface import RAY_KIND_BLOCK
from minecraft_rl.minecraft_replay import CompactVocabulary, compact_arrays
from minecraft_rl.minecraft_reward_break import BREAKS_KEY, attack_runs
from minecraft_rl.minecraft_temporal_attack import (
    HOLD_TICKS,
    TemporalBreakAgent,
    imagine_temporal,
)
from minecraft_rl.minecraft_world_model import (
    action_tensor,
    observation_tensors,
    ray_class,
)
from minecraft_rl.provenance import git_commit

RUN = Path("runs/stage4/break-continued")
AGENT = Path("runs/stage4/temporal/agent-200.pt")
EVAL = Path("runs/stage4/temporal-eval")
POLICIES = ("trained", "untrained")
HORIZONS = (1, 5, 10, 20, 32)
HORIZON = 32
DISCOUNT = 0.99
RETURN_LAMBDA = 0.95
SAMPLES = 4
ATTACK = ACTION_BUTTONS.index("attack")
SAND, SANDSTONE = 39, 115
BLOCK_NAMES = {SAND: "minecraft:sand", SANDSTONE: "minecraft:sandstone"}


def center_index(rows: int, columns: int) -> int:
    return (rows // 2) * columns + columns // 2


def controller_phase(attack: numpy.ndarray) -> numpy.ndarray:
    """Forced hold ticks left before each recorded step (T,)."""
    run = attack_run_lengths(attack)
    before = numpy.concatenate([[0], run[:-1]])
    return (-before) % HOLD_TICKS


def mining_ticks(attack, center_type, center_distance) -> numpy.ndarray:
    """For each step, the consecutive attack ticks so far with the center ray
    on the same block (same type, distance within 0.05 blocks)."""
    out = numpy.zeros(len(attack), dtype=numpy.int64)
    for t in range(len(attack)):
        same = (
            t > 0
            and attack[t - 1]
            and center_type[t] == center_type[t - 1]
            and abs(center_distance[t] - center_distance[t - 1]) < 0.05
        )
        out[t] = out[t - 1] + 1 if same and attack[t] else int(attack[t])
    return out


def load_life(episode, compact, model, generator):
    a = compact_arrays(episode.arrays, compact)
    o = observation_tensors(a)
    actions = action_tensor(
        torch.as_tensor(a["action_buttons"]),
        torch.as_tensor(a["action_camera"]),
        torch.as_tensor(a["action_hotbar"]),
    )
    with torch.no_grad():
        filtered = model.filter_embedded(
            model.encoder(o).unsqueeze(0), actions.unsqueeze(0), generator
        )
    v = model.vocabulary
    c = center_index(v.rows, v.columns)
    attack = episode.arrays["action_buttons"][:, ATTACK].astype(bool)
    raw_type = episode.arrays["obs_ray_type"][:, c].astype(numpy.int64)
    raw_kind = episode.arrays["obs_ray_kind"][:, c]
    distance = episode.arrays["obs_ray_distance"][:, c].astype(numpy.float64)
    runs = attack_runs(attack)
    complete = runs[:-1] if attack[-1] else runs
    if any(r % HOLD_TICKS for r in complete):
        raise ValueError(f"life {episode.meta.episode}: a run is not a whole hold")
    return {
        "life": episode.meta.episode,
        "h": filtered["h"][0],
        "z": filtered["z"][0],
        "actions": actions,
        "breaks": torch.as_tensor(episode.arrays[BREAKS_KEY]).float(),
        "attack": attack,
        "phase": torch.as_tensor(controller_phase(attack)),
        "center_block": numpy.where(raw_kind == RAY_KIND_BLOCK, raw_type, -1),
        "center_distance": distance,
        "center_class": ray_class(o["ray_kind"], o["ray_type"], v)[:, c],
        "mining": mining_ticks(attack, raw_type, distance),
        "inventory": episode.arrays["obs_inventory_count"].sum(1).astype(int),
    }


def block_of(life, t: int) -> str:
    block = int(life["center_block"][t])
    return {SAND: "sand", SANDSTONE: "sandstone"}.get(block, "other")


@torch.no_grad()
def reward_head_report(lives, head) -> dict[str, Any]:
    by_type: dict[str, dict[str, list]] = {
        k: {"at_break": [], "window_max": [], "before": [], "mining_at_break": []}
        for k in ("sand", "sandstone")
    }
    false = {"sand": 0.0, "sandstone": 0.0, "other": 0.0}
    false_steps = {"sand": 0, "sandstone": 0, "other": 0}
    by_phase: dict[str, list[float]] = {}
    per_life = []
    items_after = {"sand": [], "sandstone": []}
    for life in lives:
        s = torch.cat([life["h"], life["z"]], -1)
        predicted = head.reward(s[1:]).numpy()
        breaks = life["breaks"].numpy()
        per_life.append(
            {
                "life": life["life"],
                "predicted_sum": float(predicted.sum()),
                "actual_sum": float(breaks.sum()),
                "predicted_per_tick": float(predicted.mean()),
                "actual_per_tick": float(breaks.mean()),
            }
        )
        near = numpy.zeros(len(breaks), dtype=bool)
        for t in numpy.flatnonzero(breaks > 0):
            near[max(0, t - 2) : t + 3] = True
            kind = block_of(life, t)
            if kind == "other":
                continue
            row = by_type[kind]
            row["at_break"].append(float(predicted[t]))
            row["window_max"].append(float(predicted[max(0, t - 2) : t + 3].max()))
            row["before"].append(float(predicted[max(0, t - 10) : max(0, t - 2)].sum()))
            row["mining_at_break"].append(int(life["mining"][t]))
            inventory = life["inventory"]
            items_after[kind].append(
                int(inventory[min(len(inventory) - 1, t + 15)] - inventory[t])
            )
        for t in numpy.flatnonzero(~near):
            kind = block_of(life, t)
            false[kind] += float(max(predicted[t], 0.0))
            false_steps[kind] += 1
            mining = int(life["mining"][t])
            bucket = (
                "0"
                if mining == 0
                else "1-14"
                if mining < 15
                else "15-29"
                if mining < 30
                else "30-79"
                if mining < 80
                else "80+"
            )
            by_phase.setdefault(f"{kind} {bucket}", []).append(float(predicted[t]))
    return {
        "breaks": {
            kind: {
                "count": len(row["at_break"]),
                "predicted_at_break_mean": float(numpy.mean(row["at_break"]))
                if row["at_break"]
                else None,
                "predicted_window_max_mean": float(numpy.mean(row["window_max"]))
                if row["window_max"]
                else None,
                "predicted_sum_8_ticks_before_mean": float(numpy.mean(row["before"]))
                if row["before"]
                else None,
                "mining_ticks_at_break": sorted(set(row["mining_at_break"])),
                "inventory_change_15_ticks_after": sorted(set(items_after[kind])),
            }
            for kind, row in by_type.items()
        },
        "false_positive": {
            kind: {
                "steps": false_steps[kind],
                "positive_predicted_sum": false[kind],
                "per_step": false[kind] / max(1, false_steps[kind]),
            }
            for kind in false
        },
        "non_break_prediction_by_mining_ticks": {
            key: {"steps": len(v), "mean": float(numpy.mean(v))}
            for key, v in sorted(by_phase.items())
        },
        "per_life": per_life,
    }


def start_indices(life, stride: int) -> numpy.ndarray:
    steps = len(life["breaks"])
    return numpy.arange(0, steps - HORIZON, stride)


@torch.no_grad()
def open_loop(model, head, agent, life, starts, generator):
    """Imagine the recorded actions from real posterior states. Returns per
    start (n, 32) mean imagined reward, the critic value at the horizon,
    imagined continuation, and the decoded center ray at the horizons."""
    v = model.vocabulary
    c = center_index(v.rows, v.columns)
    index = torch.as_tensor(starts)
    h = life["h"][index].repeat_interleave(SAMPLES, 0)
    z = life["z"][index].repeat_interleave(SAMPLES, 0)
    n = len(starts)
    rewards, continues, classes, distances = [], [], {}, {}
    for k in range(HORIZON):
        action = life["actions"][index + k].repeat_interleave(SAMPLES, 0)
        h = model.transition(h, z, action)
        z = model.sample(model.prior_logits(h), generator, straight_through=False)
        s = torch.cat([h, z], -1)
        rewards.append(head.reward(s).reshape(n, SAMPLES).mean(1))
        continues.append(
            torch.sigmoid(model.continue_head(s).squeeze(-1))
            .reshape(n, SAMPLES)
            .mean(1)
        )
        if k + 1 in HORIZONS:
            decoded = model.decoder(s)
            classes[k + 1] = decoded["ray_class"][:, c].argmax(-1).reshape(n, SAMPLES)
            distances[k + 1] = (
                decoded["ray_distance"][:, c].reshape(n, SAMPLES) * v.max_distance
            )
    bootstrap = None
    if agent is not None:
        last = len(life["phase"]) - 1
        phase_end = life["phase"][torch.clamp(index + HORIZON, max=last)]
        x_end = agent.inputs(s, phase_end.repeat_interleave(SAMPLES, 0))
        bootstrap = agent.value(x_end).reshape(n, SAMPLES).mean(1)
    return {
        "rewards": torch.stack(rewards, 1),
        "continues": torch.stack(continues, 1),
        "bootstrap": bootstrap,
        "classes": classes,
        "distances": distances,
    }


def discounted(rewards: torch.Tensor, discount: float) -> torch.Tensor:
    weights = discount ** torch.arange(rewards.shape[-1], dtype=torch.float32)
    return (rewards * weights).sum(-1)


@torch.no_grad()
def open_loop_report(model, head, agent, lives, stride, generator) -> dict[str, Any]:
    groups = ("sand", "sandstone", "other", "all")
    acc: dict[str, dict[str, list]] = {g: {} for g in groups}

    def add(group, key, value):
        acc[group].setdefault(key, []).append(value)

    for life in lives:
        starts = start_indices(life, stride)
        out = open_loop(model, head, agent, life, starts, generator)
        s_real = torch.cat([life["h"], life["z"]], -1)
        posterior_reward = head.reward(s_real[1:])
        if agent is not None:
            phase = torch.cat([life["phase"], life["phase"][-1:]])
            real_values = agent.value(agent.inputs(s_real, phase))
        for i, t in enumerate(starts):
            group = block_of(life, int(t))
            real = life["breaks"][t : t + HORIZON]
            imagined = out["rewards"][i]
            for g in (group, "all"):
                for k in HORIZONS:
                    add(g, f"imagined_sum_{k}", float(imagined[:k].sum()))
                    add(
                        g,
                        f"posterior_sum_{k}",
                        float(posterior_reward[t : t + k].sum()),
                    )
                    add(g, f"real_sum_{k}", float(real[:k].sum()))
                    real_class = int(life["center_class"][t + k])
                    real_distance = float(life["center_distance"][t + k])
                    add(
                        g,
                        f"center_class_correct_{k}",
                        float((out["classes"][k][i] == real_class).float().mean()),
                    )
                    add(
                        g,
                        f"center_distance_error_{k}",
                        float((out["distances"][k][i] - real_distance).abs().mean()),
                    )
                    changed = int(life["center_class"][t]) != real_class
                    if changed:
                        add(
                            g,
                            f"changed_center_class_correct_{k}",
                            float((out["classes"][k][i] == real_class).float().mean()),
                        )
                add(g, "imagined_discounted_32", float(discounted(imagined, DISCOUNT)))
                add(
                    g,
                    "posterior_discounted_32",
                    float(discounted(posterior_reward[t : t + HORIZON], DISCOUNT)),
                )
                add(g, "real_discounted_32", float(discounted(real, DISCOUNT)))
                add(g, "real_breaks_in_32", float(real.sum()))
                add(g, "mining_ticks_at_start", int(life["mining"][t]))
                if agent is None:
                    continue
                add(g, "imagined_bootstrap", float(out["bootstrap"][i]))
                add(g, "real_bootstrap", float(real_values[t + HORIZON]))
                tail = DISCOUNT**HORIZON
                imagined_bootstrapped = float(
                    discounted(imagined, DISCOUNT) + tail * out["bootstrap"][i]
                )
                real_bootstrapped = float(
                    discounted(real, DISCOUNT) + tail * real_values[t + HORIZON]
                )
                add(g, "imagined_return_with_bootstrap", imagined_bootstrapped)
                add(g, "real_return_with_bootstrap", real_bootstrapped)
    report = {}
    for g in groups:
        values = acc[g]
        if not values:
            continue
        report[g] = {"starts": len(values["real_sum_1"])} | {
            key: float(numpy.mean(v)) for key, v in values.items()
        }
    return report


@torch.no_grad()
def actor_return_report(model, head, agent, lives, stride, generator) -> dict[str, Any]:
    """The actor's own imagined lambda return from the real states, as in
    training (lambda 0.95), against the real 32-tick outcome of the same
    states and the critic against the real discounted return of the rest of
    the life."""
    imagined_return, imagined_reward, real_reward_32, critic, real_rest = (
        [],
        [],
        [],
        [],
        [],
    )
    for life in lives:
        starts = torch.as_tensor(start_indices(life, stride))
        imagined = imagine_temporal(
            model,
            head,
            agent,
            life["h"][starts],
            life["z"][starts],
            HORIZON,
            generator,
            life["phase"][starts],
        )
        values = agent.value(imagined.features)
        returns = lambda_returns(
            imagined.rewards, imagined.continues, values, DISCOUNT, RETURN_LAMBDA
        )
        imagined_return += returns[:, 0].tolist()
        imagined_reward += imagined.rewards.sum(1).tolist()
        breaks = life["breaks"]
        real_reward_32 += [float(breaks[t : t + HORIZON].sum()) for t in starts]
        s_real = torch.cat([life["h"], life["z"]], -1)
        x = agent.inputs(s_real[: len(breaks)], life["phase"])
        value = agent.value(x)
        tail = len(breaks)
        weights = DISCOUNT ** torch.arange(tail, dtype=torch.float32)
        for t in starts.tolist():
            if tail - t < 460:
                continue
            critic.append(float(value[t]))
            real_rest.append(float((breaks[t:] * weights[: tail - t]).sum()))
    return {
        "imagined_lambda_return": float(numpy.mean(imagined_return)),
        "imagined_reward_sum_32": float(numpy.mean(imagined_reward)),
        "real_reward_sum_32": float(numpy.mean(real_reward_32)),
        "critic_value_on_real_states": float(numpy.mean(critic)),
        "real_discounted_return_rest_of_life": float(numpy.mean(real_rest)),
        "critic_states": len(critic),
        "note": "rest of life uses starts with at least 460 recorded ticks "
        "after them (discount weight below 0.01 beyond)",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/stage4/temporal-diagnostic")
    )
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--agent", type=Path, default=AGENT)
    parser.add_argument("--stride", type=int, default=2)
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
    agent = TemporalBreakAgent(model.features)
    agent.load_state_dict(torch.load(args.agent, weights_only=True)["agent"])
    agent.eval()
    generator = torch.Generator().manual_seed(args.seed + 1)
    out: dict[str, Any] = {
        "git_commit": git_commit(),
        "world_model": str(args.run / "world_model.pt"),
        "reward_head": str(args.run / "reward_head.pt"),
        "agent": str(args.agent),
        "horizon": HORIZON,
        "discount": DISCOUNT,
        "imagination_samples": SAMPLES,
        "start_stride": args.stride,
        "blocks": BLOCK_NAMES,
    }
    for policy in POLICIES:
        episodes = load_episodes(EVAL / policy / "eval")
        lives = [
            load_life(e, compact, model, generator)
            for e in sorted(join_segments(episodes), key=lambda e: e.meta.episode)
        ]
        out[policy] = {
            "lives": len(lives),
            "ticks": sum(len(life["breaks"]) for life in lives),
            "reward_head": reward_head_report(lives, head),
            "open_loop": open_loop_report(
                model, head, agent, lives, args.stride, generator
            ),
            "actor_return": actor_return_report(
                model, head, agent, lives, args.stride, generator
            ),
        }
        print(json.dumps({policy: out[policy]["actor_return"]}), flush=True)
    (args.output / "diagnostic.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
