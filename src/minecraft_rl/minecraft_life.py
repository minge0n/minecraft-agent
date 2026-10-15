"""Play Minecraft lives without a time limit (docs/stage4-temporal-attack.md).

A life starts with `RESET` (world seed `--world-seed`, fixed spawn) and ends
only when the player dies (`terminated`). Nothing in the environment ends a
life after a number of ticks. A run can still stop a life from outside: the
operational tick budget `--tick-budget` of this process, or a manual
interrupt. Such a life is `truncated`, with `stop_reason` "tick_budget" or
"interrupted", and `terminated` stays false, so the continuation target never
treats it as a death.

Inside one life nothing resets: the world, the player, the inventory, the
RSSM state of the policy, the attack controller, the episode-local reward
memory, the life id and the logical step count. Three things are only
bookkeeping on the same timeline:

- Measurement checkpoints (`--checkpoints`): metrics of the first N ticks.
- Data segments (`--segment-steps`): the life is written in parts, so a long
  life never sits in memory. `join_segments` joins them again.
- Video segments: one video file per data segment.

Each step is exactly one Minecraft tick and one physical action.
"""

import argparse
import json
import math
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy
import torch

from minecraft_rl import macos_energy, macos_qos, runtime
from minecraft_rl.chunk_gate import ChunkGate, listening_pid, set_background
from minecraft_rl.minecraft_actor import ActionSpace
from minecraft_rl.minecraft_dataset import EpisodeMeta, EpisodeWriter, write_json
from minecraft_rl.minecraft_interface import ACTION_SCHEMA, PolicyObservation
from minecraft_rl.minecraft_launch import launched_client, renderer_of
from minecraft_rl.minecraft_play import (
    RECORDING,
    WORKER,
    ActorPolicy,
    RandomBreakPolicy,
)
from minecraft_rl.minecraft_reward import REWARD_V1, EpisodeReward, Registries
from minecraft_rl.minecraft_reward_break import (
    ATTACK_FACTOR,
    BREAK_V1,
    BreakAgent,
    attack_runs,
)
from minecraft_rl.minecraft_temporal_attack import (
    COMMANDS,
    AttackController,
    TemporalBreakAgent,
    temporal_step,
)
from minecraft_rl.provenance import git_commit
from minecraft_rl.recording import WorkerRecorder

WORLD_MODEL = Path("runs/stage4/break-continued/world_model.pt")
PITCH_BINS = (-90, -45, 0, 45, 80, 91)


@dataclass(frozen=True)
class LifeConfig:
    world_seed: int = 100_000
    tick_budget: int | None = None
    checkpoints: tuple[int, ...] = ()
    segment_steps: int = 500

    def to_json(self) -> dict[str, Any]:
        return {
            "world_seed": self.world_seed,
            "fixed_spawn": True,
            "preset": "normal",
            "termination": "death",
            "tick_budget": self.tick_budget,
            "checkpoints": list(self.checkpoints),
            "segment_steps": self.segment_steps,
        }


class PerTickBreakPolicy(ActorPolicy):
    """The previous block-breaking actor: pitch and attack sampled every
    tick."""

    def __init__(self, agent_path: Path, world_model: Path, seed: int) -> None:
        super().__init__(agent_path, world_model, seed, BreakAgent)
        self.value: float | None = None

    @torch.no_grad()
    def act(self, observation: PolicyObservation) -> tuple[Any, list[int]]:
        self.value = float(self.agent.value(self.state))
        return super().act(observation)


class TemporalActorPolicy:
    """The temporal actor in real play: the one-step RSSM filter, the attack
    controller and `temporal_step`, the same function imagination uses."""

    def __init__(self, agent_path: Path, world_model: Path, seed: int) -> None:
        self.inner = ActorPolicy(agent_path, world_model, seed, TemporalBreakAgent)
        self.controller = AttackController.start(1)
        self.value: float | None = None
        self.decision: int | None = None

    @property
    def agent(self) -> TemporalBreakAgent:
        return self.inner.agent

    def reset(self, observation: PolicyObservation) -> None:
        self.inner.reset(observation)
        self.controller = AttackController.start(1)

    @torch.no_grad()
    def act(self, observation: PolicyObservation) -> tuple[Any, list[int]]:
        x, choices, deciding = temporal_step(
            self.agent, self.inner.state, self.controller, self.inner.generator
        )
        self.value = float(self.agent.value(x))
        self.decision = int(choices[0, ATTACK_FACTOR]) if bool(deciding[0]) else None
        choice = choices[0].tolist()
        return ActionSpace.to_player_action(choice), choice

    def observe(self, choice: list[int], observation: PolicyObservation) -> None:
        self.inner.observe(choice, observation)


@dataclass
class LifeTrace:
    """Small per-step series of one life, for checkpoint metrics. The full
    observations go to the data segments, not here."""

    attack: list[bool] = field(default_factory=list)
    breaks: list[int] = field(default_factory=list)
    pitch: list[float] = field(default_factory=list)
    decisions: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    held_items: list[int] = field(default_factory=list)
    item_types: list[int] = field(default_factory=list)
    health: list[float] = field(default_factory=list)

    def metrics(self, ticks: int) -> dict[str, Any]:
        attack = numpy.asarray(self.attack[:ticks], dtype=bool)
        breaks = numpy.asarray(self.breaks[:ticks])
        pitch = numpy.asarray(self.pitch[:ticks])
        runs = attack_runs(attack)
        events = int((breaks > 0).sum())
        decisions = Counter(d for d in self.decisions[:ticks] if d >= 0)
        made = sum(decisions.values())
        values = [v for v in self.values[:ticks] if not math.isnan(v)]
        hist, _ = numpy.histogram(pitch, PITCH_BINS)
        return {
            "ticks": len(attack),
            "blocks_broken": int(breaks.sum()),
            "blocks_broken_per_1000_ticks": 1000.0 * float(breaks.sum()) / len(attack)
            if len(attack)
            else 0.0,
            "break_events": events,
            "real_reward": float(breaks.sum()) * BREAK_V1.per_break,
            "items_held_at_checkpoint": self.held_items[len(attack) - 1]
            if len(attack)
            else 0,
            "item_types_acquired": self.item_types[len(attack) - 1]
            if len(attack)
            else 0,
            "attack_fraction": float(attack.mean()) if len(attack) else 0.0,
            "attack_runs": len(runs),
            "attack_run_lengths": runs,
            "runs_of_15_or_more": sum(r >= 15 for r in runs),
            "decisions": made,
            "commands": {COMMANDS[k]: decisions.get(k, 0) for k in range(2)}
            if self.decisions and any(d >= 0 for d in self.decisions)
            else None,
            "pitch_mean": float(pitch.mean()) if len(pitch) else None,
            "pitch_histogram": {
                f"{PITCH_BINS[i]}..{PITCH_BINS[i + 1]}": int(hist[i])
                for i in range(len(hist))
            },
            "critic_value_mean": float(numpy.mean(values)) if values else None,
            "health_at_checkpoint": self.health[len(attack) - 1]
            if len(attack)
            else None,
        }


def held_count(observation: PolicyObservation) -> int:
    return int(sum(observation.inventory_count)) + int(observation.offhand_count)


class Segment:
    """One stored part of a life, with its own video file."""

    def __init__(self, observation, index: int, first_step: int, first_tick: int):
        self.writer = EpisodeWriter(observation)
        self.index = index
        self.first_step = first_step
        self.first_tick = first_tick
        self.last_tick = first_tick


def play_life(
    client,
    policy,
    life: int,
    config: LifeConfig,
    registries: Registries,
    root: Path,
    recorder: WorkerRecorder | None,
    gate: ChunkGate | None,
    context: dict[str, Any],
) -> dict[str, Any]:
    observation, reset = client.reset(config.world_seed, "normal", fixed_spawn=True)
    if gate is not None:
        gate.wait()
        observation = client.observe()
    reward = EpisodeReward(REWARD_V1, registries)
    reward.reset(observation)
    policy.reset(observation)
    trace = LifeTrace()
    checkpoints: dict[int, dict[str, Any]] = {}
    segments: list[dict[str, Any]] = []
    videos: list[dict[str, Any]] = []
    if recorder is not None:
        recorder.run = root / "video" / f"life-{life:06d}"
        recorder.episode = 0
    segment = Segment(observation, 0, 0, reset.game_time)

    def flush(final: bool, stop_reason: str | None) -> None:
        died = stop_reason == "death"
        meta = EpisodeMeta(
            episode=life,
            world_seed=config.world_seed,
            preset="normal",
            level_id=reset.level_id,
            split="eval",
            policy_seed=context["policy_seed"],
            steps=segment.writer.steps,
            terminated=final and died,
            truncated=final and not died,
            first_tick=segment.first_tick,
            last_tick=segment.last_tick,
            action_schema=ACTION_SCHEMA,
            reward_version=REWARD_V1.version,
            policy=context["policy"],
            episode_return=float(sum(trace.breaks[segment.first_step :])),
            segment=segment.index,
            first_step=segment.first_step,
            final_segment=final,
            stop_reason=stop_reason if final else None,
        )
        segment.writer.write(
            root / "eval" / "episodes" / f"{life:06d}-{segment.index:03d}", meta
        )
        segments.append({"segment": segment.index, "steps": segment.writer.steps})
        if recorder is not None:
            _, encoded = recorder.stop()
            videos.append(
                {
                    "segment": segment.index,
                    "status": encoded.status,
                    "bytes": encoded.video_bytes,
                    "directory": str(encoded.directory),
                }
            )

    if recorder is not None:
        recorder.start()
    started = time.perf_counter()
    steps = 0
    stop_reason = None
    try:
        while True:
            if config.tick_budget is not None and steps >= config.tick_budget:
                stop_reason = "tick_budget"
                break
            if segment.writer.steps >= config.segment_steps:
                flush(False, None)
                segment = Segment(
                    observation, segment.index + 1, steps, segment.last_tick
                )
                if recorder is not None:
                    recorder.start()
            action, choice = policy.act(observation)
            if gate is not None:
                gate.wait()
            result = client.step(action)
            if result.info.tick_after != result.info.tick_before + 1:
                raise RuntimeError("a step did not advance exactly one tick")
            observation = result.observation
            step_reward = reward.step(
                observation,
                result.terminated,
                result.events.blocks_broken,
                result.events.advancements,
            )
            policy.observe(choice, observation)
            segment.writer.add(
                action.executed(),
                observation,
                result.terminated,
                result.info.step_id,
                result.info.tick_before,
                result.info.tick_after,
                result.info.client_tick,
                reward=step_reward.total,
                components=list(step_reward.components.values()),
                events=step_reward.events,
            )
            segment.writer.add_privileged_step(
                "breaks", len(result.events.blocks_broken)
            )
            segment.last_tick = result.info.tick_after
            steps += 1
            trace.attack.append(bool(action.attack))
            trace.breaks.append(len(result.events.blocks_broken))
            trace.pitch.append(float(observation.pitch))
            decision = getattr(policy, "decision", None)
            trace.decisions.append(-1 if decision is None else decision)
            value = getattr(policy, "value", None)
            trace.values.append(math.nan if value is None else value)
            trace.held_items.append(held_count(observation))
            trace.item_types.append(reward.counts["items_acquired"])
            trace.health.append(float(observation.health))
            if steps in config.checkpoints:
                checkpoints[steps] = trace.metrics(steps)
                print(json.dumps({"life": life, "checkpoint": steps}), flush=True)
            if result.terminated:
                stop_reason = "death"
                break
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        flush(True, stop_reason)
        raise
    flush(True, stop_reason)
    wall = time.perf_counter() - started
    summary = {
        "life": life,
        "policy": context["policy"],
        "policy_seed": context["policy_seed"],
        "ticks": steps,
        "died": stop_reason == "death",
        "terminated": stop_reason == "death",
        "truncated": stop_reason != "death",
        "stop_reason": stop_reason,
        "complete_life": stop_reason == "death",
        "steps_per_second": steps / wall if wall else 0.0,
        "checkpoints": {str(k): v for k, v in checkpoints.items()},
        "final": trace.metrics(steps),
        "segments": segments,
        "videos": videos,
        "reward_v1": reward.summary(),
    }
    write_json(root / "lives" / f"{life:06d}.json", summary)
    return summary


def make_policy(args, seed: int):
    if args.policy == "random-break":
        return RandomBreakPolicy(seed)
    if args.policy == "break-actor":
        return PerTickBreakPolicy(args.agent, args.world_model, seed)
    return TemporalActorPolicy(args.agent, args.world_model, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--policy",
        choices=("random-break", "break-actor", "temporal-actor"),
        required=True,
    )
    parser.add_argument("--agent", type=Path)
    parser.add_argument("--world-model", type=Path, default=WORLD_MODEL)
    parser.add_argument("--lives", type=int, required=True)
    parser.add_argument("--world-seed", type=int, default=100_000)
    parser.add_argument("--policy-seed", type=int, default=7_000_000)
    parser.add_argument(
        "--tick-budget",
        type=int,
        help="operational limit of this run; a life stopped by it is truncated",
    )
    parser.add_argument("--checkpoints", default="200,1000")
    parser.add_argument("--segment-steps", type=int, default=500)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--port", type=int, default=47140)
    parser.add_argument(
        "--foreground-game",
        action="store_true",
        help="macOS: run the game at default priority instead of on the "
        "efficiency cores behind a chunk gate",
    )
    parser.add_argument(
        "--draw-every-frame",
        action="store_true",
        help="draw every frame the render loop allows, not only the frames "
        "the recording keeps",
    )
    args = parser.parse_args()
    args.background_game = not args.foreground_game
    if args.policy != "random-break" and args.agent is None:
        parser.error("--agent is required for an actor policy")
    macos_qos.run_in_background()
    runtime.configure(0)
    config = LifeConfig(
        world_seed=args.world_seed,
        tick_budget=args.tick_budget,
        checkpoints=tuple(int(c) for c in args.checkpoints.split(",") if c),
        segment_steps=args.segment_steps,
    )
    todo = [
        life
        for life in range(args.lives)
        if not (args.output / "lives" / f"{life:06d}.json").exists()
    ]
    if not todo:
        print("all lives exist")
        return
    write_json(
        args.output / "manifest.json",
        {
            "format": "minecraft-lives-v1",
            "git_commit": git_commit(),
            "life": config.to_json(),
            "policy": args.policy,
            "agent": str(args.agent) if args.agent else None,
            "world_model": str(args.world_model),
            "policy_seed": args.policy_seed,
            "renderer": renderer_of(os.environ),
            "action_schema": ACTION_SCHEMA,
            "reward": BREAK_V1.to_json(),
            "recording": None if args.no_video else RECORDING,
            "background_game": args.background_game,
            "recorded_frames_only": not args.draw_every_frame,
        },
    )
    log = args.output / "logs" / f"{int(time.time())}.minecraft.log"
    started = time.monotonic()
    with launched_client(args.port, args.world_seed, log, 240.0) as client:
        write_json(args.output / "schema.json", client.request("SCHEMA"))
        registries = Registries.from_json(client.privileged().registry())
        client.set_pacing("unpaced", recorded_frames_only=not args.draw_every_frame)
        pid = listening_pid(args.port)
        gate = None
        if args.background_game and pid is not None:
            set_background(pid, True)
            gate = ChunkGate(client.privileged(), pid)
        energy_before = macos_energy.read(pid) if pid else None
        recorder = None
        if not args.no_video:
            recorder = WorkerRecorder(
                client,
                args.output / "video",
                WORKER,
                RECORDING,
                {"run": str(args.output), "policy": args.policy},
            )
        try:
            for life in todo:
                seed = args.policy_seed + life
                summary = play_life(
                    client,
                    make_policy(args, seed),
                    life,
                    config,
                    registries,
                    args.output,
                    recorder,
                    gate,
                    {"policy": args.policy, "policy_seed": seed},
                )
                print(
                    json.dumps(
                        {
                            k: summary[k]
                            for k in ("life", "ticks", "stop_reason", "died")
                        }
                        | {
                            "breaks": summary["final"]["blocks_broken"],
                            "attack_fraction": summary["final"]["attack_fraction"],
                        }
                    ),
                    flush=True,
                )
        finally:
            if recorder is not None:
                recorder.close()
            energy_after = macos_energy.read(pid) if pid else None
            usage: dict[str, Any] = {
                "background_game": gate is not None,
                "gate": gate.summary() if gate is not None else None,
                "wall_seconds": time.monotonic() - started,
            }
            if energy_before is not None and energy_after is not None:
                usage["game_energy_joules"] = (
                    energy_after - energy_before
                ).energy_joules
            write_json(args.output / "logs" / "usage.json", usage)


if __name__ == "__main__":
    main()
