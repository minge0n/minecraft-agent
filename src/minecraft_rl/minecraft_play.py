"""Play recorded Minecraft episodes in one fixed world and store them with
reward (docs/stage4-minecraft-actor.md).

One process launches the game and plays `--episodes` episodes of
`--max-steps` steps. Every episode resets to world seed `--world-seed` with
the fixed spawn, so episodes differ only by policy and by Minecraft's own
randomness. The policy is either the strategy-free exploration policy
(`--policy random`) or a trained actor (`--policy actor --agent PATH`), which
acts from the frozen RSSM posterior state.

Each step stores the observation, the executed physical action, the
`reward-v1` scalar and its components, termination and the privileged events
(dataset format `minecraft-episodes-v2`). Episodes are recorded on video.
"""

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy
import torch

from minecraft_rl import macos_energy, macos_qos, runtime
from minecraft_rl.chunk_gate import ChunkGate, listening_pid, set_background
from minecraft_rl.exploration import ExplorationConfig, ExplorationPolicy
from minecraft_rl.minecraft_actor import (
    ActionSpace,
    Agent,
    Filter,
    load_world_model,
)
from minecraft_rl.minecraft_dataset import (
    ACTION_BUTTONS,
    OBSERVATION_DTYPES,
    EpisodeMeta,
    EpisodeWriter,
    write_json,
)
from minecraft_rl.minecraft_interface import ACTION_SCHEMA, PolicyObservation
from minecraft_rl.minecraft_launch import launched_client
from minecraft_rl.minecraft_replay import CompactVocabulary, compact_arrays
from minecraft_rl.minecraft_reward import (
    COMPONENTS,
    REWARD_V1,
    EpisodeReward,
    Registries,
)
from minecraft_rl.minecraft_reward_break import (
    ATTACK_FACTOR,
    BREAKS_KEY,
    BreakAgent,
    break_metrics,
)
from minecraft_rl.minecraft_reward_control import episode_task
from minecraft_rl.minecraft_reward_pitch import (
    PITCH_FACTOR,
    PitchOnlyAgent,
    pitch_metrics,
)
from minecraft_rl.provenance import git_commit
from minecraft_rl.recording import WorkerRecorder

WORKER = "env-0000"
RECORDING = {"every_ticks": 2, "width": 426, "height": 240, "queue_frames": 64}


def observation_arrays(
    observation: PolicyObservation, vocabulary: CompactVocabulary
) -> dict:
    """One observation as compact dataset arrays, for the RSSM filter."""
    raw = {
        f"obs_{name}": numpy.asarray(getattr(observation, name), dtype=dtype)
        for name, dtype in OBSERVATION_DTYPES.items()
    }
    return compact_arrays(raw, vocabulary)


class ActorPolicy:
    def __init__(
        self, agent_path: Path, world_model: Path, seed: int, agent_class=Agent
    ) -> None:
        self.model, extra = load_world_model(world_model)
        self.vocabulary = CompactVocabulary.from_json(extra["vocabulary"])
        data = torch.load(agent_path, weights_only=True)
        self.agent = agent_class(self.model.features)
        self.agent.load_state_dict(data["agent"])
        self.agent.eval()
        self.generator = torch.Generator().manual_seed(seed)
        self.filter = Filter(self.model, self.generator)
        self.state: torch.Tensor | None = None

    def reset(self, observation: PolicyObservation) -> None:
        self.state = self.filter.reset(observation_arrays(observation, self.vocabulary))

    @torch.no_grad()
    def act(self, observation: PolicyObservation) -> tuple[Any, list[int]]:
        choice = self.agent.act(self.state, self.generator)[0]
        return ActionSpace.to_player_action(choice.tolist()), choice.tolist()

    def observe(self, choice: list[int], observation: PolicyObservation) -> None:
        self.state = self.filter.step(
            torch.tensor(choice), observation_arrays(observation, self.vocabulary)
        )


class RandomPitchPolicy:
    """The random baseline of the pitch diagnostic: a uniform pitch choice
    every step, every other factor at its no-op value."""

    def __init__(self, seed: int) -> None:
        self.generator = torch.Generator().manual_seed(seed)

    def reset(self, observation: PolicyObservation) -> None:
        pass

    def act(self, observation: PolicyObservation) -> tuple[Any, list[int]]:
        choice = list(ActionSpace.NOOP)
        choice[PITCH_FACTOR] = int(
            torch.randint(ActionSpace.SIZES[PITCH_FACTOR], (), generator=self.generator)
        )
        return ActionSpace.to_player_action(choice), choice

    def observe(self, choice: list[int], observation: PolicyObservation) -> None:
        pass


class RandomBreakPolicy:
    """The random baseline of the block-breaking diagnostic: persistent
    random choices for the pitch and attack factors, every other factor at
    its no-op value. A pitch choice is held for 10 steps on average, and the
    attack toggle starts with probability 0.1 per step and is held for 20
    steps on average (the geometric process of the exploration policy).
    Persistence matters because mining progress resets when the crosshair
    leaves the block (docs/stage4-block-breaking.md). Action exploration,
    not a mining routine: no choice depends on the observation."""

    ATTACK_START, ATTACK_HOLD, PITCH_HOLD = 0.1, 20.0, 10.0

    def __init__(self, seed: int) -> None:
        self.generator = torch.Generator().manual_seed(seed)
        self.attacking = False
        self.pitch = ActionSpace.NOOP[PITCH_FACTOR]

    def reset(self, observation: PolicyObservation) -> None:
        self.attacking = False
        self.pitch = ActionSpace.NOOP[PITCH_FACTOR]

    def act(self, observation: PolicyObservation) -> tuple[Any, list[int]]:
        draw = float(torch.rand((), generator=self.generator))
        if self.attacking:
            self.attacking = draw >= 1.0 / self.ATTACK_HOLD
        else:
            self.attacking = draw < self.ATTACK_START
        if float(torch.rand((), generator=self.generator)) < 1.0 / self.PITCH_HOLD:
            self.pitch = int(
                torch.randint(
                    ActionSpace.SIZES[PITCH_FACTOR], (), generator=self.generator
                )
            )
        choice = list(ActionSpace.NOOP)
        choice[PITCH_FACTOR] = self.pitch
        choice[ATTACK_FACTOR] = int(self.attacking)
        return ActionSpace.to_player_action(choice), choice

    def observe(self, choice: list[int], observation: PolicyObservation) -> None:
        pass


def run_episode(
    client, args, episode: int, policy, recorder, registries, directory: Path, gate
) -> dict:
    observation, reset = client.reset(args.world_seed, "normal", fixed_spawn=True)
    if gate is not None:
        # The world is frozen after RESET, so no tick passes while the gate
        # waits. Read the first observation again once the chunks are ready.
        gate.wait()
        observation = client.observe()
    reward = EpisodeReward(REWARD_V1, registries)
    reward.reset(observation)
    writer = EpisodeWriter(observation)
    probe = client.privileged()

    def record_position() -> None:
        server = probe.player()["server"]
        writer.add_privileged("position", [server["x"], server["z"]])
        writer.add_privileged("yaw", server["yaw"])

    record_position()
    if isinstance(policy, (ActorPolicy, RandomPitchPolicy, RandomBreakPolicy)):
        policy.reset(observation)
    else:
        policy.reset()
    if recorder is not None:
        recorder.episode = episode
        recorder.start()
    started = time.perf_counter()
    terminated = False
    choices: Counter[str] = Counter()
    steps_alive = 0
    broken_types: list[int] = []
    while writer.steps < args.max_steps and not terminated:
        if isinstance(policy, (ActorPolicy, RandomPitchPolicy, RandomBreakPolicy)):
            action, choice = policy.act(observation)
            for name, value in zip(ActionSpace.NAMES, choice, strict=True):
                choices[f"{name}={value}"] += 1
        else:
            action = policy.act(pitch=observation.pitch)
        if gate is not None:
            gate.wait()
        result = client.step(action)
        if result.info.tick_after != result.info.tick_before + 1:
            raise RuntimeError("a step did not advance exactly one tick")
        observation = result.observation
        terminated = result.terminated
        step_reward = reward.step(
            observation,
            terminated,
            result.events.blocks_broken,
            result.events.advancements,
        )
        if not math.isfinite(step_reward.total):
            raise RuntimeError(f"non-finite reward {step_reward.components}")
        if isinstance(policy, (ActorPolicy, RandomPitchPolicy, RandomBreakPolicy)):
            policy.observe(choice, observation)
        writer.add(
            action.executed(),
            observation,
            terminated,
            result.info.step_id,
            result.info.tick_before,
            result.info.tick_after,
            result.info.client_tick,
            reward=step_reward.total,
            components=[step_reward.components[name] for name in COMPONENTS],
            events=step_reward.events,
        )
        record_position()
        writer.add_privileged_step("breaks", len(result.events.blocks_broken))
        broken_types.extend(result.events.blocks_broken)
        steps_alive += 1
    wall = time.perf_counter() - started
    video = None
    if recorder is not None:
        capture, encoded = recorder.stop()
        video = {
            "status": encoded.status,
            "bytes": encoded.video_bytes,
            "directory": str(encoded.directory),
        }
    summary = reward.summary()
    meta = EpisodeMeta(
        episode=episode,
        world_seed=args.world_seed,
        preset="normal",
        level_id=reset.level_id,
        split=args.split,
        policy_seed=args.policy_seed + episode,
        steps=writer.steps,
        terminated=terminated,
        truncated=not terminated,
        first_tick=reset.game_time,
        last_tick=reset.game_time + writer.steps,
        action_schema=ACTION_SCHEMA,
        reward_version=REWARD_V1.version,
        policy=args.policy,
        episode_return=summary["total"],
    )
    writer.write(directory, meta)
    arrays = writer.arrays()
    control = episode_task(arrays)
    stats = {
        "pitch": pitch_metrics(arrays["obs_pitch"]),
        "break": break_metrics(
            arrays[BREAKS_KEY],
            arrays["action_buttons"][:, ACTION_BUTTONS.index("attack")],
            broken_types,
        ),
        "control": {k: v for k, v in control.items() if k not in ("target", "rewards")},
        "episode": episode,
        "steps": writer.steps,
        "died": terminated,
        "return": summary["total"],
        "components": summary["components"],
        "counts": summary["counts"],
        "steps_per_second": writer.steps / wall if wall else 0.0,
        "video": video,
        "choices": dict(choices),
    }
    write_json(directory / "collection.json", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--first-episode", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--world-seed", type=int, default=100_000)
    parser.add_argument(
        "--policy",
        choices=(
            "random",
            "actor",
            "random-pitch",
            "pitch-actor",
            "random-break",
            "break-actor",
        ),
        default="random",
        help="random-pitch and pitch-actor: the pitch diagnostic, only the "
        "pitch factor is chosen (docs/stage4-pitch-control.md)",
    )
    parser.add_argument("--policy-seed", type=int, default=5_000_000)
    parser.add_argument("--agent", type=Path)
    parser.add_argument(
        "--world-model",
        type=Path,
        default=Path("runs/stage3/tradeoff-P128/seed0/world_model.pt"),
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--time-budget", type=float, default=480.0)
    parser.add_argument("--port", type=int, default=47140)
    parser.add_argument(
        "--background-game",
        action="store_true",
        help="macOS: run the game on the efficiency cores behind a chunk gate",
    )
    args = parser.parse_args()
    macos_qos.run_in_background()
    runtime.configure(0)
    root = args.dataset / args.split / "episodes"
    todo = [
        e
        for e in range(args.first_episode, args.first_episode + args.episodes)
        if not (root / f"{e:06d}" / "episode.json").exists()
    ]
    if not todo:
        print("all episodes exist")
        return
    write_json(
        args.dataset / "manifest.json",
        {
            "format": "minecraft-episodes-v2",
            "git_commit": git_commit(),
            "world_seed": args.world_seed,
            "fixed_spawn": True,
            "max_steps": args.max_steps,
            "action_schema": ACTION_SCHEMA,
            "reward": REWARD_V1.to_json(),
            "exploration_policy": ExplorationConfig().to_json(),
            "recording": None if args.no_video else RECORDING,
        },
    )
    started = time.monotonic()
    log = args.dataset / "logs" / f"{args.split}-{int(time.time())}.minecraft.log"
    with launched_client(args.port, args.world_seed, log, 240.0) as client:
        write_json(args.dataset / "schema.json", client.request("SCHEMA"))
        registries = Registries.from_json(client.privileged().registry())
        client.set_pacing("unpaced", render_frames=True)
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
                args.dataset / "video" / args.split,
                WORKER,
                RECORDING,
                {
                    "dataset": str(args.dataset),
                    "split": args.split,
                    "policy": args.policy,
                },
            )
        try:
            for episode in todo:
                if time.monotonic() - started > args.time_budget:
                    break
                if args.policy in ("actor", "pitch-actor", "break-actor"):
                    policy = ActorPolicy(
                        args.agent,
                        args.world_model,
                        args.policy_seed + episode,
                        agent_class={
                            "actor": Agent,
                            "pitch-actor": PitchOnlyAgent,
                            "break-actor": BreakAgent,
                        }[args.policy],
                    )
                elif args.policy == "random-break":
                    policy = RandomBreakPolicy(args.policy_seed + episode)
                elif args.policy == "random-pitch":
                    policy = RandomPitchPolicy(args.policy_seed + episode)
                else:
                    policy = ExplorationPolicy(
                        ExplorationConfig(), args.policy_seed + episode
                    )
                stats = run_episode(
                    client,
                    args,
                    episode,
                    policy,
                    recorder,
                    registries,
                    root / f"{episode:06d}",
                    gate,
                )
                print(
                    json.dumps(
                        {
                            k: stats[k]
                            for k in (
                                "episode",
                                "steps",
                                "died",
                                "return",
                                "counts",
                                "control",
                                "pitch",
                                "break",
                            )
                        }
                    ),
                    flush=True,
                )
        finally:
            if recorder is not None:
                recorder.close()
            energy_after = macos_energy.read(pid) if pid else None
            usage = {
                "background_game": gate is not None,
                "gate": gate.summary() if gate is not None else None,
                "wall_seconds": time.monotonic() - started,
            }
            if energy_before is not None and energy_after is not None:
                spent = energy_after - energy_before
                usage["game_energy_joules"] = spent.energy_joules
                usage["game_performance_core_joules"] = (
                    spent.performance_core_energy_joules
                )
            write_json(args.dataset / "logs" / f"{args.split}-usage.json", usage)
            print(json.dumps(usage), flush=True)


if __name__ == "__main__":
    main()
