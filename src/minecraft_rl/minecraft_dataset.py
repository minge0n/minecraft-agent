"""Store Minecraft experience as contiguous episodes for sequence replay.

An episode directory holds one `episode.npz` with arrays over its T + 1
observations and T actions, and an `episode.json` with its metadata:

- Observation arrays have a leading time axis of length T + 1. Index 0 is
  the observation after reset, index t + 1 the observation after action t.
- Action arrays have length T. `terminated[t]` is true when the step of
  action t ended the episode (death). `step_id[t]`, `tick_before[t]`,
  `tick_after[t]` and `client_tick[t]` come from the environment for that
  step. The video frames of the episode (`frames.jsonl`) use the same client
  tick, so a frame maps to its step.
- Format v1 has no reward. Format v2 adds `reward[t]` (the scalar of the
  step of action t), `reward_components[t]` in the order of
  `minecraft_reward.COMPONENTS`, and the privileged events in
  `events.jsonl` (one line per step). Both formats load; a v1 episode has no
  reward arrays.

Arrays use small dtypes: categorical ids as int16 or int32, distances as
float16 in blocks, counts as uint8. The format is versioned with
`DATASET_FORMAT`. Episodes are never shuffled into single transitions: the
replay samples contiguous windows inside one episode.
"""

import json
import os
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy

from minecraft_rl.minecraft_interface import (
    OBSERVATION_SCHEMA,
    PlayerAction,
    PolicyObservation,
)

DATASET_FORMAT = "minecraft-episodes-v2"
READABLE_FORMATS = ("minecraft-episodes-v1", DATASET_FORMAT)

ACTION_BUTTONS = (
    "forward",
    "back",
    "left",
    "right",
    "jump",
    "sneak",
    "sprint",
    "attack",
    "use",
)

# Observation field -> (dtype, value or tuple). Scalars become shape (T + 1,).
OBSERVATION_DTYPES: dict[str, Any] = {
    "ray_kind": numpy.int8,
    "ray_type": numpy.int16,
    "ray_distance": numpy.float16,
    "health": numpy.float32,
    "max_health": numpy.float32,
    "absorption": numpy.float32,
    "food": numpy.int8,
    "air_bubbles": numpy.int8,
    "armor": numpy.int8,
    "xp_level": numpy.int16,
    "xp_progress": numpy.float32,
    "selected_slot": numpy.int8,
    "inventory_item": numpy.int16,
    "inventory_count": numpy.uint8,
    "inventory_durability": numpy.float16,
    "armor_item": numpy.int16,
    "armor_durability": numpy.float16,
    "offhand_item": numpy.int16,
    "offhand_count": numpy.uint8,
    "offhand_durability": numpy.float16,
    "effect_type": numpy.int16,
    "effect_amplifier": numpy.int16,
    "effect_seconds": numpy.int32,
    "pitch": numpy.float32,
}


@dataclass(frozen=True)
class EpisodeMeta:
    episode: int
    world_seed: int
    preset: str
    level_id: str
    split: str
    policy_seed: int
    steps: int
    terminated: bool
    truncated: bool
    first_tick: int
    last_tick: int
    observation_schema: str = OBSERVATION_SCHEMA
    format: str = DATASET_FORMAT
    action_schema: str | None = None
    reward_version: str | None = None
    policy: str | None = None
    episode_return: float | None = None
    # One Minecraft life can be stored as consecutive segments
    # (docs/stage4-temporal-attack.md). `episode` is then the life,
    # `segment` its index and `first_step` the life step of its first
    # transition. `terminated` and `truncated` describe the life end and
    # are false on every segment but the last. `stop_reason` is "death",
    # or the external stop of a truncated life.
    segment: int | None = None
    first_step: int | None = None
    final_segment: bool | None = None
    stop_reason: str | None = None


class EpisodeWriter:
    """Collects one episode in memory and writes it atomically."""

    def __init__(self, first: PolicyObservation) -> None:
        self.observations: dict[str, list] = {name: [] for name in OBSERVATION_DTYPES}
        self.buttons: list[list[bool]] = []
        self.camera: list[list[float]] = []
        self.hotbar: list[int] = []
        self.terminated: list[bool] = []
        self.step_id: list[int] = []
        self.tick_before: list[int] = []
        self.tick_after: list[int] = []
        self.client_tick: list[int] = []
        self.reward: list[float] = []
        self.reward_components: list[list[float]] = []
        self.events: list[dict[str, Any]] = []
        # Privileged per-observation values for reward calculation and
        # evaluation only (`privileged_*` arrays, never model input).
        self.privileged: dict[str, list] = {}
        self.privileged_steps: dict[str, list] = {}
        self._add_observation(first)

    def _add_observation(self, observation: PolicyObservation) -> None:
        for name in OBSERVATION_DTYPES:
            self.observations[name].append(getattr(observation, name))

    def add(
        self,
        action: PlayerAction,
        observation: PolicyObservation,
        terminated: bool,
        step_id: int,
        tick_before: int,
        tick_after: int,
        client_tick: int,
        reward: float | None = None,
        components: list[float] | None = None,
        events: dict[str, Any] | None = None,
    ) -> None:
        if reward is not None:
            self.reward.append(reward)
            self.reward_components.append(list(components or []))
            self.events.append(events or {})
        self.buttons.append([getattr(action, name) for name in ACTION_BUTTONS])
        self.camera.append([action.yaw_delta, action.pitch_delta])
        self.hotbar.append(action.hotbar)
        self.terminated.append(terminated)
        self.step_id.append(step_id)
        self.tick_before.append(tick_before)
        self.tick_after.append(tick_after)
        self.client_tick.append(client_tick)
        self._add_observation(observation)

    def add_privileged(self, name: str, value: Any) -> None:
        """One value per observation, the first one included."""
        self.privileged.setdefault(name, []).append(value)

    def add_privileged_step(self, name: str, value: Any) -> None:
        """One value per step (transition), not per observation."""
        self.privileged_steps.setdefault(name, []).append(value)

    @property
    def steps(self) -> int:
        return len(self.terminated)

    def arrays(self) -> dict[str, numpy.ndarray]:
        out = {
            f"obs_{name}": numpy.asarray(values, dtype=OBSERVATION_DTYPES[name])
            for name, values in self.observations.items()
        }
        out |= {
            "action_buttons": numpy.asarray(self.buttons, dtype=bool).reshape(
                -1, len(ACTION_BUTTONS)
            ),
            "action_camera": numpy.asarray(self.camera, dtype=numpy.float32).reshape(
                -1, 2
            ),
            "action_hotbar": numpy.asarray(self.hotbar, dtype=numpy.int8),
            "terminated": numpy.asarray(self.terminated, dtype=bool),
            "step_id": numpy.asarray(self.step_id, dtype=numpy.int64),
            "tick_before": numpy.asarray(self.tick_before, dtype=numpy.int64),
            "tick_after": numpy.asarray(self.tick_after, dtype=numpy.int64),
            "client_tick": numpy.asarray(self.client_tick, dtype=numpy.int64),
        }
        if self.reward:
            if len(self.reward) != len(self.terminated):
                raise ValueError("every step needs a reward once one step has one")
            out["reward"] = numpy.asarray(self.reward, dtype=numpy.float32)
            out["reward_components"] = numpy.asarray(
                self.reward_components, dtype=numpy.float32
            )
        for name, values in self.privileged.items():
            if len(values) != self.steps + 1:
                raise ValueError(f"privileged {name} needs one value per observation")
            out[f"privileged_{name}"] = numpy.asarray(values, dtype=numpy.float64)
        for name, values in self.privileged_steps.items():
            if len(values) != self.steps:
                raise ValueError(f"privileged {name} needs one value per step")
            out[f"privileged_{name}"] = numpy.asarray(values, dtype=numpy.int64)
        return out

    def write(self, directory: Path, meta: EpisodeMeta) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        partial = directory / "episode.npz.partial"
        with partial.open("wb") as file:
            numpy.savez_compressed(file, **self.arrays())
        os.replace(partial, directory / "episode.npz")
        if self.events:
            lines = "".join(json.dumps(e, sort_keys=True) + "\n" for e in self.events)
            (directory / "events.jsonl").write_text(lines, encoding="utf-8")
        write_json(directory / "episode.json", asdict(meta))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


@dataclass(frozen=True)
class Episode:
    meta: EpisodeMeta
    arrays: dict[str, numpy.ndarray]

    @property
    def steps(self) -> int:
        return self.meta.steps


def episode_directories(root: Path) -> list[Path]:
    """Complete episodes only: a directory counts once its metadata exists."""
    return sorted(
        path.parent
        for path in root.glob("episodes/*/episode.json")
        if (path.parent / "episode.npz").exists()
    )


def load_episode(directory: Path) -> Episode:
    raw = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
    if raw.get("format") not in READABLE_FORMATS:
        raise ValueError(f"{directory} is not a minecraft-episodes episode")
    names = {field.name for field in fields(EpisodeMeta)}
    meta = EpisodeMeta(**{key: value for key, value in raw.items() if key in names})
    with numpy.load(directory / "episode.npz") as data:
        arrays = {name: data[name] for name in data.files}
    if arrays["terminated"].shape[0] != meta.steps:
        raise ValueError(f"{directory}: step count differs from metadata")
    if arrays["obs_ray_kind"].shape[0] != meta.steps + 1:
        raise ValueError(f"{directory}: observation count is not steps + 1")
    return Episode(meta, arrays)


def load_episodes(root: Path, split: str | None = None) -> list[Episode]:
    episodes = [load_episode(path) for path in episode_directories(root)]
    return [e for e in episodes if split is None or e.meta.split == split]


def join_segments(segments: list[Episode]) -> list[Episode]:
    """One Episode per life from its stored segments, in step order. A
    segment's first observation is the previous segment's last, so it is
    kept once. Episodes without a segment index pass through."""
    lives: dict[int, list[Episode]] = {}
    out = []
    for episode in segments:
        if episode.meta.segment is None:
            out.append(episode)
        else:
            lives.setdefault(episode.meta.episode, []).append(episode)
    for life in lives.values():
        life.sort(key=lambda e: e.meta.segment)
        expected = 0
        for e in life:
            if e.meta.first_step != expected:
                raise ValueError(f"life {e.meta.episode}: segment gap at {expected}")
            expected += e.steps
        stored = [life[0]] + [e for e in life[1:] if e.steps]
        arrays = {}
        for name in life[0].arrays:
            per_observation = life[0].arrays[name].shape[0] == life[0].steps + 1
            parts = [life[0].arrays[name]] + [
                e.arrays[name][1:] if per_observation else e.arrays[name]
                for e in stored[1:]
            ]
            arrays[name] = numpy.concatenate(parts)
        last = life[-1].meta
        meta = replace(
            last,
            steps=expected,
            first_tick=life[0].meta.first_tick,
            segment=None,
            first_step=0,
            final_segment=None,
            episode_return=sum(e.meta.episode_return or 0.0 for e in life),
        )
        out.append(Episode(meta, arrays))
    return out
