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
- No reward array exists: the environment has no task reward yet.

Arrays use small dtypes: categorical ids as int16 or int32, distances as
float16 in blocks, counts as uint8. The format is versioned with
`DATASET_FORMAT`. Episodes are never shuffled into single transitions: the
replay samples contiguous windows inside one episode.
"""

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy

from minecraft_rl.minecraft_interface import (
    OBSERVATION_SCHEMA,
    PlayerAction,
    PolicyObservation,
)

DATASET_FORMAT = "minecraft-episodes-v1"

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
    ) -> None:
        self.buttons.append([getattr(action, name) for name in ACTION_BUTTONS])
        self.camera.append([action.yaw_delta, action.pitch_delta])
        self.hotbar.append(action.hotbar)
        self.terminated.append(terminated)
        self.step_id.append(step_id)
        self.tick_before.append(tick_before)
        self.tick_after.append(tick_after)
        self.client_tick.append(client_tick)
        self._add_observation(observation)

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
        return out

    def write(self, directory: Path, meta: EpisodeMeta) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        partial = directory / "episode.npz.partial"
        with partial.open("wb") as file:
            numpy.savez_compressed(file, **self.arrays())
        os.replace(partial, directory / "episode.npz")
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
    if raw.get("format") != DATASET_FORMAT:
        raise ValueError(f"{directory} is not a {DATASET_FORMAT} episode")
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
