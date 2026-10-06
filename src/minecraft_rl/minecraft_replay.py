"""Sequence replay of Minecraft episodes for world-model training.

The replay keeps whole episodes and samples contiguous windows of L
transitions inside one episode: L + 1 observations, L actions and L
continuation flags. A window never crosses an episode boundary, and
transitions are never shuffled one by one.

The observation registries are large (1,286 block types, 1,658 item
types), but a small dataset uses few of them. `CompactVocabulary` maps the
raw ids of each registry to a dense index over the ids that occur in the
training split, plus one shared index for any id that does not occur there
(`unknown`). The map is a data property, not Minecraft knowledge, and it
keeps the categorical output layers small. Held-out data reports how often
it hits the unknown index.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy
import torch

from minecraft_rl.minecraft_dataset import Episode
from minecraft_rl.minecraft_interface import (
    RAY_KIND_BLOCK,
    RAY_KIND_ENTITY,
    RAY_KIND_FLUID,
)
from minecraft_rl.minecraft_world_model import (
    Vocabulary,
    action_tensor,
    observation_tensors,
)

FAMILIES = ("block", "fluid", "entity", "item", "effect")


@dataclass(frozen=True)
class CompactVocabulary:
    """For each family, the sorted raw ids seen in training. Compact index i
    is `ids[i]`; index `len(ids)` is unknown."""

    ids: dict[str, list[int]]
    raw_sizes: dict[str, int]

    @classmethod
    def from_episodes(
        cls, episodes: list[Episode], raw_sizes: dict[str, int]
    ) -> "CompactVocabulary":
        seen: dict[str, set[int]] = {family: set() for family in FAMILIES}
        for episode in episodes:
            a = episode.arrays
            kind, kind_type = a["obs_ray_kind"], a["obs_ray_type"]
            for family, value in (
                ("block", RAY_KIND_BLOCK),
                ("fluid", RAY_KIND_FLUID),
                ("entity", RAY_KIND_ENTITY),
            ):
                seen[family] |= set(numpy.unique(kind_type[kind == value]).tolist())
            for name in ("obs_inventory_item", "obs_armor_item", "obs_offhand_item"):
                seen["item"] |= set(numpy.unique(a[name]).tolist())
            seen["effect"] |= set(numpy.unique(a["obs_effect_type"]).tolist())
        # Raw index 0 means an empty slot. It is always kept and always maps
        # to compact index 0, so "index > 0" still means a filled slot.
        seen["item"].add(0)
        seen["effect"].add(0)
        return cls({f: sorted(int(x) for x in seen[f]) for f in FAMILIES}, raw_sizes)

    def size(self, family: str) -> int:
        return len(self.ids[family]) + 1

    def unknown(self, family: str) -> int:
        return len(self.ids[family])

    def table(self, family: str) -> numpy.ndarray:
        out = numpy.full(
            self.raw_sizes[family], self.unknown(family), dtype=numpy.int64
        )
        out[self.ids[family]] = numpy.arange(len(self.ids[family]))
        return out

    def model_vocabulary(
        self, rows: int, columns: int, max_distance: float
    ) -> Vocabulary:
        return Vocabulary(
            rows=rows,
            columns=columns,
            max_distance=max_distance,
            block_types=self.size("block"),
            fluid_types=self.size("fluid"),
            entity_types=self.size("entity"),
            item_types=self.size("item"),
            effect_types=self.size("effect"),
        )

    def identifier(self) -> str:
        """A short hash of the id lists, for run metadata."""
        text = json.dumps(self.ids, sort_keys=True).encode()
        return hashlib.sha256(text).hexdigest()[:16]

    def to_json(self) -> dict[str, Any]:
        return {"ids": self.ids, "raw_sizes": self.raw_sizes}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "CompactVocabulary":
        return cls({f: list(data["ids"][f]) for f in FAMILIES}, dict(data["raw_sizes"]))


def raw_sizes(schema: dict[str, Any]) -> dict[str, int]:
    return {
        "block": schema["block_types"],
        "fluid": schema["fluid_types"],
        "entity": schema["entity_types"],
        "item": schema["item_types"],
        "effect": schema["effect_types"] + 1,
    }


def load_schema(dataset: Path) -> dict[str, Any]:
    return json.loads((dataset / "schema.json").read_text(encoding="utf-8"))


def compact_arrays(
    arrays: dict[str, numpy.ndarray], vocabulary: CompactVocabulary
) -> dict[str, numpy.ndarray]:
    """A copy of episode arrays with every categorical id in compact form."""
    out = dict(arrays)
    kind, kind_type = arrays["obs_ray_kind"], arrays["obs_ray_type"].astype(numpy.int64)
    compact = numpy.zeros_like(kind_type)
    for family, value in (
        ("block", RAY_KIND_BLOCK),
        ("fluid", RAY_KIND_FLUID),
        ("entity", RAY_KIND_ENTITY),
    ):
        mask = kind == value
        compact[mask] = vocabulary.table(family)[kind_type[mask]]
    out["obs_ray_type"] = compact.astype(numpy.int16)
    items = vocabulary.table("item")
    for name in ("obs_inventory_item", "obs_armor_item", "obs_offhand_item"):
        out[name] = items[arrays[name].astype(numpy.int64)].astype(numpy.int16)
    out["obs_effect_type"] = vocabulary.table("effect")[
        arrays["obs_effect_type"].astype(numpy.int64)
    ].astype(numpy.int16)
    return out


def unknown_fractions(
    episodes: list[Episode], vocabulary: CompactVocabulary
) -> dict[str, float]:
    """The share of categorical values that map to unknown, per family."""
    counts = {family: [0, 0] for family in FAMILIES}
    for episode in episodes:
        a = compact_arrays(episode.arrays, vocabulary)
        kind = a["obs_ray_kind"]
        for family, value in (
            ("block", RAY_KIND_BLOCK),
            ("fluid", RAY_KIND_FLUID),
            ("entity", RAY_KIND_ENTITY),
        ):
            values = a["obs_ray_type"][kind == value]
            counts[family][0] += int((values == vocabulary.unknown(family)).sum())
            counts[family][1] += values.size
        items = numpy.concatenate(
            [
                a["obs_inventory_item"].ravel(),
                a["obs_armor_item"].ravel(),
                a["obs_offhand_item"],
            ]
        )
        counts["item"][0] += int((items == vocabulary.unknown("item")).sum())
        counts["item"][1] += items.size
        effects = a["obs_effect_type"].ravel()
        counts["effect"][0] += int((effects == vocabulary.unknown("effect")).sum())
        counts["effect"][1] += effects.size
    return {f: (u / n if n else 0.0) for f, (u, n) in counts.items()}


@dataclass
class Batch:
    observations: dict[str, torch.Tensor]
    actions: torch.Tensor
    continues: torch.Tensor
    episodes: torch.Tensor
    starts: torch.Tensor


class SequenceReplay:
    """Episodes as tensors, sampled as windows of `length` transitions."""

    def __init__(
        self, episodes: list[Episode], vocabulary: CompactVocabulary, length: int
    ) -> None:
        self.length = length
        self.episodes = [e for e in episodes if e.steps >= length]
        self.skipped_short = len(episodes) - len(self.episodes)
        if not self.episodes:
            raise ValueError(f"no episode has {length} steps")
        self.observations: list[dict[str, torch.Tensor]] = []
        self.actions: list[torch.Tensor] = []
        self.continues: list[torch.Tensor] = []
        for episode in self.episodes:
            a = compact_arrays(episode.arrays, vocabulary)
            self.observations.append(observation_tensors(a))
            self.actions.append(
                action_tensor(
                    torch.as_tensor(a["action_buttons"]),
                    torch.as_tensor(a["action_camera"]),
                    torch.as_tensor(a["action_hotbar"]),
                )
            )
            self.continues.append(1.0 - torch.as_tensor(a["terminated"]).float())
        starts = torch.tensor([e.steps - length + 1 for e in self.episodes])
        self.weights = starts.float() / starts.sum()
        self.starts_per_episode = starts

    @property
    def transitions(self) -> int:
        return sum(e.steps for e in self.episodes)

    def memory_bytes(self) -> int:
        tensors = [t for obs in self.observations for t in obs.values()]
        tensors += self.actions + self.continues
        return sum(t.element_size() * t.numel() for t in tensors)

    def window(
        self, episode: int, start: int
    ) -> tuple[dict, torch.Tensor, torch.Tensor]:
        end = start + self.length
        observations = {
            k: v[start : end + 1] for k, v in self.observations[episode].items()
        }
        return (
            observations,
            self.actions[episode][start:end],
            self.continues[episode][start:end],
        )

    def batch(self, index: list[tuple[int, int]]) -> Batch:
        windows = [self.window(e, s) for e, s in index]
        names = windows[0][0].keys()
        return Batch(
            {name: torch.stack([w[0][name] for w in windows]) for name in names},
            torch.stack([w[1] for w in windows]),
            torch.stack([w[2] for w in windows]),
            torch.tensor([e for e, _ in index]),
            torch.tensor([s for _, s in index]),
        )

    def sample(self, size: int, generator: torch.Generator) -> Batch:
        episodes = torch.multinomial(
            self.weights, size, replacement=True, generator=generator
        )
        offsets = torch.rand(size, generator=generator)
        starts = (offsets * self.starts_per_episode[episodes].float()).long()
        return self.batch(list(zip(episodes.tolist(), starts.tolist(), strict=True)))

    def evaluation_windows(self, stride: int) -> list[tuple[int, int]]:
        """Fixed, non-overlapping windows for deterministic evaluation."""
        return [
            (e, start)
            for e, count in enumerate(self.starts_per_episode.tolist())
            for start in range(0, count, stride)
        ]
