"""The Minecraft reward contract `reward-v1` (docs/decisions/reward.md).

The actor receives one scalar per step. Inside, the reward is the sum of named
components, and each step keeps them so that a run can show where its reward
came from. Components and events are training signals and diagnostics only;
they never become policy observations.

`EpisodeReward` is a deterministic function of an episode: the observations
(block types seen in the ray grid, item types held) and the privileged step
events (block types broken, advancements completed). It holds the
episode-local novelty memory, so `reset()` starts a fresh episode.

Every constant of the contract is in `REWARD_V1`. A change of meaning needs a
new version.
"""

import fnmatch
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from minecraft_rl.minecraft_interface import RAY_KIND_BLOCK, PolicyObservation

REWARD_VERSION = "reward-v1"

COMPONENTS = (
    "block_seen_novelty",
    "block_broken_novelty",
    "item_acquired_novelty",
    "importance",
    "advancement",
    "progression",
    "survival",
    "terminal",
)

# Human Minecraft knowledge: which items count as progress and how much.
# Shell-style patterns on registry names; an item takes the highest weight.
IMPORTANCE_V1: tuple[tuple[str, float], ...] = (
    ("minecraft:*_log", 0.2),
    ("minecraft:*_stem", 0.2),
    ("minecraft:*_planks", 0.1),
    ("minecraft:stick", 0.1),
    ("minecraft:crafting_table", 0.3),
    ("minecraft:wooden_pickaxe", 0.5),
    ("minecraft:cobblestone", 0.3),
    ("minecraft:coal", 0.3),
    ("minecraft:stone_pickaxe", 0.7),
    ("minecraft:furnace", 0.7),
    ("minecraft:raw_iron", 1.0),
    ("minecraft:bucket", 1.0),
    ("minecraft:iron_ingot", 1.5),
    ("minecraft:flint_and_steel", 1.5),
    ("minecraft:iron_pickaxe", 2.0),
    ("minecraft:obsidian", 2.0),
    ("minecraft:blaze_powder", 2.0),
    ("minecraft:diamond", 3.0),
    ("minecraft:diamond_pickaxe", 3.0),
    ("minecraft:blaze_rod", 5.0),
    ("minecraft:ender_pearl", 5.0),
    ("minecraft:ender_eye", 8.0),
)

# Major progression milestones toward the End, by vanilla advancement id.
PROGRESSION_V1: tuple[tuple[str, float], ...] = (
    ("minecraft:story/enter_the_nether", 10.0),
    ("minecraft:nether/obtain_blaze_rod", 10.0),
    ("minecraft:story/follow_ender_eye", 15.0),
    ("minecraft:story/enter_the_end", 30.0),
)

DRAGON_KILL = "minecraft:end/kill_dragon"


@dataclass(frozen=True)
class RewardSpec:
    version: str
    block_seen: float
    block_broken: float
    item_acquired: float
    advancement: float
    death: float
    terminal: float
    terminal_advancement: str
    importance: tuple[tuple[str, float], ...]
    progression: tuple[tuple[str, float], ...]

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["importance"] = dict(self.importance)
        out["progression"] = dict(self.progression)
        return out

    def importance_of(self, item_name: str) -> float:
        weights = [w for p, w in self.importance if fnmatch.fnmatchcase(item_name, p)]
        return max(weights, default=0.0)


REWARD_V1 = RewardSpec(
    version=REWARD_VERSION,
    block_seen=0.01,
    block_broken=0.05,
    item_acquired=0.1,
    advancement=1.0,
    death=-1.0,
    terminal=1000.0,
    terminal_advancement=DRAGON_KILL,
    importance=IMPORTANCE_V1,
    progression=PROGRESSION_V1,
)


@dataclass(frozen=True)
class Registries:
    """Registry names by index, from the environment (`DEBUG_REGISTRY`). Index 0
    of the item registry is air, the empty slot."""

    block: tuple[str, ...]
    item: tuple[str, ...]

    @classmethod
    def from_json(cls, payload: dict[str, Sequence[str]]) -> "Registries":
        return cls(tuple(payload["block"]), tuple(payload["item"]))


@dataclass
class StepReward:
    """The reward of one step: the scalar, its components and the events that
    earned them."""

    components: dict[str, float]
    events: dict[str, list[Any]] = field(default_factory=dict)

    @property
    def total(self) -> float:
        # A fixed order, so the same components always give the same float.
        return math.fsum(self.components[name] for name in COMPONENTS)


def held_items(observation: PolicyObservation) -> set[int]:
    """Item types in the inventory, armor and offhand; 0 is empty."""
    items = set(observation.inventory_item) | set(observation.armor_item)
    items.add(observation.offhand_item)
    items.discard(0)
    return items


class EpisodeReward:
    """Reward of one episode under a `RewardSpec`. Call `reset` with the first
    observation, then `step` once per environment step."""

    def __init__(self, spec: RewardSpec, registries: Registries) -> None:
        self.spec = spec
        self.registries = registries
        self._item_importance = [spec.importance_of(name) for name in registries.item]
        self._progression = dict(spec.progression)
        self.reset_memory()

    def reset_memory(self) -> None:
        self.seen_blocks: set[int] = set()
        self.broken_blocks: set[int] = set()
        self.acquired_items: set[int] = set()
        self.advancements: set[str] = set()
        self.died = False
        self._pending_seen: frozenset[int] = frozenset()
        self.totals: Counter[str] = Counter()
        self.counts: Counter[str] = Counter()

    def reset(self, observation: PolicyObservation) -> None:
        """Start an episode. Items held at the start earn nothing. Block types
        visible at the start earn seen novelty on the first step."""
        self.reset_memory()
        self.acquired_items |= held_items(observation)
        self._pending_seen = observation.visible_types(RAY_KIND_BLOCK)

    def step(
        self,
        observation: PolicyObservation,
        terminated: bool,
        blocks_broken: Iterable[int] = (),
        advancements: Iterable[str] = (),
    ) -> StepReward:
        spec = self.spec
        components = dict.fromkeys(COMPONENTS, 0.0)
        events: dict[str, list[Any]] = {}

        seen = self._pending_seen | observation.visible_types(RAY_KIND_BLOCK)
        self._pending_seen = frozenset()
        new_seen = sorted(seen - self.seen_blocks)
        self.seen_blocks |= set(new_seen)
        components["block_seen_novelty"] = spec.block_seen * len(new_seen)
        if new_seen:
            events["blocks_seen"] = new_seen

        new_broken = []
        for block in blocks_broken:
            if block not in self.broken_blocks:
                self.broken_blocks.add(block)
                new_broken.append(block)
        components["block_broken_novelty"] = spec.block_broken * len(new_broken)
        if new_broken:
            events["blocks_broken"] = new_broken

        new_items = sorted(held_items(observation) - self.acquired_items)
        self.acquired_items |= set(new_items)
        components["item_acquired_novelty"] = spec.item_acquired * len(new_items)
        important = [item for item in new_items if self._item_importance[item] > 0]
        components["importance"] = math.fsum(
            self._item_importance[item] for item in important
        )
        if new_items:
            events["items_acquired"] = new_items
        if important:
            events["important_items"] = important

        new_advancements = []
        for advancement in advancements:
            if advancement not in self.advancements:
                self.advancements.add(advancement)
                new_advancements.append(advancement)
        components["advancement"] = spec.advancement * len(new_advancements)
        components["progression"] = math.fsum(
            self._progression.get(a, 0.0) for a in new_advancements
        )
        if spec.terminal_advancement in new_advancements:
            components["terminal"] = spec.terminal
        if new_advancements:
            events["advancements"] = new_advancements

        if terminated and not self.died:
            self.died = True
            components["survival"] = spec.death
            events["death"] = [True]

        for name, value in components.items():
            self.totals[name] += value
        self.counts["blocks_seen"] += len(new_seen)
        self.counts["blocks_broken"] += len(new_broken)
        self.counts["items_acquired"] += len(new_items)
        self.counts["important_items"] += len(important)
        self.counts["advancements"] += len(new_advancements)
        return StepReward(components, events)

    def summary(self) -> dict[str, Any]:
        return {
            "version": self.spec.version,
            "total": math.fsum(self.totals[name] for name in COMPONENTS),
            "components": {name: self.totals[name] for name in COMPONENTS},
            "counts": dict(self.counts),
        }


def shaping_upper_bound(
    spec: RewardSpec, registries: Registries, advancements: int
) -> float:
    """The largest positive reward that one episode can earn without the
    terminal reward: every block type seen and broken, every item type acquired
    with its importance, `advancements` displayed advancements and every
    milestone. The terminal reward must exceed it."""
    items = len(registries.item) - 1
    return math.fsum(
        [
            spec.block_seen * len(registries.block),
            spec.block_broken * len(registries.block),
            spec.item_acquired * items,
            math.fsum(spec.importance_of(name) for name in registries.item[1:]),
            spec.advancement * advancements,
            math.fsum(weight for _, weight in spec.progression),
        ]
    )
