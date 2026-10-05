import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Self

OBSERVATION_SCHEMA = "visible-field-v2"
RAY_KIND_NONE = 0
RAY_KIND_BLOCK = 1
RAY_KIND_FLUID = 2
RAY_KIND_ENTITY = 3
RAY_KINDS = 4
HOTBAR_SLOTS = 9
INVENTORY_SLOTS = 36
ARMOR_SLOTS = 4
EFFECT_SLOTS = 8
AIR_BUBBLES = 10
DURABILITY_STEPS = 13
INFINITE_EFFECT_SECONDS = -1
KEEP_HOTBAR_SLOT = -1
MAX_CAMERA_DELTA_DEGREES = 45.0

OBSERVATION_FIELDS = frozenset(
    {
        "schema",
        "ray_kind",
        "ray_type",
        "ray_distance",
        "health",
        "max_health",
        "absorption",
        "food",
        "air_bubbles",
        "armor",
        "xp_level",
        "xp_progress",
        "selected_slot",
        "inventory_item",
        "inventory_count",
        "inventory_durability",
        "armor_item",
        "armor_durability",
        "offhand_item",
        "offhand_count",
        "offhand_durability",
        "effect_type",
        "effect_amplifier",
        "effect_seconds",
        "pitch",
    }
)

SCHEMA_LAYOUT = {
    "ray_kinds": RAY_KINDS,
    "hotbar_slots": HOTBAR_SLOTS,
    "inventory_slots": INVENTORY_SLOTS,
    "armor_slots": ARMOR_SLOTS,
    "effect_slots": EFFECT_SLOTS,
    "air_bubbles": AIR_BUBBLES,
    "durability_steps": DURABILITY_STEPS,
}


class SchemaError(ValueError):
    pass


@dataclass(frozen=True)
class ObservationSchema:
    rows: int
    columns: int
    vertical_fov_degrees: float
    horizontal_fov_degrees: float
    max_distance: float
    block_types: int
    fluid_types: int
    entity_types: int
    item_types: int
    effect_types: int

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Self:
        if payload.get("observation_schema") != OBSERVATION_SCHEMA:
            raise SchemaError(f"unsupported schema {payload.get('observation_schema')}")
        for name, expected in SCHEMA_LAYOUT.items():
            if payload.get(name) != expected:
                raise SchemaError(
                    f"{name} differs from this client: {payload.get(name)}"
                )
        if payload["max_camera_delta_degrees"] != MAX_CAMERA_DELTA_DEGREES:
            raise SchemaError("camera delta bound differs from this client")
        return cls(
            rows=payload["rows"],
            columns=payload["columns"],
            vertical_fov_degrees=payload["vertical_fov_degrees"],
            horizontal_fov_degrees=payload["horizontal_fov_degrees"],
            max_distance=payload["max_distance"],
            block_types=payload["block_types"],
            fluid_types=payload["fluid_types"],
            entity_types=payload["entity_types"],
            item_types=payload["item_types"],
            effect_types=payload["effect_types"],
        )

    @property
    def rays(self) -> int:
        return self.rows * self.columns

    def type_vocabulary(self, kind: int) -> int:
        return {
            RAY_KIND_NONE: 1,
            RAY_KIND_BLOCK: self.block_types,
            RAY_KIND_FLUID: self.fluid_types,
            RAY_KIND_ENTITY: self.entity_types,
        }[kind]


@dataclass(frozen=True)
class PolicyObservation:
    """One policy observation of schema `visible-field-v2`. Inventory slots 0-8
    are the hotbar. Every item, block, fluid, entity and effect id is a
    categorical registry index: encode it with an embedding, never as a number."""

    ray_kind: tuple[int, ...]
    ray_type: tuple[int, ...]
    ray_distance: tuple[float, ...]
    health: float
    max_health: float
    absorption: float
    food: int
    air_bubbles: int
    armor: int
    xp_level: int
    xp_progress: float
    selected_slot: int
    inventory_item: tuple[int, ...]
    inventory_count: tuple[int, ...]
    inventory_durability: tuple[float, ...]
    armor_item: tuple[int, ...]
    armor_durability: tuple[float, ...]
    offhand_item: int
    offhand_count: int
    offhand_durability: float
    effect_type: tuple[int, ...]
    effect_amplifier: tuple[int, ...]
    effect_seconds: tuple[int, ...]
    pitch: float

    @classmethod
    def from_json(cls, payload: dict[str, Any], schema: ObservationSchema) -> Self:
        if set(payload) != OBSERVATION_FIELDS:
            differing = sorted(set(payload) ^ OBSERVATION_FIELDS)
            raise SchemaError(f"observation fields differ: {differing}")
        if payload["schema"] != OBSERVATION_SCHEMA:
            raise SchemaError(f"unsupported observation schema {payload['schema']}")

        kinds = _integers(payload["ray_kind"], schema.rays, "ray_kind")
        types = _integers(payload["ray_type"], schema.rays, "ray_type")
        distances = _numbers(payload["ray_distance"], schema.rays, "ray_distance")
        for kind, type_id, distance in zip(kinds, types, distances, strict=True):
            if not 0 <= kind < RAY_KINDS:
                raise SchemaError(f"ray kind {kind} out of range")
            if not 0 <= type_id < schema.type_vocabulary(kind):
                raise SchemaError(f"ray type {type_id} out of range for kind {kind}")
            if not 0.0 <= distance <= schema.max_distance:
                raise SchemaError(f"ray distance {distance} out of range")
            if kind == RAY_KIND_NONE and distance != schema.max_distance:
                raise SchemaError("empty ray must report the maximum distance")

        items = _items(payload["inventory_item"], INVENTORY_SLOTS, schema, "inventory")
        counts = _counts(payload["inventory_count"], INVENTORY_SLOTS, "inventory")
        durability = _fractions(
            payload["inventory_durability"], INVENTORY_SLOTS, "inventory_durability"
        )
        armor_items = _items(payload["armor_item"], ARMOR_SLOTS, schema, "armor")
        armor_durability = _fractions(
            payload["armor_durability"], ARMOR_SLOTS, "armor_durability"
        )
        (offhand_item,) = _items([payload["offhand_item"]], 1, schema, "offhand")
        (offhand_count,) = _counts([payload["offhand_count"]], 1, "offhand")
        (offhand_durability,) = _fractions(
            [payload["offhand_durability"]], 1, "offhand_durability"
        )

        effect_types = _integers(payload["effect_type"], EFFECT_SLOTS, "effect_type")
        amplifiers = _integers(
            payload["effect_amplifier"], EFFECT_SLOTS, "effect_amplifier"
        )
        seconds = _integers(payload["effect_seconds"], EFFECT_SLOTS, "effect_seconds")
        for effect, amplifier, left in zip(
            effect_types, amplifiers, seconds, strict=True
        ):
            if not 0 <= effect <= schema.effect_types:
                raise SchemaError(f"effect type {effect} out of range")
            if effect == 0 and (amplifier, left) != (0, 0):
                raise SchemaError("an empty effect slot must report zeros")
            if effect and (amplifier < 0 or left < INFINITE_EFFECT_SECONDS):
                raise SchemaError(f"effect values out of range: {amplifier}, {left}")
        present = [effect for effect in effect_types if effect]
        if present != sorted(present) or effect_types[: len(present)] != tuple(present):
            raise SchemaError("effects must fill the first slots in registry order")

        selected_slot = payload["selected_slot"]
        if type(selected_slot) is not int or not 0 <= selected_slot < HOTBAR_SLOTS:
            raise SchemaError(f"selected slot {selected_slot} out of range")
        air = payload["air_bubbles"]
        if type(air) is not int or not 0 <= air <= AIR_BUBBLES:
            raise SchemaError(f"air bubbles {air} out of range")
        xp_level = payload["xp_level"]
        if type(xp_level) is not int or xp_level < 0:
            raise SchemaError(f"xp level {xp_level} out of range")
        (xp_progress,) = _fractions([payload["xp_progress"]], 1, "xp_progress")
        armor = payload["armor"]
        if type(armor) is not int or armor < 0:
            raise SchemaError(f"armor {armor} out of range")
        pitch = float(payload["pitch"])
        if not -90.0 <= pitch <= 90.0:
            raise SchemaError(f"pitch {pitch} out of range")

        return cls(
            ray_kind=kinds,
            ray_type=types,
            ray_distance=distances,
            health=float(payload["health"]),
            max_health=float(payload["max_health"]),
            absorption=float(payload["absorption"]),
            food=int(payload["food"]),
            air_bubbles=air,
            armor=armor,
            xp_level=xp_level,
            xp_progress=xp_progress,
            selected_slot=selected_slot,
            inventory_item=items,
            inventory_count=counts,
            inventory_durability=durability,
            armor_item=armor_items,
            armor_durability=armor_durability,
            offhand_item=offhand_item,
            offhand_count=offhand_count,
            offhand_durability=offhand_durability,
            effect_type=effect_types,
            effect_amplifier=amplifiers,
            effect_seconds=seconds,
            pitch=pitch,
        )

    @property
    def hotbar_item(self) -> tuple[int, ...]:
        return self.inventory_item[:HOTBAR_SLOTS]

    @property
    def hotbar_count(self) -> tuple[int, ...]:
        return self.inventory_count[:HOTBAR_SLOTS]

    @property
    def main_hand_item(self) -> int:
        return self.inventory_item[self.selected_slot]

    def visible_types(self, kind: int) -> frozenset[int]:
        return frozenset(
            type_id
            for ray_kind, type_id in zip(self.ray_kind, self.ray_type, strict=True)
            if ray_kind == kind
        )

    def digest(self) -> str:
        """Stable SHA-256 of every field, for exact replay comparison."""
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


def _integers(values: list[Any], length: int, name: str) -> tuple[int, ...]:
    if len(values) != length or any(type(value) is not int for value in values):
        raise SchemaError(f"{name} must hold {length} integers")
    return tuple(values)


def _items(
    values: list[Any], length: int, schema: ObservationSchema, name: str
) -> tuple[int, ...]:
    items = _integers(values, length, f"{name}_item")
    if any(not 0 <= item < schema.item_types for item in items):
        raise SchemaError(f"{name} item out of range")
    return items


def _counts(values: list[Any], length: int, name: str) -> tuple[int, ...]:
    counts = _integers(values, length, f"{name}_count")
    if any(count < 0 for count in counts):
        raise SchemaError(f"negative {name} count")
    return counts


def _fractions(values: list[Any], length: int, name: str) -> tuple[float, ...]:
    fractions = _numbers(values, length, name)
    if any(not 0.0 <= value <= 1.0 for value in fractions):
        raise SchemaError(f"{name} must lie in [0, 1]")
    return fractions


def _numbers(values: list[Any], length: int, name: str) -> tuple[float, ...]:
    if len(values) != length or any(
        type(value) not in (int, float) or not math.isfinite(value) for value in values
    ):
        raise SchemaError(f"{name} must hold {length} finite numbers")
    return tuple(float(value) for value in values)


@dataclass(frozen=True)
class PlayerAction:
    forward: bool = False
    back: bool = False
    left: bool = False
    right: bool = False
    jump: bool = False
    sneak: bool = False
    sprint: bool = False
    attack: bool = False
    use: bool = False
    yaw_delta: float = 0.0
    pitch_delta: float = 0.0
    hotbar: int = KEEP_HOTBAR_SLOT

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name in ("yaw_delta", "pitch_delta"):
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError(f"{name} must be a finite number")
                if abs(value) > MAX_CAMERA_DELTA_DEGREES:
                    raise ValueError(
                        f"{name} exceeds {MAX_CAMERA_DELTA_DEGREES} degrees"
                    )
            elif name == "hotbar":
                if (
                    type(value) is not int
                    or not KEEP_HOTBAR_SLOT <= value < HOTBAR_SLOTS
                ):
                    raise ValueError("hotbar must be -1 (keep) or a slot 0..8")
            elif type(value) is not bool:
                raise ValueError(f"{name} must be a boolean")

    def to_json(self) -> dict[str, bool | float | int]:
        return asdict(self)
