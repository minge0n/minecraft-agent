import math
from dataclasses import asdict, dataclass
from typing import Any, Self

OBSERVATION_SCHEMA = "visible-field-v1"
RAY_KIND_NONE = 0
RAY_KIND_BLOCK = 1
RAY_KIND_FLUID = 2
RAY_KIND_ENTITY = 3
RAY_KINDS = 4
HOTBAR_SLOTS = 9
KEEP_HOTBAR_SLOT = -1
MAX_CAMERA_DELTA_DEGREES = 45.0

OBSERVATION_FIELDS = frozenset(
    {
        "schema",
        "ray_kind",
        "ray_type",
        "ray_distance",
        "health",
        "food",
        "selected_slot",
        "hotbar_item",
        "hotbar_count",
        "pitch",
    }
)


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

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Self:
        if payload.get("observation_schema") != OBSERVATION_SCHEMA:
            raise SchemaError(f"unsupported schema {payload.get('observation_schema')}")
        if payload["ray_kinds"] != RAY_KINDS or payload["hotbar_slots"] != HOTBAR_SLOTS:
            raise SchemaError("ray kind or hotbar layout differs from this client")
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
    ray_kind: tuple[int, ...]
    ray_type: tuple[int, ...]
    ray_distance: tuple[float, ...]
    health: float
    food: int
    selected_slot: int
    hotbar_item: tuple[int, ...]
    hotbar_count: tuple[int, ...]
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

        items = _integers(payload["hotbar_item"], HOTBAR_SLOTS, "hotbar_item")
        counts = _integers(payload["hotbar_count"], HOTBAR_SLOTS, "hotbar_count")
        if any(not 0 <= item < schema.item_types for item in items):
            raise SchemaError("hotbar item out of range")
        if any(count < 0 for count in counts):
            raise SchemaError("negative hotbar count")
        selected_slot = payload["selected_slot"]
        if not 0 <= selected_slot < HOTBAR_SLOTS:
            raise SchemaError(f"selected slot {selected_slot} out of range")
        pitch = float(payload["pitch"])
        if not -90.0 <= pitch <= 90.0:
            raise SchemaError(f"pitch {pitch} out of range")

        return cls(
            ray_kind=kinds,
            ray_type=types,
            ray_distance=distances,
            health=float(payload["health"]),
            food=int(payload["food"]),
            selected_slot=selected_slot,
            hotbar_item=items,
            hotbar_count=counts,
            pitch=pitch,
        )

    def visible_types(self, kind: int) -> frozenset[int]:
        return frozenset(
            type_id
            for ray_kind, type_id in zip(self.ray_kind, self.ray_type, strict=True)
            if ray_kind == kind
        )


def _integers(values: list[Any], length: int, name: str) -> tuple[int, ...]:
    if len(values) != length or any(type(value) is not int for value in values):
        raise SchemaError(f"{name} must hold {length} integers")
    return tuple(values)


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
