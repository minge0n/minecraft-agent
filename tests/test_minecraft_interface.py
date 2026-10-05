import pytest

from minecraft_rl.minecraft_interface import (
    OBSERVATION_SCHEMA,
    RAY_KIND_BLOCK,
    RAY_KIND_ENTITY,
    RAY_KIND_NONE,
    ObservationSchema,
    PlayerAction,
    PolicyObservation,
    SchemaError,
)

SCHEMA_JSON = {
    "observation_schema": OBSERVATION_SCHEMA,
    "rows": 2,
    "columns": 3,
    "vertical_fov_degrees": 70.0,
    "horizontal_fov_degrees": 94.0,
    "max_distance": 32.0,
    "ray_kinds": 4,
    "block_types": 10,
    "fluid_types": 3,
    "entity_types": 5,
    "item_types": 7,
    "effect_types": 4,
    "hotbar_slots": 9,
    "inventory_slots": 36,
    "armor_slots": 4,
    "effect_slots": 8,
    "air_bubbles": 10,
    "durability_steps": 13,
    "max_camera_delta_degrees": 45.0,
}
SCHEMA = ObservationSchema.from_json(SCHEMA_JSON)


def slots(length, values=(), fill=0):
    return list(values) + [fill] * (length - len(values))


def observation_json(**overrides):
    observation = {
        "schema": OBSERVATION_SCHEMA,
        "ray_kind": [1, 1, 0, 3, 1, 0],
        "ray_type": [4, 9, 0, 2, 4, 0],
        "ray_distance": [3.5, 7.25, 32.0, 2.0, 1.5, 32.0],
        "health": 20.0,
        "max_health": 20.0,
        "absorption": 0.0,
        "food": 20,
        "air_bubbles": 10,
        "armor": 0,
        "xp_level": 0,
        "xp_progress": 0.0,
        "selected_slot": 0,
        "inventory_item": slots(36, [0, 6, 0, 0, 0, 0, 0, 0, 0, 3]),
        "inventory_count": slots(36, [0, 1, 0, 0, 0, 0, 0, 0, 0, 12]),
        "inventory_durability": slots(36, [1.0, 0.5], fill=1.0),
        "armor_item": [0, 0, 0, 5],
        "armor_durability": [1.0, 1.0, 1.0, 0.25],
        "offhand_item": 0,
        "offhand_count": 0,
        "offhand_durability": 1.0,
        "effect_type": slots(8, [1, 3]),
        "effect_amplifier": slots(8, [0, 1]),
        "effect_seconds": slots(8, [30, -1]),
        "pitch": -12.5,
    }
    observation.update(overrides)
    return observation


def test_schema_reports_ray_count_and_vocabularies() -> None:
    assert SCHEMA.rays == 6
    assert SCHEMA.type_vocabulary(RAY_KIND_NONE) == 1
    assert SCHEMA.type_vocabulary(RAY_KIND_BLOCK) == 10
    assert SCHEMA.type_vocabulary(RAY_KIND_ENTITY) == 5


def test_schema_rejects_other_versions_and_layouts() -> None:
    with pytest.raises(SchemaError, match="unsupported schema"):
        ObservationSchema.from_json(SCHEMA_JSON | {"observation_schema": "rgb-v1"})
    with pytest.raises(SchemaError, match="hotbar"):
        ObservationSchema.from_json(SCHEMA_JSON | {"hotbar_slots": 10})
    with pytest.raises(SchemaError, match="effect_slots"):
        ObservationSchema.from_json(SCHEMA_JSON | {"effect_slots": 4})


def test_observation_parses_fixed_shapes_and_categorical_sets() -> None:
    observation = PolicyObservation.from_json(observation_json(), SCHEMA)
    assert len(observation.ray_kind) == SCHEMA.rays
    assert len(observation.inventory_item) == 36
    assert observation.hotbar_item == (0, 6, 0, 0, 0, 0, 0, 0, 0)
    assert observation.hotbar_count[1] == 1
    assert observation.inventory_count[9] == 12
    assert observation.main_hand_item == 0
    assert observation.armor_item == (0, 0, 0, 5)
    assert observation.effect_type[:2] == (1, 3)
    assert observation.effect_seconds[1] == -1
    assert observation.visible_types(RAY_KIND_BLOCK) == {4, 9}
    assert observation.visible_types(RAY_KIND_ENTITY) == {2}
    assert observation.pitch == -12.5


def test_digest_covers_the_whole_self_state() -> None:
    base = PolicyObservation.from_json(observation_json(), SCHEMA).digest()
    changed = PolicyObservation.from_json(
        observation_json(effect_seconds=slots(8, [29, -1])), SCHEMA
    ).digest()
    assert base != changed


def test_observation_rejects_extra_or_missing_fields() -> None:
    with pytest.raises(SchemaError, match="fields differ"):
        PolicyObservation.from_json(observation_json(x=100.5), SCHEMA)
    without_food = observation_json()
    del without_food["food"]
    with pytest.raises(SchemaError, match="fields differ"):
        PolicyObservation.from_json(without_food, SCHEMA)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ray_kind", [1, 1, 0, 3, 1], "6 integers"),
        ("ray_kind", [1, 1, 0, 4, 1, 0], "kind 4"),
        ("ray_type", [4, 10, 0, 2, 4, 0], "type 10"),
        ("ray_type", [4, 9, 1, 2, 4, 0], "type 1 out of range for kind 0"),
        ("ray_type", [4, 9.0, 0, 2, 4, 0], "integers"),
        ("ray_distance", [3.5, 7.25, 32.0, 2.0, 40.0, 32.0], "distance 40"),
        ("ray_distance", [3.5, 7.25, 3.0, 2.0, 1.5, 32.0], "maximum distance"),
        ("inventory_item", slots(36, [0, 7]), "inventory item"),
        ("inventory_item", slots(9), "36 integers"),
        ("inventory_count", slots(36, [-1]), "negative inventory"),
        ("inventory_durability", slots(36, [1.5], fill=1.0), r"\[0, 1\]"),
        ("armor_item", [0, 0, 0, 7], "armor item"),
        ("offhand_item", 7, "offhand item"),
        ("effect_type", slots(8, [5]), "effect type 5"),
        ("effect_type", slots(8, [3, 1]), "registry order"),
        ("effect_type", slots(8, [0, 1]), "empty effect slot"),
        ("effect_amplifier", slots(8, [0, 0, 2]), "empty effect slot"),
        ("effect_seconds", slots(8, [-2, -1]), "effect values"),
        ("air_bubbles", 11, "air bubbles"),
        ("xp_level", -1, "xp level"),
        ("xp_progress", 1.2, r"\[0, 1\]"),
        ("armor", 1.5, "armor"),
        ("selected_slot", 9, "selected slot"),
        ("pitch", 91.0, "pitch"),
    ],
)
def test_observation_rejects_out_of_contract_values(field, value, message) -> None:
    with pytest.raises(SchemaError, match=message):
        PolicyObservation.from_json(observation_json(**{field: value}), SCHEMA)


def test_player_action_serializes_every_factor() -> None:
    action = PlayerAction(forward=True, jump=True, attack=True, yaw_delta=-15.0)
    assert action.to_json() == {
        "forward": True,
        "back": False,
        "left": False,
        "right": False,
        "jump": True,
        "sneak": False,
        "sprint": False,
        "attack": True,
        "use": False,
        "yaw_delta": -15.0,
        "pitch_delta": 0.0,
        "hotbar": -1,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"yaw_delta": 45.5},
        {"pitch_delta": float("nan")},
        {"hotbar": 9},
        {"hotbar": True},
        {"forward": 1},
    ],
)
def test_player_action_rejects_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError):
        PlayerAction(**kwargs)
