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
    "hotbar_slots": 9,
    "max_camera_delta_degrees": 45.0,
}
SCHEMA = ObservationSchema.from_json(SCHEMA_JSON)


def observation_json(**overrides):
    observation = {
        "schema": OBSERVATION_SCHEMA,
        "ray_kind": [1, 1, 0, 3, 1, 0],
        "ray_type": [4, 9, 0, 2, 4, 0],
        "ray_distance": [3.5, 7.25, 32.0, 2.0, 1.5, 32.0],
        "health": 20.0,
        "food": 20,
        "selected_slot": 0,
        "hotbar_item": [0, 6, 0, 0, 0, 0, 0, 0, 0],
        "hotbar_count": [0, 1, 0, 0, 0, 0, 0, 0, 0],
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


def test_observation_parses_fixed_shapes_and_categorical_sets() -> None:
    observation = PolicyObservation.from_json(observation_json(), SCHEMA)
    assert len(observation.ray_kind) == SCHEMA.rays
    assert len(observation.hotbar_item) == 9
    assert observation.visible_types(RAY_KIND_BLOCK) == {4, 9}
    assert observation.visible_types(RAY_KIND_ENTITY) == {2}
    assert observation.pitch == -12.5


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
        ("hotbar_item", [0, 7, 0, 0, 0, 0, 0, 0, 0], "hotbar item"),
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
