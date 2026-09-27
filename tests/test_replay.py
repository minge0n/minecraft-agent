import copy

import pytest

from minecraft_rl.minecraft_client import StepInfo, StepResult, StepTiming
from minecraft_rl.minecraft_interface import (
    ObservationSchema,
    PlayerAction,
    PolicyObservation,
)
from minecraft_rl.replay import (
    client_view_alignment,
    compare_traces,
    scripted_actions,
    scripted_events,
    trace_record,
)
from test_minecraft_interface import SCHEMA_JSON, observation_json

TIMING = StepTiming(1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 6.0)


def record(step: int, x: float = 0.5, pig_x: float = 3.5, ray: float = 32.0) -> dict:
    schema = ObservationSchema.from_json(SCHEMA_JSON)
    payload = observation_json()
    payload["ray_distance"] = [ray] + payload["ray_distance"][1:]
    if ray != 32.0:
        payload["ray_kind"] = [1] + payload["ray_kind"][1:]
        payload["ray_type"] = [5] + payload["ray_type"][1:]
    observation = PolicyObservation.from_json(payload, schema)
    result = StepResult(
        observation=observation,
        terminated=False,
        info=StepInfo(100 + step, step, step + 1, 900 + step, step + 1, TIMING),
    )
    trace = {
        "server": {"x": x, "y": 64.0, "z": 0.5, "yaw": 0.0},
        "client": {"tick": 900 + step, "x": x, "view_block_crc32": f"b{step}"},
        "entities": [
            {"id": 7, "type": "minecraft:pig", "x": pig_x, "y": 64.0, "z": 1.0},
            {"id": 9, "type": "minecraft:husk", "x": 1.0, "y": 64.0, "z": 2.0},
        ],
        "block_crc32": f"b{step + 1}",
        "entity_count": 2,
        "raining": False,
        "thundering": False,
    }
    trace["client"]["view_entity_count"] = 2
    return trace_record(step, PlayerAction(), None, result, trace, {7: "pig"})


def test_scripted_actions_are_fixed_and_cover_every_action_kind() -> None:
    actions = scripted_actions()
    assert actions == scripted_actions()
    for field in ("forward", "back", "left", "right", "jump", "sneak", "sprint"):
        assert any(getattr(action, field) for action in actions), field
    assert any(action.attack for action in actions)
    assert any(action.use for action in actions)
    assert any(action.yaw_delta != 0 for action in actions)
    assert any(action.pitch_delta != 0 for action in actions)
    assert {action.hotbar for action in actions} >= {-1, 0, 1, 2}
    assert all(step < len(actions) for step in scripted_events())


def test_trace_record_separates_policy_from_privileged_data() -> None:
    traced = record(0)
    assert set(traced) == {"step", "action", "event", "info", "policy", "privileged"}
    assert set(traced["policy"]) == {
        "observation_sha256",
        "observation",
        "terminated",
        "reward",
    }
    assert traced["privileged"]["entities"]["pig"]["x"] == 3.5
    assert traced["privileged"]["other_entities"][0]["type"] == "minecraft:husk"


def test_identical_traces_compare_identical_despite_process_counters() -> None:
    reference = [record(step) for step in range(4)]
    other = copy.deepcopy(reference)
    for traced in other:
        traced["info"]["step_id"] += 1000
        traced["info"]["client_tick"] += 50
        traced["privileged"]["client_player"]["tick"] += 50
        traced["info"]["timing"]["total_ms"] = 99.0
    comparison = compare_traces(reference, other)
    assert comparison["identical"]
    assert comparison["first_difference_step"] is None


def test_divergence_is_located_per_field_with_magnitude() -> None:
    reference = [record(step) for step in range(4)]
    other = [record(0), record(1), record(2, pig_x=4.0), record(3, x=0.75, ray=5.0)]
    comparison = compare_traces(reference, other)
    fields = comparison["first_divergence_by_field"]
    assert fields["privileged.entities.pig.x"] == 2
    assert fields["privileged.server_player.x"] == 3
    assert fields["policy.observation_sha256"] == 3
    assert comparison["max_abs_difference_by_field"]["privileged.entities.pig.x"] == 0.5
    assert comparison["policy_observation"]["first_differing_step"] == 3
    assert comparison["policy_observation"]["max_rays_differing"] == 1
    assert comparison["player_position"]["first_nonzero_step"] == 3
    assert comparison["entity_position"]["pig"]["max_distance"] == pytest.approx(0.5)


def test_compare_rejects_traces_of_different_length() -> None:
    with pytest.raises(ValueError):
        compare_traces([record(0)], [record(0), record(1)])


def test_client_view_alignment_detects_one_step_lag() -> None:
    alignment = client_view_alignment([record(step) for step in range(5)])
    assert alignment["blocks_matching_server_lag_0"] == 0
    assert alignment["blocks_matching_server_lag_1"] == 4
    assert alignment["entity_count_matching_server_lag_0"] == 5
