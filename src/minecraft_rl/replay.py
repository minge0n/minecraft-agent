"""Replay characterization: predetermined action scripts, per-step traces, comparison.

The trace keeps the policy observation and privileged diagnostics in separate
sub-objects. Privileged data exists only to measure reproducibility and is never a
policy input.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from minecraft_rl.minecraft_client import MinecraftClient, StepResult
from minecraft_rl.minecraft_interface import PlayerAction

NOOP = PlayerAction()
REMOVE_SAND_SUPPORT = "remove_sand_support"
CHECKPOINT_STEPS = (25, 50, 100, 150, 200, 250, 300)
REPLAY_COLUMN = (8, 8)
# Fields expected to differ between otherwise identical runs; see
# docs/replay-characterization.md.
RUN_OFFSET_FIELDS = frozenset(
    {"privileged.server_player.tick_count", "privileged.client_player.tick_count"}
)


def _repeat(action: PlayerAction, count: int) -> list[PlayerAction]:
    return [action] * count


def scripted_actions() -> list[PlayerAction]:
    """The fixed open-loop action sequence for the `replay` debug scene.

    It never reacts to state, so every run receives exactly the same inputs. The
    player starts at the scene origin facing +z (yaw 0) with an empty hand selected.
    """
    actions: list[PlayerAction] = []
    actions += _repeat(NOOP, 10)
    # Mine the floating dirt block two blocks ahead with an empty hand.
    actions += _repeat(PlayerAction(attack=True, hotbar=0), 1)
    actions += _repeat(PlayerAction(attack=True), 21)
    actions += _repeat(NOOP, 5)
    # Face the no-AI husk, select the sword, and hit it twice at full charge.
    actions += [PlayerAction(yaw_delta=-45.0, hotbar=1)]
    actions += _repeat(NOOP, 20)
    actions += [PlayerAction(attack=True)]
    actions += _repeat(NOOP, 20)
    actions += [PlayerAction(attack=True)]
    actions += _repeat(NOOP, 5)
    # Walk forward over the dropped dirt item and jump onto the stone step.
    actions += [PlayerAction(yaw_delta=45.0)]
    actions += _repeat(PlayerAction(forward=True), 30)
    actions += _repeat(PlayerAction(forward=True, jump=True), 8)
    actions += _repeat(NOOP, 10)
    # Look around: turn 180 degrees, pitch up and down, strafe.
    actions += _repeat(PlayerAction(yaw_delta=45.0), 4)
    actions += [PlayerAction(pitch_delta=-30.0), PlayerAction(pitch_delta=30.0)]
    actions += _repeat(PlayerAction(left=True), 6)
    actions += _repeat(PlayerAction(right=True), 6)
    # Sneak backwards, then sprint forward.
    actions += _repeat(PlayerAction(sneak=True, back=True), 10)
    actions += _repeat(PlayerAction(sprint=True, forward=True), 10)
    actions += _repeat(NOOP, 5)
    # Place a cobblestone block on the ground about two blocks ahead; a steeper pitch
    # would target the cell the player occupies and the placement would be refused.
    actions += [PlayerAction(hotbar=2, pitch_delta=45.0)]
    actions += [PlayerAction(use=True)]
    actions += _repeat(NOOP, 3)
    actions += [PlayerAction(pitch_delta=-45.0)]
    # Let the falling sand, dropped items and any mob AI run.
    actions += _repeat(NOOP, 60)
    return actions


def scripted_events() -> dict[int, str]:
    """Privileged world edits applied before the given step, identically per run."""
    return {150: REMOVE_SAND_SUPPORT}


@dataclass(frozen=True)
class ReplayScene:
    """A built `replay` debug scene and what its trace records refer to."""

    scene: dict[str, Any]
    roles: dict[int, str]
    region: tuple[list[int], list[int]]


def build_controlled_replay(client: MinecraftClient, seed: int) -> ReplayScene:
    """Reset to a fresh flat world and build the controlled `replay` scene."""
    client.set_pacing("paced")
    client.reset(seed, "flat")
    scene = client.privileged().scene("replay", at=REPLAY_COLUMN, controlled=True)
    return ReplayScene(
        scene, {scene["husk"]: "husk"}, (scene["arena_from"], scene["arena_to"])
    )


def play_replay(client: MinecraftClient, replay: ReplayScene) -> list[dict[str, Any]]:
    """Play the fixed action script once and return one trace record per step.

    Raises AssertionError if a step does not advance exactly one world tick and one
    client tick.
    """
    privileged = client.privileged()
    events = scripted_events()
    support = replay.scene["sand_support"]
    records = []
    for step, action in enumerate(scripted_actions()):
        if events.get(step) == REMOVE_SAND_SUPPORT:
            privileged.fill(support, support, "minecraft:air")
        result = client.step(action)
        if result.info.tick_after != result.info.tick_before + 1:
            raise AssertionError(f"step {step} did not advance exactly one tick")
        trace = privileged.trace(*replay.region)
        records.append(
            trace_record(step, action, events.get(step), result, trace, replay.roles)
        )
    for previous, current in zip(records, records[1:], strict=False):
        if current["info"]["client_tick"] != previous["info"]["client_tick"] + 1:
            raise AssertionError(f"client ticks not consecutive at {current['step']}")
    return records


def trace_record(
    step: int,
    action: PlayerAction,
    event: str | None,
    result: StepResult,
    trace: Mapping[str, Any],
    roles: Mapping[int, str],
) -> dict[str, Any]:
    """Combine one step's reply and privileged snapshot into a trace record."""
    named: dict[str, Any] = {}
    others = []
    for entity in trace["entities"]:
        role = roles.get(entity["id"])
        fields = {key: value for key, value in entity.items() if key != "id"}
        if role is None:
            others.append(fields)
        else:
            named[role] = fields
    others.sort(
        key=lambda entity: (entity["type"], entity["x"], entity["y"], entity["z"])
    )
    info = result.info
    return {
        "step": step,
        "action": action.to_json(),
        "event": event,
        "info": {
            "step_id": info.step_id,
            "tick_before": info.tick_before,
            "tick_after": info.tick_after,
            "client_tick": info.client_tick,
            "game_time": info.game_time,
            "pacing": info.pacing,
            "timing": asdict(info.timing),
        },
        "policy": {
            "observation_sha256": result.observation.digest(),
            "observation": asdict(result.observation),
            "terminated": result.terminated,
            "reward": None,
        },
        "privileged": {
            "server_player": trace["server"],
            "client_player": trace["client"],
            "entities": named,
            "other_entities": others,
            "block_crc32": trace["block_crc32"],
            "entity_count": trace["entity_count"],
            "raining": trace["raining"],
            "thundering": trace["thundering"],
        },
    }


def _flatten(value: Any, prefix: str, out: dict[str, Any]) -> None:
    if isinstance(value, Mapping):
        for key in sorted(value):
            _flatten(value[key], f"{prefix}.{key}" if prefix else key, out)
    elif isinstance(value, list | tuple):
        out[prefix] = tuple(_freeze(item) for item in value)
    else:
        out[prefix] = value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple((key, _freeze(value[key])) for key in sorted(value))
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return value


def comparable_fields(
    record: Mapping[str, Any], first: Mapping[str, Any]
) -> dict[str, Any]:
    """Flatten a record into the fields compared between runs.

    Counters that accumulate across episodes of one process (step ID, client tick)
    are made relative to the run's first step; wall-clock timings are excluded.
    """
    fields: dict[str, Any] = {}
    info = record["info"]
    fields["info.step_offset"] = info["step_id"] - first["info"]["step_id"]
    fields["info.client_tick_offset"] = (
        info["client_tick"] - first["info"]["client_tick"]
    )
    for key in ("tick_before", "tick_after", "game_time"):
        fields[f"info.{key}"] = info[key]
    policy = record["policy"]
    fields["policy.observation_sha256"] = policy["observation_sha256"]
    fields["policy.terminated"] = policy["terminated"]
    fields["policy.reward"] = policy["reward"]
    _flatten(record["privileged"], "privileged", fields)
    client = "privileged.client_player."
    fields.pop(client + "tick", None)
    return fields


def _position(entity: Mapping[str, Any] | None) -> tuple[float, float, float] | None:
    if entity is None or "x" not in entity:
        return None
    return (entity["x"], entity["y"], entity["z"])


def _distance(a: Sequence[float] | None, b: Sequence[float] | None) -> float:
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return math.inf
    return math.dist(a, b)


def _ray_mismatches(a: Mapping[str, Any], b: Mapping[str, Any]) -> int:
    rays = zip(
        a["ray_kind"],
        a["ray_type"],
        a["ray_distance"],
        b["ray_kind"],
        b["ray_type"],
        b["ray_distance"],
        strict=True,
    )
    return sum(
        1
        for kind_a, type_a, distance_a, kind_b, type_b, distance_b in rays
        if (kind_a, type_a, distance_a) != (kind_b, type_b, distance_b)
    )


def _first(steps: Iterable[int]) -> int | None:
    return next(iter(steps), None)


def compare_traces(
    reference: Sequence[Mapping[str, Any]], other: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Report where and how much `other` diverges from `reference`, step by step."""
    if len(reference) != len(other):
        raise ValueError("traces must cover the same number of steps")
    first_divergence: dict[str, int] = {}
    numeric_max: dict[str, float] = {}
    differing_steps = []
    observation_steps = []
    ray_mismatch = []
    player_distance = []
    entity_distance: dict[str, list[float]] = {}
    for index, (a, b) in enumerate(zip(reference, other, strict=True)):
        fields_a = comparable_fields(a, reference[0])
        fields_b = comparable_fields(b, other[0])
        step_differs = False
        for name in sorted(fields_a.keys() | fields_b.keys()):
            value_a, value_b = fields_a.get(name), fields_b.get(name)
            if value_a == value_b:
                continue
            step_differs = True
            first_divergence.setdefault(name, index)
            if isinstance(value_a, int | float) and isinstance(value_b, int | float):
                if not isinstance(value_a, bool) and not isinstance(value_b, bool):
                    delta = abs(value_a - value_b)
                    numeric_max[name] = max(numeric_max.get(name, 0.0), delta)
        if step_differs:
            differing_steps.append(index)
        obs_a, obs_b = a["policy"]["observation"], b["policy"]["observation"]
        mismatches = _ray_mismatches(obs_a, obs_b)
        ray_mismatch.append(mismatches)
        if a["policy"]["observation_sha256"] != b["policy"]["observation_sha256"]:
            observation_steps.append(index)
        player_distance.append(
            _distance(
                _position(a["privileged"]["server_player"]),
                _position(b["privileged"]["server_player"]),
            )
        )
        roles = a["privileged"]["entities"].keys() | b["privileged"]["entities"].keys()
        for role in roles:
            entity_distance.setdefault(role, []).append(
                _distance(
                    _position(a["privileged"]["entities"].get(role)),
                    _position(b["privileged"]["entities"].get(role)),
                )
            )
    return {
        "steps": len(reference),
        "identical": not differing_steps,
        "steps_with_any_difference": len(differing_steps),
        "first_difference_step": _first(differing_steps),
        "first_divergence_by_field": dict(
            sorted(first_divergence.items(), key=lambda item: item[1])
        ),
        "max_abs_difference_by_field": numeric_max,
        "policy_observation": {
            "differing_steps": len(observation_steps),
            "first_differing_step": _first(observation_steps),
            "max_rays_differing": max(ray_mismatch, default=0),
            "rays_differing_at_checkpoints": _checkpoints(ray_mismatch),
        },
        "player_position": _series_summary(player_distance),
        "entity_position": {
            role: _series_summary(series)
            for role, series in sorted(entity_distance.items())
        },
    }


def unexplained_fields(comparison: Mapping[str, Any]) -> list[str]:
    """Fields of a `compare_traces` result that differ beyond the run offset."""
    return sorted(set(comparison["first_divergence_by_field"]) - RUN_OFFSET_FIELDS)


def client_view_alignment(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Measure how far the client's copy of the scene lags the server within one run.

    The client fingerprint is taken at the end of the client tick of step N, before
    the server ticks step N, so a client that is exactly one server step behind
    matches the server fingerprint of step N-1.
    """
    lags: dict[str, int] = {}
    for name, server_key, client_key in (
        ("blocks", "block_crc32", "view_block_crc32"),
        ("entity_count", "entity_count", "view_entity_count"),
    ):
        server = [record["privileged"][server_key] for record in records]
        client = [
            record["privileged"]["client_player"][client_key] for record in records
        ]
        for lag in range(4):
            matches = sum(
                1
                for step in range(lag, len(records))
                if client[step] == server[step - lag]
            )
            lags[f"{name}_matching_server_lag_{lag}"] = matches
        lags[f"{name}_steps"] = len(records)
    return lags


def _checkpoints(series: Sequence[float]) -> dict[str, float]:
    return {str(step): series[step] for step in CHECKPOINT_STEPS if step < len(series)}


def _series_summary(series: Sequence[float]) -> dict[str, Any]:
    nonzero = [index for index, value in enumerate(series) if value != 0.0]
    return {
        "first_nonzero_step": _first(nonzero),
        "max_distance": max(series, default=0.0),
        "final_distance": series[-1] if series else 0.0,
        "distance_at_checkpoints": _checkpoints(series),
    }
