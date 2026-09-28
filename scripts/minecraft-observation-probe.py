import argparse
import json
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from minecraft_rl.minecraft_client import MinecraftClient, StepResult
from minecraft_rl.minecraft_interface import (
    RAY_KIND_BLOCK,
    RAY_KIND_ENTITY,
    PlayerAction,
    PolicyObservation,
)
from minecraft_rl.minecraft_launch import launched_client

NOOP = PlayerAction()
CENTER_TOLERANCE_DEGREES = 1e-3


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class Probe:
    def __init__(self, client: MinecraftClient) -> None:
        self.client = client
        self.privileged = client.privileged()
        registry = self.privileged.registry()
        self.block_id = {name: index for index, name in enumerate(registry["block"])}
        self.entity_id = {name: index for index, name in enumerate(registry["entity"])}
        self.item_id = {name: index for index, name in enumerate(registry["item"])}
        self.steps: list[StepResult] = []

    def step(self, action: PlayerAction = NOOP) -> StepResult:
        tick = self.client.status().game_time
        result = self.client.step(action)
        check(
            (result.info.tick_before, result.info.tick_after) == (tick, tick + 1),
            f"step advanced {result.info.tick_before}->{result.info.tick_after}",
        )
        self.steps.append(result)
        return result

    def settle(self, steps: int = 5) -> StepResult:
        result = self.step()
        for _ in range(steps - 1):
            result = self.step()
        return result

    def sees_block(self, observation: PolicyObservation, name: str) -> bool:
        return self.block_id[name] in observation.visible_types(RAY_KIND_BLOCK)

    def sees_entity(self, observation: PolicyObservation, name: str) -> bool:
        return self.entity_id[name] in observation.visible_types(RAY_KIND_ENTITY)

    def turn(self, degrees: float) -> StepResult:
        result = None
        remaining = degrees
        while abs(remaining) > 1e-9:
            delta = max(-45.0, min(45.0, remaining))
            result = self.step(PlayerAction(yaw_delta=delta))
            remaining -= delta
        return result


def run_visibility(probe: Probe, wait_seconds: float) -> dict:
    scene = probe.privileged.scene("visibility")
    observation = probe.settle().observation
    hidden = scene["hidden_block"]
    check(
        probe.privileged.block(hidden) == "minecraft:emerald_block",
        "hidden block missing from world",
    )
    hidden_entity = probe.privileged.entity(scene["hidden_entity"])
    check(hidden_entity["exists"], "hidden husk missing from world")

    nearby = probe.privileged.nearby(12)
    nearby_types = {entity["type"] for entity in nearby["entities"]}
    audit = {
        "privileged_nearby_blocks": nearby["blocks"],
        "privileged_nearby_entities": sorted(nearby_types),
    }
    check("minecraft:emerald_block" in nearby["blocks"], "audit lacks hidden block")
    check("minecraft:gold_block" in nearby["blocks"], "audit lacks behind block")
    check("minecraft:husk" in nearby_types, "audit lacks hidden husk")

    check(
        probe.sees_block(observation, "minecraft:diamond_block"),
        "visible block absent",
    )
    check(
        not probe.sees_block(observation, "minecraft:emerald_block"),
        "occluded block leaked",
    )
    check(
        not probe.sees_block(observation, "minecraft:gold_block"),
        "block behind the player leaked",
    )
    check(not probe.sees_entity(observation, "minecraft:husk"), "hidden husk leaked")

    idle_before = probe.client.observe()
    time.sleep(wait_seconds)
    idle_after = probe.client.observe()
    check(idle_after == idle_before, "observation changed while idle")

    turned = probe.turn(180.0).observation
    check(
        probe.sees_block(turned, "minecraft:gold_block"),
        "block behind did not appear after turning around",
    )
    check(
        not probe.sees_block(turned, "minecraft:diamond_block"),
        "front block still visible after turning around",
    )
    restored = probe.turn(180.0).observation
    check(
        probe.sees_block(restored, "minecraft:diamond_block"),
        "front block did not reappear after turning back",
    )

    wall_from, wall_to = scene["wall_from"], scene["wall_to"]
    probe.privileged.fill(wall_from, wall_to, "minecraft:air")
    revealed = probe.step().observation
    check(
        probe.sees_block(revealed, "minecraft:emerald_block"),
        "block did not appear once the wall was removed",
    )
    check(
        probe.sees_entity(revealed, "minecraft:husk"),
        "husk did not appear once the wall was removed",
    )
    probe.privileged.fill(wall_from, wall_to, "minecraft:stone")
    covered = probe.step().observation
    check(
        not probe.sees_entity(covered, "minecraft:husk"),
        "husk still visible after the wall was restored",
    )

    return audit | {
        "visible_block_seen": True,
        "occluded_block_hidden": True,
        "behind_block_hidden": True,
        "occluded_husk_hidden": True,
        "idle_observation_unchanged_seconds": wait_seconds,
        "rotation_hides_and_restores": True,
        "wall_removal_reveals_block_and_husk": True,
    }


def run_same_step_sync(probe: Probe) -> dict:
    probe.privileged.scene("visibility")
    probe.settle()
    before = probe.privileged.player()["server"]
    result = probe.step(PlayerAction(yaw_delta=30.0))
    after = probe.privileged.player()["server"]
    turned = abs(after["yaw"] - before["yaw"] - 30.0) < CENTER_TOLERANCE_DEGREES
    check(turned, f"yaw did not change by 30 on the same step: {before} -> {after}")
    check(result.info.tick_after == probe.client.status().game_time, "stale tick")
    turned_observation = result.observation

    pitched = probe.step(PlayerAction(pitch_delta=20.0))
    pitch_after = probe.privileged.player()["server"]["pitch"]
    check(
        abs(pitched.observation.pitch - pitch_after) < CENTER_TOLERANCE_DEGREES,
        "observation pitch is not the post-step server pitch",
    )
    check(
        abs(pitched.observation.pitch - turned_observation.pitch - 20.0)
        < CENTER_TOLERANCE_DEGREES,
        "pitch change did not appear in the same step's observation",
    )
    check(
        pitched.observation.ray_distance != turned_observation.ray_distance,
        "ray field did not change after pitching down",
    )
    probe.step(PlayerAction(pitch_delta=-20.0, yaw_delta=-30.0))

    move_before = probe.privileged.player()["server"]
    moved = probe.step(PlayerAction(forward=True))
    move_after = probe.privileged.player()["server"]
    moved_blocks = math.dist(
        (move_before["x"], move_before["z"]), (move_after["x"], move_after["z"])
    )
    check(moved_blocks > 1e-6, "forward did not move on the same step")
    diamond_distance_before = distance_to(probe, turned_observation, "diamond_block")
    diamond_distance_after = distance_to(probe, moved.observation, "diamond_block")
    return {
        "yaw_before": before["yaw"],
        "yaw_after": after["yaw"],
        "pitch_observed": pitched.observation.pitch,
        "pitch_server": pitch_after,
        "forward_move_blocks": moved_blocks,
        "diamond_min_distance_before": diamond_distance_before,
        "diamond_min_distance_after_forward": diamond_distance_after,
    }


def distance_to(probe: Probe, observation: PolicyObservation, block: str) -> float:
    block_id = probe.block_id[f"minecraft:{block}"]
    distances = [
        distance
        for kind, type_id, distance in zip(
            observation.ray_kind,
            observation.ray_type,
            observation.ray_distance,
            strict=True,
        )
        if kind == RAY_KIND_BLOCK and type_id == block_id
    ]
    return min(distances) if distances else math.inf


def run_combat(probe: Probe) -> dict:
    scene = probe.privileged.scene("combat")
    husk = scene["entity"]
    probe.settle()
    before = probe.privileged.entity(husk)
    first_hit = probe.step(PlayerAction(attack=True, hotbar=scene["weapon_slot"]))
    after = probe.privileged.entity(husk)
    check(
        after["health"] < before["health"],
        f"attack did not damage on the same step: {before} -> {after}",
    )
    check(first_hit.observation.selected_slot == scene["weapon_slot"], "slot unused")

    held = [probe.step(PlayerAction(attack=True)) for _ in range(3)]
    held_health = probe.privileged.entity(husk)["health"]
    ticks_to_next_hit = None
    for index in range(1, 40):
        probe.step()
        probe.step(PlayerAction(attack=True))
        health = probe.privileged.entity(husk)
        if not health["exists"] or health["health"] < held_health:
            ticks_to_next_hit = 2 * index
            break
    check(len(held) == 3, "held attack steps missing")
    return {
        "husk_health_before": before["health"],
        "husk_health_after_first_attack": after["health"],
        "hurt_time_after_first_attack": after["hurt_time"],
        "health_after_holding_attack_3_ticks": held_health,
        "ticks_until_repeated_click_hit_again": ticks_to_next_hit,
    }


def run_mining(probe: Probe) -> dict:
    scene = probe.privileged.scene("mining")
    target = scene["target_block"]
    probe.settle()
    probe.step(PlayerAction(pitch_delta=20.0))
    looking = probe.step()
    check(
        probe.sees_block(looking.observation, "minecraft:dirt"), "dirt target not seen"
    )
    held_ticks = 0
    while probe.privileged.block(target) != "minecraft:air":
        check(held_ticks < 200, "holding attack never broke the dirt block")
        probe.step(PlayerAction(attack=True))
        held_ticks += 1
    probe.step(PlayerAction(pitch_delta=-20.0))
    return {"held_attack_ticks_to_break_dirt": held_ticks}


def run_reset(probe: Probe, seed: int) -> dict:
    check(probe.privileged.kill(), "debug kill did not kill the player")
    dead = probe.step()
    check(dead.terminated, "step after death did not report terminated")

    results = []
    for preset in ("flat", "flat", "normal"):
        observation, info = probe.client.reset(seed, preset)
        status = probe.client.status()
        check(status.frozen and status.client_gated, "reset left the world ungated")
        step = probe.step()
        check(step.info.tick_before == status.game_time, "first step after reset")
        check(not step.terminated, "fresh episode reported terminated")
        results.append(asdict(info) | {"health": observation.health})
    return {"terminated_after_death": True, "resets": results}


THROUGHPUT_CONFIGS = {
    "paced": ("paced", True),
    "unpaced_render": ("unpaced", True),
    "unpaced_no_render": ("unpaced", False),
    "paced_again": ("paced", True),
}


def run_throughput(probe: Probe, steps: int) -> dict:
    probe.privileged.scene("visibility")
    probe.settle()
    return {
        label: measure_throughput(probe.client, steps, mode, render)
        for label, (mode, render) in THROUGHPUT_CONFIGS.items()
    }


def measure_throughput(
    client: MinecraftClient, steps: int, mode: str, render_frames: bool
) -> dict:
    previous = client.status_json()["pacing"]
    client.set_pacing(mode, render_frames=render_frames)
    tick_before = client.status().game_time
    started = time.perf_counter()
    samples = [client.step(NOOP).info.timing for _ in range(steps)]
    elapsed = time.perf_counter() - started
    ticks = client.status().game_time - tick_before
    client.set_pacing(previous["mode"], render_frames=previous["render_frames"])
    check(ticks == steps, f"{steps} steps advanced {ticks} ticks")
    phases = {}
    for name in [*asdict(samples[0]), "ipc_ms"]:
        values = [getattr(sample, name) for sample in samples]
        phases[name] = {
            "median": statistics.median(values),
            "p95": sorted(values)[int(0.95 * (len(values) - 1))],
        }
    return {
        "pacing": mode,
        "render_frames": render_frames,
        "steps": steps,
        "simulated_ticks": ticks,
        "wall_seconds": elapsed,
        "steps_per_second": steps / elapsed,
        "ticks_per_second": ticks / elapsed,
        "phases_ms": phases,
    }


def run_probes(
    client: MinecraftClient, wait_seconds: float, seed: int, steps: int, pacing: str
) -> dict:
    client.set_pacing(pacing)
    probe = Probe(client)
    return {
        "pacing": pacing,
        "client_status": client.status_json(),
        "schema": asdict(client.schema()),
        "visibility": run_visibility(probe, wait_seconds),
        "same_step_sync": run_same_step_sync(probe),
        "combat": run_combat(probe),
        "mining": run_mining(probe),
        "throughput": run_throughput(probe, steps),
        "reset": run_reset(probe, seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Minecraft structured-observation probe"
    )
    parser.add_argument("--port", type=int, default=47124)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--wait", type=float, default=3.0)
    parser.add_argument("--pacing", choices=("paced", "unpaced"), default="paced")
    parser.add_argument("--startup-timeout", type=float, default=240.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/minecraft-observation/result.json"),
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_suffix(".minecraft.log")
    result: dict[str, object] = {"seed": args.seed, "minecraft_log": str(log_path)}
    try:
        with launched_client(
            args.port, args.seed, log_path, args.startup_timeout
        ) as client:
            result |= run_probes(client, args.wait, args.seed, args.steps, args.pacing)
        result["passed"] = True
    except Exception as error:
        result |= {"passed": False, "error": f"{type(error).__name__}: {error}"}

    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in result.items() if key != "schema"}
    visibility = summary.get("visibility")
    if isinstance(visibility, dict):
        summary["visibility"] = {
            key: value
            for key, value in visibility.items()
            if not key.startswith("privileged_")
        }
    print(json.dumps(summary, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
