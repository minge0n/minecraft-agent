"""Runtime probe of the abstract actions of `player-action-v3`
(docs/decisions/actions.md) and the privileged step events of the reward
(docs/decisions/reward.md).

The privileged `crafting` scene gives the player logs, cobblestone, a helmet and
two dirt stacks, with a crafting table two blocks ahead. The probe checks that:

- every step, with or without an abstract action, advances exactly one tick;
- an abstract action replaces the physical part of its step;
- move_item moves, merges, swaps, equips armor and fills the offhand, and an
  invalid move changes nothing;
- craft makes planks anywhere, needs the crafting table for a 3 x 3 recipe,
  consumes the real ingredients and creates nothing when it fails;
- a hostile mob keeps moving while the agent does inventory work;
- broken blocks and completed advancements arrive as step events, and never in
  the observation.

A non-recording infrastructure test (docs/decisions/recording.md).
"""

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

from minecraft_rl.minecraft_client import MinecraftClient, StepResult
from minecraft_rl.minecraft_interface import AbstractAction, PlayerAction
from minecraft_rl.minecraft_launch import launched_client
from minecraft_rl.provenance import git_commit
from minecraft_rl.recording import write_json


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class Probe:
    def __init__(self, client: MinecraftClient) -> None:
        self.client = client
        self.privileged = client.privileged()
        registry = self.privileged.registry()
        self.items = {name: index for index, name in enumerate(registry["item"])}
        self.blocks = {name: index for index, name in enumerate(registry["block"])}
        self.steps: list[dict[str, Any]] = []

    def item(self, name: str) -> int:
        return self.items[f"minecraft:{name}"]

    def step(self, action: PlayerAction) -> StepResult:
        result = self.client.step(action)
        check(
            result.info.tick_after == result.info.tick_before + 1,
            f"step advanced {result.info.tick_before}->{result.info.tick_after}",
        )
        self.steps.append(
            {
                "action": action.to_json(),
                "events": result.events.abstract,
                "advancements": list(result.events.advancements),
            }
        )
        return result

    def abstract(self, kind: str, **arguments: int) -> StepResult:
        return self.step(PlayerAction(abstract=AbstractAction(kind, **arguments)))

    def outcome(self, result: StepResult) -> str:
        check(result.events.abstract is not None, "no abstract outcome reported")
        return result.events.abstract["outcome"]


def slot(result: StepResult, index: int) -> tuple[int, int]:
    o = result.observation
    if index < 36:
        return o.inventory_item[index], o.inventory_count[index]
    if index < 40:
        return o.armor_item[index - 36], int(o.armor_item[index - 36] != 0)
    return o.offhand_item, o.offhand_count


def holdings(result: StepResult) -> dict[int, int]:
    """Total count per item type in the main inventory."""
    totals: dict[int, int] = {}
    o = result.observation
    for item, count in zip(o.inventory_item, o.inventory_count, strict=True):
        if item:
            totals[item] = totals.get(item, 0) + count
    return totals


def run(probe: Probe) -> dict[str, Any]:
    out: dict[str, Any] = {}
    client = probe.client
    client.reset(12345, "flat")
    client.set_pacing("unpaced")
    scene = probe.privileged.scene("crafting")
    out["scene"] = scene
    log, cobble = probe.item("oak_log"), probe.item("cobblestone")
    helmet, dirt = probe.item("iron_helmet"), probe.item("dirt")
    planks, table = probe.item("oak_planks"), probe.item("crafting_table")

    # A physical step for a reference, then the abstract action that replaces it.
    probe.step(PlayerAction())
    moved = probe.step(
        PlayerAction(
            forward=True, abstract=AbstractAction("move_item", source=1, destination=5)
        )
    )
    check(probe.outcome(moved) == "moved", f"move: {moved.events.abstract}")
    check(
        slot(moved, 5) == (cobble, 8) and slot(moved, 1) == (0, 0),
        "cobblestone not moved",
    )
    player = probe.privileged.player()["server"]
    out["position_after_move"] = [player["x"], player["z"]]
    check(
        abs(player["z"] - scene["origin"][2] - 0.5) < 1e-6,
        "the physical part of an abstract step was not replaced by the no-op",
    )

    merged = probe.abstract("move_item", source=9, destination=10)
    check(probe.outcome(merged) == "merged", f"merge: {merged.events.abstract}")
    check(
        slot(merged, 10) == (dirt, 64) and slot(merged, 9) == (dirt, 26), "merge counts"
    )

    swapped = probe.abstract("move_item", source=0, destination=10)
    check(probe.outcome(swapped) == "swapped", f"swap: {swapped.events.abstract}")
    check(slot(swapped, 0) == (dirt, 64) and slot(swapped, 10) == (log, 4), "swap")

    hotbar = probe.abstract("move_item", source=10, destination=3)
    check(slot(hotbar, 3) == (log, 4), "move into the hotbar")

    equipped = probe.abstract("move_item", source=2, destination=39)
    check(
        probe.outcome(equipped) == "moved" and slot(equipped, 39)[0] == helmet, "equip"
    )
    # Vanilla applies equipment attributes in the next entity tick.
    worn = probe.step(PlayerAction())
    check(worn.observation.armor > 0, "the helmet gives no armor points")
    out["armor_points_same_step"] = equipped.observation.armor
    out["armor_points_next_step"] = worn.observation.armor

    wrong = probe.abstract("move_item", source=0, destination=36)
    check(
        probe.outcome(wrong) == "invalid_destination",
        f"dirt on feet: {wrong.events.abstract}",
    )
    check(
        wrong.observation.inventory_item == equipped.observation.inventory_item,
        "invalid move changed state",
    )

    empty = probe.abstract("move_item", source=20, destination=21)
    check(probe.outcome(empty) == "invalid_empty_source", "empty source")

    offhand = probe.abstract("move_item", source=9, destination=40)
    check(slot(offhand, 40) == (dirt, 26), "offhand")

    crafted = probe.abstract("craft", item=planks)
    check(probe.outcome(crafted) == "crafted", f"planks: {crafted.events.abstract}")
    held = dict(
        zip(
            crafted.observation.inventory_item,
            crafted.observation.inventory_count,
            strict=False,
        )
    )
    check(
        held.get(planks) == 4 and held.get(log) == 3,
        f"planks did not use one log: {held}",
    )

    # Turn away from the table: a 3 x 3 recipe must fail and change nothing.
    probe.step(PlayerAction(yaw_delta=45.0))
    probe.step(PlayerAction(yaw_delta=45.0))
    probe.step(PlayerAction(yaw_delta=45.0))
    probe.step(PlayerAction(yaw_delta=45.0))
    away = probe.abstract("craft", item=probe.item("stone_pickaxe"))
    check(
        probe.outcome(away).startswith("invalid"),
        f"pickaxe without table: {away.events.abstract}",
    )
    out["craft_away_from_table"] = probe.outcome(away)

    no_ingredients = probe.abstract("craft", item=probe.item("diamond_pickaxe"))
    check(
        probe.outcome(no_ingredients).startswith("invalid"), "diamond pickaxe crafted"
    )
    check(
        no_ingredients.observation.inventory_item == away.observation.inventory_item
        and no_ingredients.observation.inventory_count
        == away.observation.inventory_count,
        "a failed craft changed the inventory",
    )

    # Four more planks: one log each time. The table needs four planks after the
    # sticks have used two.
    more = probe.abstract("craft", item=planks)
    check(holdings(more).get(planks) == 8, f"second planks craft: {holdings(more)}")
    sticks = probe.abstract("craft", item=probe.item("stick"))
    check(probe.outcome(sticks) == "crafted", f"sticks: {sticks.events.abstract}")
    crafting_table = probe.abstract("craft", item=table)
    check(
        probe.outcome(crafting_table) == "crafted",
        f"table: {crafting_table.events.abstract}",
    )

    # Face the table again and craft a stone pickaxe: 3 cobblestone and 2 sticks.
    for _ in range(4):
        probe.step(PlayerAction(yaw_delta=-45.0))
    pickaxe = probe.abstract("craft", item=probe.item("stone_pickaxe"))
    check(
        probe.outcome(pickaxe) == "crafted",
        f"pickaxe at table: {pickaxe.events.abstract}",
    )
    held = dict(
        zip(
            pickaxe.observation.inventory_item,
            pickaxe.observation.inventory_count,
            strict=False,
        )
    )
    check(held.get(cobble) == 5, f"the pickaxe did not use 3 cobblestone: {held}")
    advancements = [a for s in probe.steps for a in s["advancements"]]
    out["advancement_events"] = advancements
    # The root advancement needs a crafting table in the inventory.
    check(
        "minecraft:story/root" in advancements, f"no root advancement: {advancements}"
    )
    out["outcomes"] = [s["events"]["outcome"] for s in probe.steps if s["events"]]
    return out


def run_world_keeps_moving(probe: Probe) -> dict[str, Any]:
    """A zombie two blocks away keeps ticking while the agent moves items."""
    client = probe.client
    client.reset(12345, "flat")
    client.set_pacing("unpaced")
    scene = probe.privileged.scene("combat", ai=True)
    husk = scene["entity"]
    start = probe.privileged.entity(husk)
    ticks = []
    for _ in range(40):
        result = probe.abstract("move_item", source=1, destination=2)
        ticks.append(result.info.tick_after)
    end = probe.privileged.entity(husk)
    moved = abs(end["x"] - start["x"]) + abs(end["z"] - start["z"])
    check(ticks == list(range(ticks[0], ticks[0] + 40)), "ticks are not consecutive")
    check(
        moved > 0.1 or end.get("health", 20) != start.get("health", 20),
        "the husk froze",
    )
    return {"husk_moved_blocks": moved, "ticks": [ticks[0], ticks[-1]]}


def run_break_and_advancement_events(probe: Probe) -> dict[str, Any]:
    """Mine the scene's dirt block: the break arrives as a step event."""
    client = probe.client
    client.reset(12345, "flat")
    client.set_pacing("unpaced")
    probe.privileged.scene("mining")
    broken: list[int] = []
    advancements: list[str] = []
    for _ in range(60):
        result = probe.step(PlayerAction(attack=True, pitch_delta=0.0))
        broken += result.events.blocks_broken
        advancements += result.events.advancements
    dirt = probe.blocks["minecraft:dirt"]
    check(dirt in broken, f"no dirt break event: {broken}")
    observation_fields = set(result.observation.__dataclass_fields__)
    check("events" not in observation_fields, "events leaked into the observation")
    return {"blocks_broken": broken, "advancements": advancements}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=47132)
    parser.add_argument("--startup-timeout", type=float, default=240.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/minecraft-abstract-actions")
        / time.strftime("%Y%m%d-%H%M%S"),
    )
    args = parser.parse_args()
    result: dict[str, Any] = {
        "tool": "minecraft-abstract-action-probe",
        "commit": git_commit(),
        "mode": "non-recording infrastructure test",
        "host": {"system": platform.system(), "machine": platform.machine()},
    }
    try:
        with launched_client(
            args.port, 12345, args.output / "minecraft.log", args.startup_timeout
        ) as client:
            schema = client.request("SCHEMA")
            check(schema["action_schema"] == "player-action-v3", "action schema")
            probe = Probe(client)
            result["inventory_and_crafting"] = run(probe)
            result["world_keeps_moving"] = run_world_keeps_moving(probe)
            result["events"] = run_break_and_advancement_events(probe)
            result["steps"] = len(probe.steps)
        result["passed"] = True
    except Exception as error:
        result |= {"passed": False, "error": f"{type(error).__name__}: {error}"}
    write_json(args.output / "metadata.json", result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
