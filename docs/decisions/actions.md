# Action abstraction

Status: decided for the first Minecraft actor milestone. Action schema `player-action-v3`. Implemented in `minecraft/fabric-mod/src/main/java/com/mcbot/AbstractActions.java`, `InventoryMoves.java` and `src/minecraft_rl/minecraft_interface.py`. Supersedes nothing: the physical factors stay as validated in `docs/decisions/lockstep.md`.

## Principle

Physical interaction stays low-level. The agent must perceive, aim, move, attack and use. The environment never offers a world skill such as `mine_tree()`, `find_iron()` or `navigate_to_stronghold()`; those would solve the task outside the learned policy.

Only the mouse work inside GUI screens is abstracted: dragging an item between slots and clicking a recipe result. The agent still chooses which item goes where and what to craft. The environment only performs the clicks.

## Physical factors (unchanged)

| Factor | Values | No-op |
|---|---|---|
| `forward`, `back`, `left`, `right` | boolean each | false |
| `jump`, `sneak`, `sprint` | boolean each | false |
| `attack`, `use` | boolean each | false |
| `yaw_delta`, `pitch_delta` | degrees in [-45, 45] | 0 |
| `hotbar` | -1 (keep) or slot 0-8 | -1 |

`use` keeps all generic world use: eat the selected food, place the selected block, open a door, use a workstation, interact with an entity, draw a bow. No affordance flag (edible, mineable, interactable, correct tool) and no list of nearby interactable objects exists. The ray grid and the crosshair geometry are the only basis for interaction.

## Abstract factor

One optional `abstract` object per step, of one kind:

| Kind | Arguments | Vocabulary |
|---|---|---|
| `none` | - | - |
| `move_item` | `source`, `destination`: logical slots | 41 each |
| `craft` | `item`: item registry id | the item registry (1,658 in 26.3) |

Logical slots: 0-8 hotbar, 9-35 main inventory (both as in the observation `inventory_item`), 36-39 armor in the observation order feet, legs, chest, head, and 40 offhand.

A missing `abstract` field means `none`, so every `player-action-v2` action stays valid.

## Time cost

Every step is exactly one logical tick, with or without an abstract action. An abstract action never runs outside the tick timeline, so the agent cannot freeze hostile mobs with inventory work.

When the kind is not `none`, the physical part of that step is replaced by the no-op: the player stands still, as it does with the inventory screen open. The world ticks as usual. The operation runs at the end of the stepped server tick, so the observation of the step shows its result. The dataset stores the executed physical action, which is the no-op.

## `move_item(source, destination)`

The server moves items between two logical slots, as two vanilla clicks would:

| Case | Result |
|---|---|
| `source` empty, or `source == destination` | invalid |
| `destination` empty | the whole stack moves |
| same item and components, stackable | merge up to the maximum stack size; the rest stays in `source`; invalid if `destination` is full |
| different items | swap |
| an armor slot gets an item that cannot be worn there | invalid |
| an armor item with the Curse of Binding would leave its armor slot | invalid |

The offhand accepts any item. Stack splitting is not supported in `player-action-v3`.

## `craft(item)`

The server crafts one batch of the requested item from the player's inventory with the vanilla recipe system:

1. Candidate recipes: the vanilla shaped and shapeless crafting recipes whose result is `item`, in order of recipe id.
2. A recipe that fits a 2 x 2 grid works anywhere, as in the inventory screen. A larger recipe needs a crafting table: the crosshair must hit a crafting table block within the player's block interaction range. No crafting table is placed or opened for the agent.
3. The first candidate whose ingredients the inventory holds (main inventory only, not armor or offhand, and only items that vanilla accepts for crafting) runs: the ingredients are removed, the result and any leftover items (for example empty buckets) go into the inventory or drop at the player's feet if it is full.
4. Otherwise nothing changes.

One action crafts one batch: one log gives four planks.

The policy never sees recipes, ingredients, the Recipe Book, a craftable-item list or missing ingredients. It learns from experience in which states `craft(item)` changes the inventory.

## Smelting: deferred

`SMELT(item)` that gives a finished item at once is rejected: it would remove the furnace, the fuel and the waiting time. The decided contract, for a later schema version, abstracts only the furnace GUI:

- `furnace_put(source, furnace_slot)` with `furnace_slot` in {input, fuel}, and `furnace_take()`, valid only while the crosshair hits a furnace within reach.
- The furnace block entity keeps vanilla behavior: it needs valid fuel, burns it, takes vanilla time and produces its output later. The agent must come back to take it.

It is not implemented in `player-action-v3`, because the first fixed-world actor smoke test does not need it.

## Invalid actions

An invalid abstract action changes no state, still costs its tick with the no-op physical action, and gives the policy no explanation. The next observation shows that nothing useful happened. The step reply carries the exact reason (for example `missing_ingredients` or `no_crafting_table`) in the privileged `events` field, for logs and tests only. A policy never reads it.

## Actor action space

The world model and the actor use factorized discrete factors. The first actor (`docs/stage4-minecraft-actor.md`) uses:

| Factor | Classes |
|---|---|
| movement | 9 (still or 8 directions, as the exploration policy) |
| jump, sneak, sprint, attack, use | 2 each |
| yaw delta | 7: -30, -10, -3, 0, 3, 10, 30 degrees |
| pitch delta | 5: -10, -3, 0, 3, 10 degrees |
| hotbar | 10: keep or slot 0-8 |

No factor is a Cartesian product of others.

The frozen Stage 3 world model has no input for abstract actions: its action vector is the 21 physical values. The first actor therefore keeps the abstract factor at `none`. Abstract actions are implemented and tested in the environment, and their use by the actor waits for a world model whose action encoder includes them.

## Where human knowledge enters

The action vocabulary itself: which physical controls exist, the logical slot layout and the item registry as craft targets. Recipes, prerequisites and the value of an action stay learned.
