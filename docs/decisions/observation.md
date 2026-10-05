# Policy observation: structured visible field

Status: **decided for schema `visible-field-v2`; implemented over protocol v2 and runtime-verified for a scripted occlusion scene and a scripted self-state scene (see `docs/minecraft-spike.md` and `scripts/minecraft-observation-probe.py`).** `visible-field-v2` supersedes `visible-field-v1`. It keeps the camera rays unchanged and replaces the hotbar-only self state with the full self state that a player can read from the HUD and the inventory screen. Earlier decisions: this ADR superseded the target of a primarily RGB/visual policy input (RGB is deferred, not rejected) and the allowance of the vanilla Recipe Book as policy information.

## Principle

Remove the computer-vision problem without giving the agent omniscient world state. The policy receives, in machine-readable form, approximately what the player can currently perceive from its viewpoint:

```text
current camera view -> structured visible-world sensor -> agent        (chosen)
chunk / radius cube -> all block states -> agent                       (forbidden)
```

A block or entity behind the player, outside the camera field of view, or hidden behind an opaque surface never appears merely because it is nearby. The Fabric environment may inspect complete game state to implement and test visibility; the policy never receives that hidden state. There is no persistent environment-provided map: remembering what is no longer visible is the job of the learned recurrent state.

## Alternatives compared

| Criterion | A. Sparse visible-surface list | B. Camera-ray grid (chosen) | Rejected: local voxel cube | Deferred: RGB |
| --- | --- | --- | --- | --- |
| Hidden-information leakage | Needs a separate per-element visibility test (face culling plus line of sight); easy to get subtly wrong and leak occluded faces | First hit per ray is visible by construction; nothing behind it is reported | Leaks everything in range by design | None beyond vanilla |
| Tensor shape | Variable length; needs padding, masks and ordering rules | Fixed `H x W` per channel | Fixed | Fixed |
| Cost | Visible-set search over a volume | `H*W` voxel traversals; measured 0.7-0.8 ms per step for 825 rays including JSON | Cheap | Needs render sync and capture |
| Neural encoding | Set/attention encoder | Small conv or MLP encoder, like a depth + semantic camera | 3D conv | Conv encoder plus vision learning |
| Dreamer fit | Awkward observation-reconstruction loss over sets | Per-ray categorical and distance reconstruction losses | n/a | Standard but expensive |
| Version stability | Depends on culling details | Depends on `BlockGetter.clip` and outline shapes, stable vanilla APIs | n/a | Depends on renderer |
| Testability | Harder to assert completeness | Direct: aim at a known block, assert the ray hit, rotate away, assert absence | n/a | Hard |

The camera-ray grid is the simplest defensible choice: fixed shape, occlusion by construction, straightforward tests, and a natural Dreamer reconstruction target. Its main weakness is angular resolution: small or distant entities (dropped items, far mobs) can fall between rays. A fixed-size visible-entity slot list is the planned extension if experiments show the need; it is not implemented.

## Schema `visible-field-v2`

The observation is computed on the integrated server thread at the end of the stepped server tick, from post-step authoritative server state (player eye position and rotation after the step's client input was applied). With RGB deferred, server state defines what is visible; client rendering may interpolate entity positions differently and is not part of this contract.

### Camera rays (`H = 25` rows, `W = 33` columns, row-major, row 0 top, column 0 left)

A pinhole camera at the player's eye with vertical field of view 70 degrees (the vanilla default) and square angular pixels (horizontal field of view about 85.5 degrees). Ray `(r, c)` has camera-frame direction `x = (2(c + 0.5)/W - 1) * tan(35 deg) * W/H` (right), `y = (1 - 2(r + 0.5)/H) * tan(35 deg)` (up), `z = 1` (forward), rotated by the player's current yaw and pitch. The center ray `(12, 16)` is the crosshair direction. Each ray is traced up to `max_distance = 32` blocks; only the first hit is reported.

| Field | Shape | Type | Meaning, units, normalization |
| --- | --- | --- | --- |
| `ray_kind` | `H x W` | categorical, 4 values | `0` none within range, `1` block, `2` fluid, `3` entity. |
| `ray_type` | `H x W` | categorical | Registry index within the kind's registry: block, fluid or entity type. `0` when kind is none. Never treat as ordinal; encode with embeddings or one-hot. Vocabulary sizes come from `SCHEMA`. |
| `ray_distance` | `H x W` | continuous, float | Euclidean distance from the eye to the hit, in blocks, in `[0, 32]`; `32` when kind is none. Suggested normalization: divide by 32. |

Visibility rules: blocks use their outline shape (the same shape the crosshair targets), so glass and leaves occlude; this is conservative and never reveals more than vanilla rendering. Blocks with an invisible render shape (air, barrier, light, structure void) are skipped unless they carry a block entity; infested blocks report their host block type, because that is what a player sees. Unloaded chunks end the ray as none. Fluids are hit at their surface. Entities are hit by their bounding box when closer than the block hit; the player itself, spectators, markers, interaction entities and entities invisible to the player are excluded. Only the entity type is exposed: no AI target, path, hidden health, UUID-derived data or status.

### Self state (always available, HUD-like)

Every self-state field is something that a player can read from the HUD or from the inventory screen. Where the screen shows less than the engine knows, the field has the resolution of the screen. Inventory slots `0..8` are the hotbar, so there are no separate hotbar fields. Python exposes `hotbar_item`, `hotbar_count` and `main_hand_item` as views of the inventory fields.

| Field | Shape | Type | Meaning, units, normalization |
| --- | --- | --- | --- |
| `health` | scalar | continuous | Server-side player health in half-hearts, `[0, max_health]`. Divide by 20. |
| `max_health` | scalar | continuous | Maximum health in half-hearts, 20 by default. The HUD shows it as the number of heart containers. Divide by 20. |
| `absorption` | scalar | continuous | Absorption health in half-hearts (the golden hearts). Divide by 20. |
| `food` | scalar | integer | Food level `[0, 20]`. Divide by 20. Saturation is not exposed, because the HUD does not show it. |
| `air_bubbles` | scalar | integer | The air bubbles that the HUD draws, `[0, 10]`, 10 at full air, rounded up. Divide by 10. |
| `armor` | scalar | integer | Armor points that the HUD draws, `[0, 20]` in vanilla. Divide by 20. |
| `xp_level` | scalar | integer | Experience level, `>= 0`. Suggested encoding `log1p(level) / log1p(30)`. |
| `xp_progress` | scalar | continuous | Fill of the experience bar, `[0, 1]`. |
| `selected_slot` | scalar | categorical, 9 values | Selected hotbar slot `0..8`. |
| `inventory_item` | `36` | categorical | Item registry index per main-inventory slot. Slots `0..8` are the hotbar. `0` (air) when empty. |
| `inventory_count` | `36` | integer | Stack size per slot, `0` when empty. Suggested normalization `log1p(count) / log1p(64)`. |
| `inventory_durability` | `36` | continuous | The durability bar of the item, `k / 13` for `k` in `0..13`. It is `1.0` when the item takes no damage or has no bar. |
| `armor_item` | `4` | categorical | Item registry index of the feet, legs, chest and head slot, in this order. |
| `armor_durability` | `4` | continuous | Durability bar of each armor slot, as for the inventory. |
| `offhand_item`, `offhand_count`, `offhand_durability` | scalar | as above | The offhand slot. |
| `effect_type` | `8` | categorical | Active effects with a HUD icon, in registry order, in the first slots. The value is the effect registry index plus 1. `0` marks an empty slot. |
| `effect_amplifier` | `8` | integer | Effect level minus 1 (the vanilla amplifier), `0` in an empty slot. |
| `effect_seconds` | `8` | integer | Remaining duration in seconds, rounded up as the inventory screen shows it. `-1` for an infinite effect. `0` in an empty slot. Suggested normalization `log1p(seconds) / log1p(600)` with a separate flag for `-1`. |
| `pitch` | scalar | continuous | Camera pitch in degrees `[-90, 90]`, positive looking down (Minecraft convention). Divide by 90. Pitch relative to gravity is perceivable from the horizon, so it is egocentric. |

Not exposed, because an ordinary player cannot read it: saturation, exhaustion, exact air ticks, exact item damage values, item NBT and components beyond the item type, effects without a HUD icon, and the internal experience point total. Item, block, fluid, entity and effect ids are categorical. Never treat them as ordered numbers. Encode them with embeddings or one-hot vectors. The parser checks that every id lies inside its vocabulary, that empty effect slots report zeros, and that effects fill the first slots in registry order.

Absolute world coordinates and absolute yaw are not policy fields. If they are ever used, they are an explicit privileged ablation.

### GUI-visible state (mode-dependent, not implemented)

While an inventory or crafting GUI is open, a later schema version may expose inventory slots, item categorical IDs, stack counts, crafting-grid slots and the cursor stack. Recipe suggestions, recipe lists and Recipe Book contents are never exposed, even though the vanilla client has a Recipe Book internally. The exact GUI boundary will be documented and may be ablated.

## Known limitations of v2 (open, not hidden)

- Lighting is ignored: a ray reports a block in an unlit cave that a player would see as nearly black. Adding per-ray light level, or masking unlit hits, must be decided before Minecraft training.
- Underwater, rays start inside the fluid and hit it immediately.
- Block state properties (orientation, crop age) and surface normals are not exposed; only the block type.
- Small and distant entities can be missed between rays.
- More than 8 effects with an icon: the effects after the eighth in registry order are dropped. Vanilla play rarely has more than 8.
- The armor points, the air bubbles and the durability bars are HUD resolutions. A player who opens the item tooltip can read more exact durability. v2 does not model tooltips.

## Separation and versioning

Policy observations and privileged/debug data travel in separate protocol commands and parse into separate Python types: `PolicyObservation` (in `minecraft_rl.minecraft_interface`) versus `StepInfo`, `ResetInfo` and the `PrivilegedProbe` debug commands (in `minecraft_rl.minecraft_client`). The parser rejects missing or unknown observation fields, wrong types, out-of-range categorical IDs and distances, so a protocol change cannot silently widen the policy input. A change to any field, shape, range or visibility rule bumps the schema version. `v2 SCHEMA` reports every layout size (`inventory_slots`, `armor_slots`, `effect_slots`, `air_bubbles`, `durability_steps`) and the vocabulary sizes, including `effect_types`. The Python client refuses a server whose layout differs from its own. Vocabulary sizes come from `v2 SCHEMA`; human-readable registry names come only from privileged `DEBUG_REGISTRY` for tests and are never a policy input.
