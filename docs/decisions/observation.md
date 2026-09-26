# Policy observation: structured visible field

Status: **decided for schema `visible-field-v1`; implementation and runtime validation tracked in `docs/minecraft-spike.md`.** Supersedes the earlier target of a primarily RGB/visual policy input (RGB is deferred, not rejected) and the earlier allowance of the vanilla Recipe Book as policy information.

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
| Cost | Visible-set search over a volume | `H*W` voxel traversals (measured per step, see spike ledger) | Cheap | Needs render sync and capture |
| Neural encoding | Set/attention encoder | Small conv or MLP encoder, like a depth + semantic camera | 3D conv | Conv encoder plus vision learning |
| Dreamer fit | Awkward observation-reconstruction loss over sets | Per-ray categorical and distance reconstruction losses | n/a | Standard but expensive |
| Version stability | Depends on culling details | Depends on `BlockGetter.clip` and outline shapes, stable vanilla APIs | n/a | Depends on renderer |
| Testability | Harder to assert completeness | Direct: aim at a known block, assert the ray hit, rotate away, assert absence | n/a | Hard |

The camera-ray grid is the simplest defensible choice: fixed shape, occlusion by construction, straightforward tests, and a natural Dreamer reconstruction target. Its main weakness is angular resolution: small or distant entities (dropped items, far mobs) can fall between rays. A fixed-size visible-entity slot list is the planned extension if experiments show the need; it is not implemented.

## Schema `visible-field-v1`

The observation is computed on the integrated server thread at the end of the stepped server tick, from post-step authoritative server state (player eye position and rotation after the step's client input was applied). With RGB deferred, server state defines what is visible; client rendering may interpolate entity positions differently and is not part of this contract.

### Camera rays (`H = 25` rows, `W = 33` columns, row-major, row 0 top, column 0 left)

A pinhole camera at the player's eye with vertical field of view 70 degrees (the vanilla default) and square angular pixels (horizontal field of view about 85.5 degrees). Ray `(r, c)` has camera-frame direction `x = (2(c + 0.5)/W - 1) * tan(35 deg) * W/H` (right), `y = (1 - 2(r + 0.5)/H) * tan(35 deg)` (up), `z = 1` (forward), rotated by the player's current yaw and pitch. The center ray `(12, 16)` is the crosshair direction. Each ray is traced up to `max_distance = 32` blocks; only the first hit is reported.

| Field | Shape | Type | Meaning, units, normalization |
| --- | --- | --- | --- |
| `ray_kind` | `H x W` | categorical, 4 values | `0` none within range, `1` block, `2` fluid, `3` entity. |
| `ray_type` | `H x W` | categorical | Registry index within the kind's registry: block, fluid or entity type. `0` when kind is none. Never treat as ordinal; encode with embeddings or one-hot. Vocabulary sizes come from `SCHEMA`. |
| `ray_distance` | `H x W` | continuous, float | Euclidean distance from the eye to the hit, in blocks, in `[0, 32]`; `32` when kind is none. Suggested normalization: divide by 32. |

Visibility rules: blocks use their outline shape (the same shape the crosshair targets), so glass and leaves occlude; this is conservative and never reveals more than vanilla rendering. Fluids are hit at their surface. Entities are hit by their bounding box when closer than the block hit; the player itself, spectators and entities invisible to the player are excluded. Only the entity type is exposed: no AI target, path, hidden health, UUID-derived data or status.

### Self state (always available, HUD-like)

| Field | Shape | Type | Meaning, units, normalization |
| --- | --- | --- | --- |
| `health` | scalar | continuous | Server-side player health in half-hearts, `[0, max health]` (20 by default); divide by 20. |
| `food` | scalar | integer, continuous use | Food level `[0, 20]`; divide by 20. |
| `selected_slot` | scalar | categorical, 9 values | Selected hotbar slot `0..8`. |
| `hotbar_item` | `9` | categorical | Item registry index per hotbar slot; `0` (air) when empty. |
| `hotbar_count` | `9` | integer | Stack size `0..99`; suggested normalization `log1p` or divide by 64. |
| `pitch` | scalar | continuous | Camera pitch in degrees `[-90, 90]`, positive looking down (Minecraft convention); divide by 90. Pitch relative to gravity is perceivable from the horizon, so it is egocentric. |

Absolute world coordinates and absolute yaw are not policy fields. If they are ever used, they are an explicit privileged ablation.

### GUI-visible state (mode-dependent, not implemented)

While an inventory or crafting GUI is open, a later schema version may expose inventory slots, item categorical IDs, stack counts, crafting-grid slots and the cursor stack. Recipe suggestions, recipe lists and Recipe Book contents are never exposed, even though the vanilla client has a Recipe Book internally. The exact GUI boundary will be documented and may be ablated.

## Known limitations of v1 (open, not hidden)

- Lighting is ignored: a ray reports a block in an unlit cave that a player would see as nearly black. Adding per-ray light level, or masking unlit hits, must be decided before Minecraft training.
- Underwater, rays start inside the fluid and hit it immediately.
- Block state properties (orientation, crop age) and surface normals are not exposed; only the block type.
- Small and distant entities can be missed between rays.
- Status bars other than health and food (air, armor, experience) are not yet included.

## Separation and versioning

Policy observations and privileged/debug data travel in separate protocol fields and parse into separate Python types (`PolicyObservation` versus `StepInfo` and `Debug*`). The parser rejects unknown observation fields so a protocol change cannot silently widen the policy input. A change to any field, shape, range or visibility rule bumps the schema version.
