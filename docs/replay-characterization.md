# Replay and determinism characterization

Status: **measured (Stage 1.5 gate item).** This measures what Minecraft 26.3 reproduces under identical inputs through the lockstep environment. The goal is to characterize, not to force determinism. Tool: `scripts/minecraft-replay-probe.py`; comparison logic: `src/minecraft_rl/replay.py`, with unit tests in `tests/test_replay.py`.

## Method

Each run resets to a fresh `flat` world from seed 12345. It builds the privileged `replay` debug scene at column (8, 8), with natural mob spawning off. Then it plays a fixed open-loop script of 243 `PlayerAction` steps:

- NOOP waits
- mining a dirt block with an empty hand
- two full-charge sword hits on a no-AI husk
- walking over the dropped item and jumping onto a stone step
- 180-degree turns, pitch changes, strafing
- sneaking backwards and sprinting forwards
- hotbar switches 0 → 1 → 2
- placing cobblestone with `use`
- a scripted privileged removal of a sand block's glass support before step 150, so the sand falls
- a final 60-step wait

After every step a privileged `DEBUG_TRACE` snapshot is recorded. It is kept separate from the policy record, which holds the observation, its SHA-256, `terminated`, and `reward` (null, since no reward exists yet). The snapshot holds:

- server player: position, velocity, yaw/pitch, health, food, saturation, attack strength, selected slot, full inventory
- the client's local-player copy
- every non-player entity in the scene region
- a CRC32 of all block states in the region
- a CRC32 of the client's own copy of the same region, taken at the end of the client tick

Counters that accumulate across episodes in one process (step ID, client tick) are compared relative to the first step. Wall-clock timings are excluded from comparison.

Two modes run back to back in every episode slot:

- `natural`: vanilla world randomness plus an AI pig fenced in view.
- `controlled`: no AI mob, `random_tick_speed=0` and `block_drops=false`. This mode exists only to attribute divergence to sources. It is not a training setting.

Run: 2 client processes × 3 runs × 2 modes (`runs/minecraft-replay/r2/`, run on macOS arm64 against the Stage 1.5 working tree). Every pair within a mode is compared: 15 pairs per mode, 6 same-process and 9 cross-process.

## Results

| Quantity | `controlled` (15 pairs) | `natural` (15 pairs) |
| --- | --- | --- |
| Policy observations identical at every step | 15/15 | 0/15; first difference at step 6 (pig in view, 5 pairs) or 24 (dirt item drop, 10 pairs); 9-107 of 243 steps differ, at most 36 of 825 rays |
| Player position, velocity, rotation, health, food, saturation, selected slot | identical in all pairs | identical in all pairs |
| Player inventory and attack strength | identical | inventory differs from step 33-91 in 14 pairs, attack strength from step 33 in 8 pairs (item pickup timing) |
| Husk (no AI) position and health | identical | identical |
| Region block fingerprint | identical | differs from step 160-164 in 9 pairs |
| Pig (AI) trajectory | n/a | first difference at step 5-167, up to 4.0 blocks apart |
| Tick counters (world tick before/after, relative client tick) | identical | identical |
| `terminated` | identical | identical |
| Only remaining difference | `tick_count` of the server and client player: constant offset 18-21 per run | same |

Interpretation:

- **Stable.** Lockstep transitions driven only by the player's actions reproduce exactly, within a process and across processes: movement physics, jump and step-up, sneak and sprint, camera, hotbar selection, mining duration, melee damage and cooldown, block placement, falling-sand physics, and the resulting 825-ray structured observation.
- **Not stable.** Divergence comes from vanilla `RandomSource` instances seeded from `System.nanoTime` (`RandomSupport.generateUniqueSeed`). This covers each entity's `random` (pig wandering, item-drop scatter and hence pickup timing) and the level `random` used for random block ticks. The world seed does not control them. Once a random item scatter or a mob position enters the camera field, the policy observation diverges too.
- **Player `tick_count` offset.** The offset is the number of player ticks that run between world load and the moment lockstep is armed, which varies with wall-clock loading time. It is diagnostic only, is not in the observation, and did not change any other compared field in these runs. Episodes do not start at identical internal player age.
- **No accumulation in the player trajectory.** Player-state divergence was zero at every checkpoint (25, 50, 100, 150, 200) in both modes. Divergence is confined to the RNG-driven entities and to observation rays that hit them.

## Client view of the world

Across all 12 runs, 98 server-side changes to the region fingerprint (blocks or entity count) were checked against the client copy. 74 reached the client copy one step later and 24 on the same step; none were missing. This confirms the earlier source-based inference: the client's copy of non-player world state is usually one step behind the server. The policy observation is unaffected because it is built on the server from post-step state. This matters for recording (`docs/decisions/recording.md`): a rendered frame after step N typically shows non-player world state from step N-1.

## Consequences

- Equivalence tests between paced and accelerated modes use the `controlled` scene and require exact equality of every compared field except player `tick_count`. They also compare `natural` runs by the same distributional summary, to show acceleration adds no divergence beyond this baseline.
- Exact replay of stochastic world events would need seeding of per-entity and level `RandomSource`s. That is a change to vanilla behavior and is not planned. Evaluation must therefore use held-out seeds with several episodes rather than expect bit-identical rollouts.
- The world model must treat mob behavior, drop scatter and random ticks as stochastic. That matches the Dreamer design, which has a stochastic latent state.

## Limitations

One host, one flat scene, 243 steps. Natural terrain, weather changes, day-night cycle, hostile mob AI, pathfinding over longer horizons, redstone, fluids and multi-chunk simulation were not replayed. Autosave did not trigger within the runs.
