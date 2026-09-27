# Accelerated lockstep simulation

Status: **decided requirement; not implemented.** Current throughput is ~10 steps/s (`docs/minecraft-spike.md`). Extends `docs/decisions/lockstep.md` without changing its one-action, one-tick contract. Work order: `docs/roadmap.md` (Minecraft environment track).

## Invariant

```text
one env.step(action) = exactly one Minecraft logical simulation tick
```

Acceleration removes wall-clock pacing between logical ticks (`tick N -> immediately N+1 -> immediately N+2`). It never skips, merges or batches ticks behind one policy action. Action repeat or frame skip may later be studied as an explicitly labeled RL experiment built from one-tick steps; it is never the acceleration mechanism.

Acceleration must not change physics constants, movement rates, cooldowns, entity timers, tick-rate semantics or any other gameplay rule. RL throughput is reported as environment steps and simulated ticks per wall-clock second, never as Minecraft TPS.

## Modes

- **Interactive/debug:** human-watchable pacing (current behavior).
- **Training:** unpaced, as fast as the host permits, still one exact tick per request.

Both modes must produce equivalent per-tick logical transitions for the same start state and action sequence, subject to documented Minecraft nondeterminism. Pacing is a scheduler property only and is not visible in the policy observation.

## Rendering and the training path

The policy observation is built on the server from post-step state. The training path must not require framebuffer capture, GPU rendering for perception, display refresh pacing or VSync. Normal workers still render because session recording is mandatory (`docs/decisions/recording.md`), but rendering must not pace simulation. Whether rendering can be reduced, subsampled, decoupled from the step path, or disabled for non-recording infrastructure tests must be verified on 26.3, not assumed. The visible-field sensor keeps its camera, FOV and occlusion semantics without pixels by construction, because it never reads the framebuffer.

## Established facts (26.3 Loom sources, current probes)

- The client tick is granted through the gated `DeltaTracker.Timer.advanceGameTime`, so every step waits for the next client frame (`client_wait_ms`). `Minecraft.runTick` runs at most 10 client ticks per frame before rendering, and frames are limited by `FramerateLimiter` and VSync. The render loop is on the step critical path today.
- `stepGameIfPaused` issued at `START_SERVER_TICK` takes effect on the next server loop iteration, and the stepped tick then waits for the 50 ms `waitUntilNextTick` deadline (`server_step_ms` ~55 ms).
- Vanilla `/tick sprint` (`ServerTickRateManager.requestGameToSprint`) makes `MinecraftServer` skip the wall-clock wait, but it unfreezes the game while sprinting, so it is not directly a lockstep mechanism.
- The integrated server takes its view distance from the client `renderDistance` option and its simulation distance from `simulationDistance`; the two are independent.
- `.runtime/minecraft/client/options.txt` still holds vanilla defaults (VSync on, 120 FPS cap, render distance 16, simulation distance 12, fancy graphics). The mod does not set them yet.

## Hypotheses (to test, not facts)

- Granting the server step before the tick-rate check of the same loop iteration, and skipping `waitUntilNextTick` only while a granted step is pending, removes both ~50 ms server waits.
- Driving the client tick from the step request instead of the next render frame removes `client_wait_ms`; frames can then be rendered only when the recorder needs one.

## Profiling requirement

Measure before optimizing; no target rate is predetermined. Per-step latency is decomposed at least into: client tick, packet synchronization (client tick end until the server consumes it), server tick work, structured observation construction, IPC (Python round trip minus server total, including JSON encode and parse), and wall-clock scheduler waiting. Current `server_step_ms` and `client_wait_ms` mix work with waiting and must be split. Reports include simulated ticks per wall-clock second, environment steps per second, observation construction latency and IPC latency.

## Equivalence tests

Repeatable tests start paced and accelerated runs from identical conditions (fresh world from a fixed seed and preset, the same scripted scene and player state) and replay a predetermined action sequence. They compare per tick: player position, rotation, health and inventory, the structured observation, and scripted-scene entity state. Differences must be zero or explained by characterized nondeterminism; mob AI already diverged between same-seed paced runs, so determinism characterization comes first. The v1 and v2 probes must keep passing in both modes.
