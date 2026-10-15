# Accelerated lockstep simulation

Status: **implemented and runtime-validated on 26.3 (macOS arm64).** Paced ~20 steps/s; unpaced with rendering ~58-99 steps/s and unpaced without rendering ~140-154 steps/s, varying between sessions on the same host (see Results). Paced and unpaced traces are equivalent in the controlled scene. Extends `docs/decisions/lockstep.md` without changing its one-action, one-tick contract. Work order: `docs/roadmap.md` (Minecraft environment track).

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

## Established facts (26.3 Loom sources, probes before acceleration)

- The client tick is granted through the gated `DeltaTracker.Timer.advanceGameTime`, so every step waits for the next client frame (`client_wait_ms`). `Minecraft.runTick` runs at most 10 client ticks per frame before rendering, and frames are limited by `FramerateLimiter` and VSync.
- `stepGameIfPaused` issued at `START_SERVER_TICK` takes effect on the next server loop iteration, and the stepped tick then waits for the 50 ms `waitUntilNextTick` deadline.
- Vanilla `/tick sprint` (`ServerTickRateManager.requestGameToSprint`) makes `MinecraftServer` skip the wall-clock wait, but it unfreezes the game while sprinting, so it is not a lockstep mechanism and is not used.
- The integrated server takes its view distance from the client `renderDistance` option and its simulation distance from `simulationDistance`; the two are independent.
- A privileged edit runs as a server task, which can execute after that iteration's chunk broadcast (`ServerChunkCache.broadcastChangedChunks`); its block update is then only sent at the end of the next server iteration.

## Implementation

`Pacing.PACED` keeps vanilla scheduling; `Pacing.UNPACED` removes wall-clock waiting only. Selected by `MCBOT_PACING` at launch or the v2 `PACING` command between steps.

- **Same-iteration grant.** A ready step is granted by a hook just before `ServerTickRateManager.tick()` in `MinecraftServer.tickServer`, so the stepped tick is that iteration, not the next one (both modes).
- **No server deadline wait (unpaced).** While a granted step, an ungranted finished client tick, an unsent marker or an unprocessed client tick-end packet is pending, `MinecraftServer.haveTime` reports no remaining time and the deadline restarts from now. Threads are woken with `LockSupport.unpark` when such work appears. Idle frozen iterations keep vanilla pacing.
- **No frame wait (unpaced).** The `FramerateLimiter` call in `renderFrame` returns as soon as a granted client tick is ready.
- **Ordering barrier (both modes).** After a completed step, and after a world-mutating `DEBUG_*` command, the server sends a `ClientboundPingPacket` marker at the end of an iteration that has already broadcast the relevant changes: the stepped iteration itself for a step, the next iteration for a privileged edit. The client starts its next granted tick only once that marker is queued on its side. Without the second rule, paced mode missed privileged edits in the next client tick (0/12) while unpaced mode saw them (12/12); with it, both modes see 12/12.
- **No sound.** In observer mode the client sets the master volume to 0 at start and before each world (`EnvironmentSettings.mute`). The policy never hears sound. `MCBOT_SOUND=on` keeps the volumes of the options file. `SCHEMA` reports `client_settings.master_volume`.
- **Optional no-render stepping.** `MCBOT_RENDER=off` (or `render_frames=false`) skips `renderFrame` between steps in unpaced mode. Client ticks still run. This is only for non-recording infrastructure benchmarks; recorded sessions render.
- **Low-cost client settings** (`EnvironmentSettings`, applied in observer mode and reported by `STATUS`): fast graphics preset, render distance 6 (was 8, see "Render distance"; `MCBOT_RENDER_DISTANCE` overrides it for a benchmark), simulation distance 6, VSync off, frame limit 260, minimal particles, clouds and entity shadows off, no pause on lost focus.

## Results (macOS arm64, seed 12345)

`scripts/minecraft-observation-probe.py` (`runs/minecraft-observation/final-{paced,unpaced}.json`), 200 steps per phase, medians in ms:

| Phase | Steps/s | client wait | client tick | tick-end sync | server tick | observation | IPC | round trip |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paced | 20.0 | 6.8 | 1.58 | 35.7 | 3.46 | 0.71 | 0.61 | 49.7 |
| unpaced, rendering | 83.8 | 4.07 | 1.42 | 0.15 | 2.91 | 0.67 | 0.56 | 10.2 |
| unpaced, no rendering | 153.5 | 0.09 | 1.52 | 0.11 | 2.62 | 0.70 | 0.52 | 5.8 |
| paced again | 20.0 | 6.3 | 1.34 | 36.2 | 2.97 | 0.71 | 0.55 | 49.6 |

In paced mode the step waits for the 50 ms server deadline (`tick_end_sync_ms`). With rendering, unpaced steps are bounded by frame rendering (`client_wait_ms`); without rendering the step is dominated by server tick work, the client tick and observation construction. Visibility, combat, mining, reset and exact-step checks pass in both modes, and the v1 tick-gate probe still passes.

`scripts/minecraft-equivalence-probe.py` (`runs/minecraft-equivalence/e4/`): 2 client processes, 2 runs per process and mode, 243 fixed replay steps per run, paced and unpaced alternating inside each process. Passed.

- Controlled scene, 16 paced-vs-unpaced pairs: zero unexplained differences in the policy observation, server and client player state, inventory, region block fingerprint, client world copy and scene entities. Mining, combat, placement and falling-sand milestones are tick-identical (husk hit on steps 58 and 79, block changes on steps 24, 150, 151, 163, 178). The only difference is the characterized per-run player `tick_count` offset.
- Client world copy: of 5 server block changes per controlled run, 3 appear in the client copy on the same step and 2 one step later, identically in every paced and unpaced run.
- Natural scene: pig AI, item drops and inventory diverge in every pair, paced-vs-paced as well as paced-vs-unpaced, as characterized in `docs/replay-characterization.md`. Player positions stay identical (maximum distance 0.0).
- Every run keeps one world tick and one client tick per STEP and does not advance while idle for 2 s.
- Throughput in this probe: paced 20.4, unpaced (rendering) 98.5 steps/s.

These results come from one host, one flat scene and one action script. Transparent blocks, fluids, lighting, natural terrain, many entities and longer episodes are not yet covered.

Run-to-run variation: later sessions on the same host measured unpaced with rendering at 57-68 steps/s and without rendering at 140-145 steps/s (`runs/minecraft-observation/after-recording.json`, `runs/minecraft-equivalence/e5/`). An A/B build with and without the recording frame hook gave 59.6 and 57.5 steps/s, so the spread is host variation, not a code regression. Paced stayed at 20 steps/s and every correctness check passed in all sessions. Throughput comparisons should therefore be made within one session.

## Profiling requirement

Measure before optimizing; no target rate is predetermined. Per-step latency is decomposed at least into: client tick, packet synchronization (client tick end until the server consumes it), server tick work, structured observation construction, IPC (Python round trip minus server total, including JSON encode and parse), and wall-clock scheduler waiting. The v2 STEP reply reports `dispatch_ms`, `client_wait_ms`, `client_tick_ms`, `tick_end_sync_ms`, `server_wait_ms`, `server_tick_ms`, `observation_ms`, `encode_ms` and `server_total_ms`; Python adds `round_trip_ms` and `ipc_ms`. Reports include simulated ticks per wall-clock second, environment steps per second, observation construction latency and IPC latency.

## Equivalence tests

Repeatable tests start paced and accelerated runs from identical conditions (fresh world from a fixed seed and preset, the same scripted scene and player state) and replay a predetermined action sequence. They compare per tick: player position, rotation, health and inventory, the structured observation, and scripted-scene entity state. Differences must be zero or explained by characterized nondeterminism; mob AI already diverged between same-seed paced runs, so determinism characterization comes first. The v1 and v2 probes must keep passing in both modes.

## Render distance

Decision: render distance 6 chunks, the lowest value that keeps environment semantics. It supersedes 8.

The integrated server takes its view distance from the client render distance, so this setting also decides which chunks the server keeps loaded and how far chunk-ticking reaches. It is therefore a simulation setting, not only a rendering setting. Two conditions must hold:

1. Every chunk that a 32-block sensor ray can reach is loaded on the server. Otherwise `VisibleFieldSensor` returns a miss for blocks that exist.
2. Every chunk inside the simulation distance (6) becomes entity-ticking. Otherwise mobs, items and natural spawning stop near the player, and the world changes.

Method: `scripts/minecraft-render-distance-probe.py`, one fresh client process per render distance, macOS arm64, seed 12345. The privileged command `DEBUG_CHUNKS` reports the chunks within 32 blocks of the eye and the chunks inside the simulation distance, with their load and entity-ticking state. It is read-only and never part of a policy observation.

- Controlled replay scene (flat world, 243 fixed steps, unpaced): the trace of every render distance is compared with that of render distance 8.
- Walk (`normal` world, seed 12345, 420 fixed steps of sprinting with turns and jumps, unpaced, twice per process): chunk state after every step.
- Throughput: the same walk with rendering, once without and once with recording (426 x 240, every tick), timed. CPU time and resident memory of the game process come from `ps`, its energy from `proc_pid_rusage`.

| Render distance | 8 | 6 | 4 |
|---|---|---|---|
| Controlled replay: policy steps differing from 8 | 0 | 0 | 0 |
| Controlled replay: unexplained trace fields | none | none | none |
| Walk: steps with an unloaded chunk in sensor range | 0, 0 | 0, 0 | 0, 0 |
| Walk: steps with a non-ticking chunk in sensor range | 0, 0 | 0, 0 | 0, 0 |
| Walk: simulation chunks entity-ticking at the end | 169 / 169 | 169 / 169 | 81 / 169 |
| Walk: mean server loaded chunks | 1,699, 1,634 | 1,466, 1,472 | 1,220, 1,218 |
| Walk: loaded entities at the end | 303, 423 | 249, 412 | 177, 178 |
| Steps/s, rendering, no recording | 119.0 | 191.6 | 120.8 |
| Steps/s, rendering and recording | 118.1 | 142.4 | 116.7 |
| Median step, recording (ms) | 8.16 | 7.50 | 8.21 |
| Game CPU per step, recording (ms) | 34.4 | 20.9 | 25.5 |
| Game energy per step, recording (J) | 0.117 | 0.076 | 0.072 |
| Game resident memory (MiB) | 3,132 | 2,736 | 2,179 |
| Recording: frames written / due, dropped | 421 / 421, 0 | 421 / 421, 0 | 421 / 421, 0 |
| Client start to first world (s) | 32.2 | 14.1 | 14.1 |

Every step advanced exactly one tick at every distance.

Interpretation:

- The sensor range is always loaded: 32 blocks is two chunks, and even render distance 4 keeps them.
- At render distance 4 the view distance (4) is smaller than the simulation distance (6). Only 81 of the 169 simulation chunks (a 9 x 9 square) became entity-ticking, so fewer entities loaded (177 against 303-423). That changes the simulated world near the player, so render distance 4 is rejected even though the sensor itself was unaffected.
- At render distance 6 all 169 simulation chunks tick, as at 8.
- Throughput is noisy between processes: one rendering pass at distance 6 reached 191.6 steps/s. Game CPU per step and energy per step are lower at 6 and 4 than at 8 (20.9 and 25.5 against 34.4 ms; 0.076 and 0.072 against 0.117 J), and memory falls by 400 MiB at 6. The first measurement at 8 also started more slowly (32.2 against 14.1 s), probably a cold cache, so the startup times are not compared.
- Walk ray grids are not compared between render distances: two resets of the same `normal` seed already differ from the first step in one process, because the spawn point varies. The controlled replay is the exact comparison.

Limitations: one host, one run per distance, one seed. The throughput numbers come from one 420-step walk each.

After the change, the observation, paced/unpaced equivalence and v1 tick-gate probes pass with render distance 6.

### Worker regression at render distance 6

`scripts/minecraft-worker-probe.py` passes with the default render distance 6 (`runs/minecraft-workers/20261013-211912`): two concurrent workers keep their own identities, directories and ports, stepping one worker never advances the other, the concurrent controlled replay of 243 steps is identical on both workers apart from the known `tick_count` offset, and each worker records its own video with no dropped frame. Concurrent replay throughput was 136 and 138 steps/s.

Decision: render distance 6 is the training default. Evaluation, showcase and benchmark runs can set another value with `MCBOT_RENDER_DISTANCE`. Keep it at least the simulation distance (6).

### The game process on the efficiency cores

`launched_client(..., background=True)` starts Gradle and the game under `taskpolicy -b`, so the game inherits the macOS background policy. The render-distance probe takes `--background`. Measured at render distance 6, one process, same walk:

| | Default priority | Background policy |
|---|---|---|
| Steps/s, rendering, no recording | 191.6 | 38.8 |
| Steps/s, rendering and recording | 142.4 | 45.2 |
| Game energy per step, recording (J) | 0.076 | 0.036 |
| Game mean power, recording (W) | 10.8 | 1.6 |
| Game energy on performance cores | - | 0 J |
| Controlled replay against render distance 8 | identical | identical |

The background policy cuts the game power by a factor of about 7 and the energy per step by half, at a quarter of the speed.

Negative finding: chunk generation then falls behind lockstep. In the first walk after a reset, no chunk inside the simulation distance was entity-ticking at the end and 320 of 420 steps had an unloaded chunk in sensor range; the second walk had 17 steps with a non-ticking chunk in sensor range. With default priority no step had an unloaded or non-ticking sensor chunk. Chunks keep loading while the world is frozen (a separate check: 9 to 169 ticking chunks in 10 s of idle time after a reset on the efficiency cores). The background policy is therefore not a training default until a reset waits for its chunks and a new measurement shows that sensor-range chunks stay ready during movement.

### Chunk gate for the background policy

`src/minecraft_rl/chunk_gate.py` and `minecraft_play --background-game` (macOS) address the negative finding above. The game runs with the background policy. Before the first observation of an episode and before every step, a privileged `DEBUG_CHUNKS` query checks that every sensor-range chunk is loaded and entity-ticking and every simulation chunk is entity-ticking. If not, the gate waits while the world is frozen (no tick runs, chunk loading continues); after 0.5 s it moves the game to default priority until the chunks are ready, then back. The policy never sees the query, and a step only runs once its chunks are ready, so the gate never changes what a step does.

Measured once, on the Stage 4 actor-retry evaluation (`docs/stage4-reward-actor-data.md`, 4 episodes, 4,000 steps, recording on): 73 waits in 4,004 checks, 42 s waiting, longest 6.4 s, 11 boosts for 22 s; game energy 153 J (0.038 J per step, 0.1 J on the performance cores) in 195 s (20.5 steps per second). Not yet compared with a default-priority run of the same episodes, and not yet a training default.


### Sodium and Sodium Extra

Question: do Sodium 0.9.2 and Sodium Extra 0.9.4 with Reduce Resolution on macOS lower the game's energy per step without hurting throughput or correctness?

Answer: **no. The default stays vanilla.** Reduce Resolution works (the framebuffer falls from 1708 x 960 to 854 x 480), but both mods raise the game's CPU energy per step by about a quarter. The GPU was not the main cost: the game's heat is CPU work, and Sodium's chunk meshing threads add CPU work.

Setup: `minecraft/fabric-mod/build.gradle` adds the two mods as Loom `localRuntime` dependencies from the Modrinth Maven repository, never bundled into the mcbot jar (the unobfuscated Loom has no `modLocalRuntime`; `localRuntime` is its equivalent). Versions are pinned in `gradle.properties`:

| Mod | Version | Maven coordinate | File |
|---|---|---|---|
| Sodium | mc26.3-0.9.2-fabric | `maven.modrinth:AANobbMI:bAZQdGpg` | sodium-fabric-0.9.2+mc26.3.jar |
| Sodium Extra | mc26.3-0.9.4+fabric | `maven.modrinth:PtjYWJkn:te2y9qZn` | sodium-extra-fabric-0.9.4+mc26.3.jar |

One option selects the renderer: `MCBOT_RENDERER` (`vanilla`, the default; `sodium`; `sodium-extra`), passed by `launched_client` to Gradle as `-Pmcbot.renderer`. With `sodium-extra`, `launched_client` first writes `extra_settings.reduce_resolution_on_mac = true` into `sodium-extra-options.json` of the project-owned game directory (`.runtime/minecraft/client`, or the worker's own directory); other settings keep their values. `STATUS` reports the renderer, which mods loaded, the logical window size and the framebuffer size.

Measured framebuffer (window 854 x 480 points on the Retina display):

| Renderer | Framebuffer | Pixel density | Recording output |
|---|---|---|---|
| vanilla | 1708 x 960 | 2 | 426 x 240 |
| sodium | 1708 x 960 | 2 | 426 x 240 |
| sodium-extra | 854 x 480 | 1 | 426 x 240 |

Benchmark: `scripts/minecraft-render-distance-probe.py --render-distance 6 --renderer R`, normal priority, render distance 6, simulation distance 6, the same 420-step sprint walk (seed 12345, as for the render-distance benchmark), the same window, recording 426 x 240 every tick. Three passes per renderer in the order vanilla, sodium, sodium-extra and back (`runs/minecraft-renderer/first`, `second`, and the top level); means and ranges:

| | vanilla | sodium | sodium-extra |
|---|---|---|---|
| Game CPU energy per step, rendering (J) | 0.113 (0.111-0.116) | 0.143 (0.136-0.147) | 0.139 (0.129-0.145) |
| Game CPU energy per step, recording (J) | 0.109 | 0.137 | 0.135 |
| Steps/s, rendering | 86.7 (81.4-93.6) | 96.8 (84.4-104.9) | 89.8 (82.6-102.0) |
| Steps/s, recording | 83.8 | 90.0 | 89.5 |
| Game CPU per step (ms) | 28.9 | 36.4 | 36.0 |
| Game CPU cores in use | 2.5 | 3.5 | 3.2 |
| Game resident memory (MiB) | 2,176 | 2,606 | 2,667 |
| Recording: frames written / due, dropped | 421 / 421, 0 | 421 / 421, 0 | 421 / 421, 0 |
| Steps with an unloaded sensor chunk, 6 walks | 0 | 540 (in 2 walks) | 0 |

Against vanilla, energy per step: Sodium +27% (rendering) and +26% (recording); Sodium Extra +23% and +24%. Steps per second: Sodium +12% and +7%; Sodium Extra +4% and +7%, inside the run-to-run noise (vanilla alone spans 81-94).

GPU: per-process GPU time from `powermetrics --show-process-gpu` read 0 for the game, so only system-wide GPU power was available (`sudo -n powermetrics`, recorded by the probe's `PowerSampler`). It moves with other programs on the machine (5-8 W idle before different passes) and showed no consistent difference between renderers; it is reported per run, not used for the decision.

Correctness, all with no regression:

- The controlled replay trace is identical to vanilla for both mods (0 differing policy steps, no unexplained field).
- The tick-gate, observation, equivalence, recording, worker and abstract-action probes pass with the default and with `MCBOT_RENDERER=sodium-extra`; each worker got its own Sodium Extra config.
- Recording: no dropped frame; videos are 426 x 240 (aspect 1.775), no black frame, the same scene brightness by thirds of the image at the same times as vanilla. Frames were checked as image statistics, not watched.
- Sodium alone left sensor-range chunks unloaded in 2 of 6 walks (302 and 238 of 420 steps). Sodium Extra and vanilla had none. The cause was not investigated, because Sodium is not selected.

Background policy, one pass each (`--background`): vanilla 43.7 steps/s at 0.032-0.036 J per step; sodium-extra 66.6 steps/s at 0.013 J per step without recording but 43.7 steps/s at 0.044 J per step with recording. One pass with this spread does not support a change.

Decision: vanilla stays the default renderer. `MCBOT_RENDERER=sodium-extra` remains available for viewing at a lower framebuffer, and the dependency stays pinned. No other mod was tried.

### Recorded frames only

Question: in unpaced lockstep, the render thread draws as many frames as it can (frame limit 260), but the recording keeps one frame every 2 ticks. Can the client draw only the frames that the recording keeps, and does that lower the cost?

Answer: **yes, and it is now the default of `minecraft_life`.** The mode never changes a step, the recording stays complete, and the game draws less than half the frames. The energy gain is small on the efficiency cores and clearer at default priority. It is not a heat solution on its own.

Mechanism: the PACING command takes `recorded_frames_only` (Python `set_pacing(..., recorded_frames_only=True)`, or `MCBOT_RENDER=recorded` at launch). In unpaced mode, `Lockstep.skipFrame` then skips `renderFrame` unless `FrameRecorder.needsFrame()`: the first frame after a client tick that the recording keeps, and every frame while a requested stop drains. With no recording, no frame is drawn. Client ticks still run, so a step does the same work. `STATUS` reports `frames_drawn` and `frames_skipped`. `minecraft_life` uses the mode and the efficiency cores by default; `--draw-every-frame` and `--foreground-game` turn them off.

Correctness (`scripts/minecraft-frame-probe.py`, and the existing probes):

- The controlled replay trace with the mode equals the trace without it: 0 of 243 policy steps differ, no unexplained field.
- Every recorded walk: 211 of 211 frames due were captured and sent, 0 missed, 0 dropped, video complete. A 300-tick `minecraft_life` smoke run: 3 segments, 51 of 51 frames each, complete.
- The tick-gate, equivalence and recording probes pass.

Measured, the 420-step sprint walk with recording every 2 ticks, one warm-up walk, then 3 walks per mode interleaved (`runs/minecraft-frames/`):

| | Frames drawn per step | Steps/s | Game J per step | Game CPU ms per step |
|---|---|---|---|---|
| Default priority, every frame | 1.07 | 87 | 0.112 | 27.0 |
| Default priority, recorded frames only | 0.50 | 118 | 0.102 | 23.6 |
| Efficiency cores, every frame | 1.22 | 41 | 0.032 | 69.0 |
| Efficiency cores, recorded frames only | 0.50 | 48 | 0.031 | 64.0 |

Without recording (no frame at all, 3 walks each on the efficiency cores, `/tmp` script, not kept): 0.025-0.036 J per step drawing every frame against 0.014-0.025 drawing none, at 36-39 against 57-90 steps/s.

Die temperature, one 60-second run per row in one client process, `CpuTemperature`, idle 56 C before (ordered, so later rows start warmer):

| | Steps/s | Game W | Die mean / max |
|---|---|---|---|
| Efficiency cores, every frame | 57 | 0.58 | 56 / 57 C |
| Efficiency cores, no frames | 76 | 0.59 | 58 / 62 C |
| Default priority, every frame | 59 | 0.53 | 63 / 68 C |
| Default priority, no frames | 216 | 0.55 | 64 / 74 C |

Reading: the game's power stays nearly the same, about 0.5-1.5 W on the efficiency cores. Fewer frames make each step cheaper, so the game runs more steps in the same power. Work per step falls, but heat per second does not. The efficiency cores are what keep the die near idle temperature. Heat per second falls only with fewer steps per second, which is the pause the duty cycle already gives, or with less work per step. These are single measurements on one machine with other programs running.
