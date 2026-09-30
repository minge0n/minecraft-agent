# Minecraft 26.3 lockstep engineering spike

**Scope:** No agent training, neural network or reward. This spike asks whether an external controller can pause Minecraft's logical simulation, advance it by exactly one tick per request with a player action applied on that tick, and return a structured visible-field observation of the result. Stage 1.5 is **complete**: every gate requirement below is validated, with the limitations listed at the end of this file.

## Stage 1.5 completion gate

Revised by the observation architecture update (`docs/decisions/observation.md`): structured-observation synchronization replaces RGB framebuffer freshness, which is deferred together with RGB.

| Requirement | Status |
| --- | --- |
| Exact world/client/player one-tick stepping | proven (v1 probe 3 runs; v1 regression after v2 changes) |
| No wall-clock simulation progress while idle | proven (v1 probe; v2 observation unchanged over 3 s idle) |
| Physics and hostile-mob progression only on steps | proven (v1 probe) |
| Same-step action application | proven for v2 factorized yaw, pitch, forward, attack, hotbar |
| Basic combat timing | proven: first attack damages on its step; held attack does not re-hit; see below |
| Structured visible observation | proven: `visible-field-v1` over v2 |
| Observation/action tick alignment | proven: post-step yaw/pitch and positions appear in the same step's observation |
| Visibility and occlusion boundary, hidden-information audit | proven for the scripted scene (4 runs); see limitations |
| Episode reset semantics for experiments | proven: fresh disposable world per reset, gated and frozen, `terminated` after death |
| Replay/determinism characterization | measured: player-driven transitions and policy observations bit-identical across runs and processes once world RNG consumers are removed; mob AI, drop scatter and random ticks diverge (`docs/replay-characterization.md`) |
| Protocol and schema correctness | proven: v1 and v2 unit tests, strict schema parsing, JUnit for actions and ray geometry |
| Measured throughput with per-phase breakdown | measured: paced ~20 steps/s; unpaced ~58-99 with rendering and ~140-154 without, per-phase breakdown in `docs/decisions/simulation-throughput.md` |
| Accelerated stepping validated against paced | proven: 16 controlled paced/unpaced replay pairs with no unexplained differences; one tick per STEP and idle freeze in both modes (`docs/decisions/simulation-throughput.md`) |
| Session recording | proven for one worker: recorded and unrecorded replays identical, every due frame written, recorder failure isolated, cost benchmarked (`docs/decisions/recording.md`) |
| Isolated workers with deterministic identities | proven for two concurrent workers: separate game directories, `AgentNNNN` offline identities, ports, seeds and recordings; no cross-worker stepping or interference (`docs/decisions/parallel-workers.md`) |
| RGB framebuffer synchronization | deferred, not required |

## Automated runtime probe

```sh
.venv/bin/python scripts/minecraft-tick-gate-probe.py              # run checks, then quit
.venv/bin/python scripts/minecraft-tick-gate-probe.py --keep-open  # keep stepping for watching
```

The script launches the Loom `runClient` configuration through `./scripts/gradle` (project-local JDK 25, `.runtime/` game data), waits until the world is frozen, gated and unpaused, runs every check, asks the client to quit, and writes `runs/minecraft-tick-gate/result.json` plus `result.minecraft.log`. A full run takes about 80 s. It is an explicit integration test, not a Lefthook hook. The Loom dev launch uses an offline development profile (`Player###`); no Mojang login or authentication bypass is involved, and Realms fails to authorize as expected.

When `MCBOT_TICK_PORT` is set, the client runs in **observer mode**: the pause menu never opens (ESC or focus loss), the cursor is not captured, and window input cannot act on the player. Pressing **`` ` `` (grave/backtick)** in the window toggles **human control**: the mod captures the mouse and passes keyboard and mouse input through; pressing it again releases the mouse and all held keys. The pause menu stays suppressed in both states. Human input only acts during granted ticks, so the player moves at the probe's stepping rate. The client disables the tutorial overlay and creates a disposable flat survival world named `mcbot-episode-<millis>` from `MCBOT_WORLD_SEED` (default 12345). The integrated server freezes world ticking when it starts, and the client arms player/client gating once the player has loaded. With `--keep-open` the probe keeps stepping after the checks, fails if the pause state ever appears, and quits on Ctrl+C. The probe enables `MCBOT_TICK_TRACE` and fails unless the log shows `Server thread` tick callbacks.

The Python protocol client lives in `src/minecraft_rl/tick_control.py`; `tests/test_tick_control.py` covers request encoding, reply parsing, actions, error and EOF handling over a socket pair, so it runs in the normal fast checks.

## Structured-observation probe (v2)

```sh
.venv/bin/python scripts/minecraft-observation-probe.py
```

Same launch and observer mode as the tick-gate probe, on port 47124. It builds deterministic scenes through privileged `v2 DEBUG_*` commands (arena cleared, player teleported to face south, inventory cleared) and checks only the policy observation for visibility claims. It writes `runs/minecraft-observation/result.json`; a run takes about 80 s.

### Protocol v2

Newline-delimited `v2 <COMMAND> <json>` on the same loopback port; `v1` requests stay unchanged. Policy commands: `SCHEMA`, `STATUS`, `OBSERVE`, `STEP <PlayerAction>`, `RESET {"seed", "preset"}`, `QUIT`. `STEP` and `RESET` reply `{"observation", "terminated", "info"}`; `info` holds `step_id`, `tick_before`, `tick_after`, `client_tick`, `game_time` and phase timings, and is diagnostic. `DEBUG_SCENE`, `DEBUG_FILL`, `DEBUG_BLOCK`, `DEBUG_ENTITY`, `DEBUG_PLAYER`, `DEBUG_NEARBY`, `DEBUG_KILL`, `DEBUG_TRACE` and `DEBUG_REGISTRY` are privileged test instrumentation. In Python, `MinecraftClient` returns `PolicyObservation` and `StepInfo`; privileged data is reachable only through `client.privileged()` (`PrivilegedProbe`) and has no conversion into a policy observation.

`PlayerAction` has nine buttons (`forward`, `back`, `left`, `right`, `jump`, `sneak`, `sprint`, `attack`, `use`), bounded `yaw_delta`/`pitch_delta` in degrees (`|delta| <= 45`), and `hotbar` (`-1` keep, `0..8` select). Every field is required and unknown fields are rejected on both sides. Buttons are held for exactly the granted client tick; a button press that starts on a step also registers one vanilla click, so `attack` held across steps behaves like a held mouse button (continued mining, no repeated melee). Camera deltas are applied to the local player at `START_CLIENT_TICK`, before movement is simulated and sent. The probe world disables toggle-crouch/sprint/attack/use so buttons mean "held this tick".

`RESET` disarms the gate, disconnects the integrated world, deletes the previous `mcbot-episode-*` world, creates a fresh disposable world from `seed` and `preset` (`flat` or `normal`, survival, normal difficulty), and replies once the new world is frozen and the client is gated again. Game time restarts at 0. `terminated` is true when the player is dead or dying; respawn is not used, reset is the only continuation.

### Results (macOS arm64, seed 12345, 4 passing runs)

Visibility scene: a diamond block 5 blocks ahead in the open, a stone wall at `z+4` (x 1..7, height 4), an emerald block and a no-AI husk behind the wall, a gold block 5 blocks behind the player. Privileged `DEBUG_NEARBY` confirms all hidden objects exist within 12 blocks.

| Check | Result (all runs) |
| --- | --- |
| Visible block in view | present |
| Block behind an opaque wall | absent |
| Block behind the player | absent |
| Husk behind the wall | absent |
| Wall removed, next step | emerald block and husk present |
| Wall restored, next step | husk absent again |
| Turn 180 degrees (four 45-degree steps) | gold block present, diamond block absent |
| Turn back | diamond block present |
| 3 s idle, `OBSERVE` twice | identical observations |
| `yaw_delta=30` | server yaw 0 -> 30 on the same step |
| `pitch_delta=20` | observation pitch equals post-step server pitch (20.0); ray field changed |
| `forward` | 0.098 blocks on the same step; nearest diamond ray distance 4.500 -> 4.402 |
| Combat (stone sword, no-AI husk 2 blocks ahead) | first `attack` step: health 20.0 -> 19.06, hurt time 9; attack held 3 more steps: no further damage; release then press: hits again 2 steps later |
| Mining (held `attack` on dirt, empty hand) | block broken after 15 steps |
| Death | `DEBUG_KILL`, next step reports `terminated` (run 4) |
| Reset flat, flat, normal | 2.3-3.9 s flat, 4.5-6.9 s normal; each new world frozen and gated, game time 0 |

Combat note: the stone sword's attack cooldown is not fully recharged in the scene (low 0.94 damage). The probe characterizes click-versus-hold timing, not damage values.

Throughput (200 `STEP NOOP`, median of 3 runs, ms per step):

| Phase | Median | Meaning |
| --- | --- | --- |
| `client_wait_ms` | 3-12 | request accepted until the next frame grants the client tick |
| `client_tick_ms` | 1.6 | client tick incl. input, movement and packet send |
| `tick_end_wait_ms` | 30-39 | client tick end until the server loop consumes the tick-end packet |
| `server_step_ms` | 54-55 | step granted at `START_SERVER_TICK` until `END_SERVER_TICK` of the stepped tick |
| `observation_ms` | 0.7-0.8 | 825-ray sensor plus JSON encoding |
| `total_ms` | 99 | server-side total; Python round trip 99.7-99.9 |

Interpretation: `stepGameIfPaused` called at `START_SERVER_TICK` only takes effect on the following server loop iteration, because `TickRateManager.tick()` already ran for the current one; the stepped tick then waits for the 50 ms `waitUntilNextTick` deadline. Together with waiting for the tick-end packet this puts roughly two server-loop periods in every step. The sensor is not the bottleneck. Removing the wall-clock pacing is the next throughput experiment and must preserve one request = one exact transition.

## 26.3 tick lifecycle (from Loom-generated sources)

Sources were produced with `./scripts/gradle genSources` into the ignored Loom cache.

| Loop | Thread | Call path |
| --- | --- | --- |
| Client game/render loop | `Render thread` | `Minecraft.run()` → `runTick(advanceGameTime)`: process queued packets and tasks → `DeltaTracker.Timer.advanceGameTime()` returns how many ticks wall time allows → up to 10 × `Minecraft.tick()` → `renderFrame()` → `GameRenderer.render()` → swapchain blit. |
| Client tick | `Render thread` | `Minecraft.tick()`: client `TickRateManager.tick()`, `gameMode.tick()` (flushes the connection), keybinds, `ClientLevel.tickEntities()` (local player `aiStep` reads `KeyboardInput`, moves locally), `LocalPlayer.sendChanges()` (input and move packets), `ClientLevel.tick()`, then `ServerboundClientTickEndPacket`. |
| Integrated server loop | `Server thread` | `MinecraftServer.runServer()` → `processPacketsAndTick()`: process queued packets, then `IntegratedServer.tickServer()` → (unless paused) `MinecraftServer.tickServer()` → `tickCount++`, `ServerTickRateManager.tick()`, `tickChildren()` → each `ServerLevel.tick()`, then `tickConnection()` → `ServerGamePacketListenerImpl.tick()` → `tickPlayer()` → `ServerPlayer.doTick()`; then `waitUntilNextTick()` sleeps to the next 50 ms deadline. |
| World/entity tick | `Server thread` | `ServerLevel.tick()` runs world border, weather, `tickTime()` (game time +1), block/fluid ticks, raids, chunk source, block events and entity ticks only when `tickRateManager.runsNormally()`; entities are skipped by `isEntityFrozen()`. |

Player movement is client-authoritative: the client simulates the local player and sends move packets, and the server applies them in `handleMovePlayer` when packets are processed. `ServerPlayer.doTick()` then runs the server-side player tick (food, stats, effects).

Five clocks are distinct: wall-clock frames, client ticks (`Minecraft.tick`, wall-clock paced), integrated server loop iterations (`MinecraftServer.tickCount`, wall-clock paced), world logical time (`ServerLevel.getGameTime()`, advanced only when world elements run), and per-player server ticks (`tickPlayer`, run from the connection tick). `IntegratedServer.tickServer()` skips the whole server tick while the singleplayer pause menu is open, which is why an early manual attempt timed out.

## Gate design (experimental)

One `STEP <action>` advances every simulation clock exactly once, in this order:

1. The server thread records `tick_before` and the client tick count, and grants one client tick carrying the action.
2. On the next frame, `DeltaTracker.Timer.advanceGameTime()` (mixin) returns 1 instead of a wall-clock count. At `START_CLIENT_TICK` the action is applied (`FORWARD` holds the forward key). The client tick runs the local player, sends its input and move packets and the tick-end packet, and releases the key at `END_CLIENT_TICK`.
3. The server processes those packets in order, so the move is applied before the server tick. A mixin on `handleClientTickEnd` counts processed tick-end packets.
4. At `START_SERVER_TICK`, once the client tick happened and every sent tick-end packet has been processed, the gate calls `ServerTickRateManager.stepGameIfPaused(1)`. The same `tickServer()` then runs world, entities and players once.
5. At `END_SERVER_TICK` the gate verifies game time +1 and client tick +1, and replies `v1 STEP <step_id> <tick_before> <tick_after> <client_tick>`.

- Paused clocks: world time, non-player entities, server player ticks (`tickPlayer` is skipped and `isEntityFrozen` includes players while frozen; mixins, server `TickRateManager` only) and client game ticks. Rendering, input polling, packet processing and the server loop keep running, so no thread blocks or busy-waits.
- Control surface: newline text protocol on loopback TCP only (default port 47123, one connection). A daemon thread submits each request to the server thread with `execute()` and waits at most 5 s. Requests: `v1 STATUS`, `v1 STEP NOOP|FORWARD`, `v1 QUIT`, and privileged test-only `v1 DEBUG_SPAWN`, `v1 DEBUG_PROBE`, `v1 DEBUG_PLAYER`. A timed-out step is cancelled and the client grant revoked.
- `TickGate` holds the step state machine and wire encoding. JUnit tests cover waiting for the client tick and tick-end processing, exactly-once completion, monotonic IDs, rejection (paused, unfrozen, ungated, concurrent), server or client over-advance, cancellation and encoding.
- Privileged probe state (entity and player positions, tick counters) exists only for the integration test and is not a policy observation.

## Runtime results (macOS arm64, seed 12345)

World-tick gate only (earlier checkpoints; server loop kept running at 20 Hz while world time was frozen):

| Run | Passed | Ticks for 100 STEPs | Steps/s | Armor stand fall | Husk approach |
| --- | --- | --- | --- | --- | --- |
| 1 | yes | 100 | 19.99 | 10.0 | 4.52 |
| 2 | yes | 100 | 20.01 | 10.0 | 4.30 |
| 3 | yes | 100 | 19.98 | 10.0 | 4.52 |

Full client/player gate (current probe):

| Check | Result (run 2) |
| --- | --- |
| Idle 5 s, world | game time 40 → 40, client ticks 160 → 160 (server loop 125 → 225) |
| Idle 5 s, player | server/client position, player tick count and play-time stat unchanged |
| One STEP | game time 40 → 41, client tick 160 → 161 |
| 100 STEPs | exactly 100 world ticks and 100 client ticks |
| Per-STEP clock deltas | game time, server player ticks, play-time stat, client ticks and client player ticks each +1 |
| NOOP step | player moved 0.0 blocks |
| First FORWARD step | server player moved 0.098 blocks on that same step |
| 10 FORWARD steps | 1.90 blocks total |
| Physics / mob | armor stand fell 10 blocks; husk approached 4.63 blocks; both unchanged over 5 s idle |
| Throughput | ~10 steps/s |

Repeated runs of the full gate (each a fresh world, same seed):

| Run | Passed | World / client ticks for 100 STEPs | Steps/s | NOOP move | First FORWARD move | 10 FORWARD total | Stand fall | Husk approach |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | yes | 100 / 100 | 10.01 | 0.0 | 0.098 | 1.8996004544472491 | 10.0 | 4.630 |
| 3 | yes | 100 / 100 | 10.00 | 0.0 | 0.098 | 1.8996004544472485 | 10.0 | 4.514 |
| 4 | yes | 100 / 100 | 10.00 | 0.0 | 0.098 | 1.8996004544472491 | 10.0 | 4.630 |

The scripted player trajectory repeated to within 1e-15 blocks and physics was identical across runs, but the husk approach still varied (4.51 vs 4.63). Run 1 of the full gate exceeded a 58 s shell timeout while starting; it is not counted.

## Known limitations and open work

- Accelerated (unpaced) stepping is implemented and validated against paced stepping: paced ~20, unpaced ~58-99 with rendering and ~140-154 without rendering steps/s, varying between sessions. Results, mechanism and equivalence evidence are in `docs/decisions/simulation-throughput.md`. The tables above predate it.
- v1 `NOOP` and `FORWARD` remain as the proven timing probe; v2 `PlayerAction` is the factorized interface. GUI and inventory actions are not implemented.
- The client's copy of non-player world state can lag the server by one step (`docs/replay-characterization.md`). An ordering barrier now makes the lag identical in paced and unpaced modes (`docs/decisions/simulation-throughput.md`). The v2 observation is computed from server state, so this does not affect it; it matters for recording.
- Visibility evidence covers one scripted scene with full opaque blocks, one mob type and daylight. Transparent blocks, partial shapes, fluids, small entities, lighting and entities between rays are not yet tested; see `docs/decisions/observation.md` for known v1 limitations.
- Replay characterization covers one flat scene and 243 steps; see `docs/replay-characterization.md`.
- Both runtime probes are non-recording infrastructure tests under `docs/decisions/recording.md`; `scripts/minecraft-recording-probe.py` records and validates session recording for one worker.
- Probes without `worker` get a random development username (`PlayerNNN`, the 26.3 `--username` default); workers launched with `-Pmcbot.worker=NNNN` use the deterministic `AgentNNNN` identity (`docs/decisions/parallel-workers.md`).
- Low-cost client settings (fast graphics, render distance 8, simulation distance 6, VSync off) are applied in observer mode and reported by `STATUS`.
- Worker isolation is proven for two workers only; scaling, per-worker resource use, orchestration and restart handling are open (`docs/decisions/parallel-workers.md`). Recording uses the macOS `h264_videotoolbox` encoder; Linux needs another encoder.
