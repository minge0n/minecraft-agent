# Minecraft 26.3 lockstep engineering spike

**Scope:** No agent training, neural network, reward, reset or RGB transport. This spike asks whether an external controller can pause Minecraft's logical simulation and advance it by exactly one tick per request, with a player action applied on the intended tick. Stage 1.5 is **incomplete**: world, entity, server-player and client ticks are gated and runtime-verified; render/capture synchronization, reset, replay determinism and combat are not.

## Automated runtime probe

```sh
.venv/bin/python scripts/minecraft-tick-gate-probe.py              # run checks, then quit
.venv/bin/python scripts/minecraft-tick-gate-probe.py --keep-open  # keep stepping for watching
```

The script launches the Loom `runClient` configuration through `./scripts/gradle` (project-local JDK 25, `.runtime/` game data), waits until the world is frozen, gated and unpaused, runs every check, asks the client to quit, and writes `runs/minecraft-tick-gate/result.json` plus `result.minecraft.log`. A full run takes about 80 s. It is an explicit integration test, not a Lefthook hook. The Loom dev launch uses an offline development profile (`Player###`); no Mojang login or authentication bypass is involved, and Realms fails to authorize as expected.

When `MCBOT_TICK_PORT` is set, the client runs in **observer mode**: the pause menu never opens (ESC or focus loss), the cursor is not captured, and window input cannot act on the player. Pressing **`` ` `` (grave/backtick)** in the window toggles **human control**: the mod captures the mouse and passes keyboard and mouse input through; pressing it again releases the mouse and all held keys. The pause menu stays suppressed in both states. Human input only acts during granted ticks, so the player moves at the probe's stepping rate. The client disables the tutorial overlay and creates a disposable flat survival world named `mcbot-tick-probe-<millis>` from `MCBOT_WORLD_SEED` (default 12345). The integrated server freezes world ticking when it starts, and the client arms player/client gating once the player has loaded. With `--keep-open` the probe keeps stepping after the checks, fails if the pause state ever appears, and quits on Ctrl+C. The probe enables `MCBOT_TICK_TRACE` and fails unless the log shows `Server thread` tick callbacks.

The Python protocol client lives in `src/minecraft_rl/tick_control.py`; `tests/test_tick_control.py` covers request encoding, reply parsing, actions, error and EOF handling over a socket pair, so it runs in the normal fast checks.

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

- Throughput is ~10 steps/s: each step waits for a render frame and then up to two 50 ms server iterations, one for the tick-end packet and one for the gated tick. Faster stepping needs a different mechanism and must be re-verified.
- Only `NOOP` and `FORWARD` actions exist. They are scaffolding for timing tests, not the agent action space.
- The client learns about a server step through `ClientboundTickingStepPacket`, which is processed before a later client tick. Client-side views of non-player entities are therefore likely one step behind the server. This is inferred from source, not measured, and matters for observation synchronization.
- Combat, reset, a proper replay/determinism comparison, render synchronization and framebuffer capture are untested. Mob AI already differed slightly between same-seed runs.
