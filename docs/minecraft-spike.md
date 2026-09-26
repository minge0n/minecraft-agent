# Minecraft 26.3 lockstep engineering spike

**Scope:** No agent training, neural network, reward, reset or RGB transport. This spike asks whether an external controller can pause Minecraft's logical simulation and advance it by exactly one tick per request. Stage 1.5 is **incomplete**; only the world-tick layer below has runtime evidence.

## Automated runtime probe

```sh
.venv/bin/python scripts/minecraft-tick-gate-probe.py
```

The script launches the Loom `runClient` configuration through `./scripts/gradle` (project-local JDK 25, `.runtime/` game data), waits until the world is frozen and unpaused, runs every check, asks the client to quit, and writes `runs/minecraft-tick-gate/result.json` plus `result.minecraft.log`. No mouse or keyboard input is needed: when `MCBOT_TICK_PORT` is set, the client entrypoint disables pause-on-lost-focus, creates a disposable flat survival world named `mcbot-tick-probe-<millis>` from `MCBOT_WORLD_SEED` (default 12345), and the integrated server freezes world ticking as soon as it starts. The probe also enables `MCBOT_TICK_TRACE` and fails unless the log shows `Server thread` tick callbacks. The Loom dev launch uses an offline development profile (`Player###`); no Mojang login or authentication bypass is involved, and online services such as Realms fail to authorize as expected. It is a manual/explicit integration test, not a Lefthook hook.

The Python protocol client lives in `src/minecraft_rl/tick_control.py`; `tests/test_tick_control.py` covers request encoding, reply parsing, error and EOF handling over a socket pair, so it runs in the normal fast checks.

## 26.3 tick lifecycle (from Loom-generated sources)

Sources were produced with `./scripts/gradle genSources` into the ignored Loom cache.

| Loop | Thread | Call path |
| --- | --- | --- |
| Client game/render loop | `Render thread` | `Minecraft.run()` → `runTick(advanceGameTime)`: process queued packets and tasks → up to 10 × `Minecraft.tick()` as `DeltaTracker` accrues wall time → `renderFrame()` → `GameRenderer.render()` → swapchain blit. |
| Integrated server loop | `Server thread` | `MinecraftServer.runServer()` → `processPacketsAndTick()` → `IntegratedServer.tickServer()` → (unless paused) `MinecraftServer.tickServer()` → `tickCount++`, `ServerTickRateManager.tick()`, `tickChildren()` → each `ServerLevel.tick()`; then `waitUntilNextTick()` sleeps until the next 50 ms deadline. |
| World/entity tick | `Server thread` | `ServerLevel.tick()` runs world border, weather, `tickTime()` (game time +1), block/fluid ticks, raids, chunk source, block events and entity ticks only when `tickRateManager.runsNormally()`; entities are skipped by `isEntityFrozen()`. |

Four clocks are therefore distinct: wall-clock frames, client ticks (`Minecraft.tick`, wall-clock paced), integrated server loop iterations (`MinecraftServer.tickCount`, wall-clock paced), and world logical time (`ServerLevel.getGameTime()`, advanced only when world elements run). `IntegratedServer.tickServer()` skips the whole server tick while the singleplayer pause menu is open (`Minecraft.isPaused()`), which is why an earlier manual attempt timed out.

## Gate design (experimental)

- Control point: vanilla `ServerTickRateManager` (the mechanism behind `/tick freeze` and `/tick step`). `setFrozen(true)` stops world elements; `stepGameIfPaused(1)` lets exactly one server tick run them.
- Paused clock: world logical time and non-player entities. The server loop keeps running at 20 Hz, so packets, chunk sending and scheduled tasks are never blocked and no Minecraft thread busy-waits.
- Control surface: newline text protocol on loopback TCP only (default port 47123, one connection), handled by a daemon thread that submits each request to the server thread with `execute()` and waits at most 5 s. Requests: `v1 STATUS`, `v1 STEP`, `v1 QUIT`, plus privileged test-only `v1 DEBUG_SPAWN` / `v1 DEBUG_PROBE`. Replies carry `step_id`, `tick_before` and `tick_after` (world game time). A timed-out step is cancelled with `stopStepping()`.
- `TickGate` holds the step state machine and wire encoding; JUnit tests cover step completion, monotonic IDs, rejection when paused/unfrozen/concurrent, over-advance detection, cancellation and encoding.
- Privileged probe state (armor stand height, husk position/distance) exists only for the integration test and is not a policy observation.

## Runtime results (macOS arm64, seed 12345)

First run (earlier checkpoint, 20 entity steps):

| Check | Result |
| --- | --- |
| Client launched, `mcbot 0.1.0` loaded, integrated server started | yes |
| Idle 5 s without STEP | world time 40 → 40 (server loop 47 → 147) |
| One STEP | world time 40 → 41 |
| 100 STEPs | exactly 100 world ticks, each reply `N → N+1`; no drift after the last step |
| Armor stand (physics) idle 5 s / while stepping | unchanged / fell 10 blocks |
| Husk (hostile AI) idle 5 s / while stepping | unchanged / began moving toward the player |

Repeated runs with the current probe (60 entity steps, each probe snapshot required to match the stepped tick, husk must approach ≥ 1 block):

| Run | Passed | Ticks for 100 STEPs | Steps/s | Armor stand fall | Husk approach | Server-thread trace |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | yes | 100 | 19.99 | 10.0 | 4.52 | yes |
| 2 | yes | 100 | 20.01 | 10.0 | 4.30 | yes |
| 3 | yes | 100 | 19.98 | 10.0 | 4.52 | yes |

Each run created a fresh world with the same seed. Physics was identical across runs, but the husk's approach differed (4.30 vs 4.52 blocks): mob AI is not bit-for-bit reproducible across launches. This is an early replay/determinism observation, not a finished determinism study.

## Known limitations and open work

- **Players are exempt from the freeze** (`TickRateManager.isEntityFrozen` excludes `Player`), and client ticks keep running. Player movement, input and client prediction are not gated; action application ordering is unproven.
- Throughput is ~20 steps/s, capped by the server's 50 ms tick deadline, because a step still waits for the next scheduled server tick. Faster stepping needs a different mechanism (for example controlled sprinting) and must be re-verified.
- Only three runs on one machine. Combat, reset, a proper replay/determinism comparison, render synchronization and framebuffer capture are untested.
