# Minecraft 26.3 lockstep engineering spike

**Scope:** No agent training, neural network, reward, reset or RGB transport. This spike asks whether an external controller can pause Minecraft's logical simulation and advance it by exactly one tick per request. Stage 1.5 is **incomplete**; only the world-tick layer below has runtime evidence.

## Automated runtime probe

```sh
.venv/bin/python scripts/minecraft-tick-gate-probe.py
```

The script launches the Loom `runClient` configuration through `./scripts/gradle` (project-local JDK 25, `.runtime/` game data), waits for the tick-control port, runs every check, asks the client to quit, and writes `runs/minecraft-tick-gate/result.json` plus the full game log. No mouse or keyboard input is needed: when `MCBOT_TICK_PORT` is set, the client entrypoint disables pause-on-lost-focus, creates a disposable flat survival world named `mcbot-tick-probe-<millis>` from `MCBOT_WORLD_SEED` (default 12345), and the integrated server freezes world ticking as soon as it starts. The Loom dev launch uses an offline development profile (`Player###`); no Mojang login or authentication bypass is involved, and online services such as Realms fail to authorize as expected. It is a manual/explicit integration test, not a Lefthook hook.

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

## Runtime result (macOS arm64, seed 12345, one run)

| Check | Result |
| --- | --- |
| Client launched, `mcbot 0.1.0` loaded, integrated server started | yes |
| Idle 5 s without STEP | world time 40 → 40 (server loop 47 → 147) |
| One STEP | world time 40 → 41 |
| 100 STEPs | exactly 100 world ticks, each reply `N → N+1`; no drift after the last step |
| Armor stand (physics) idle 5 s | y unchanged at −50.0 |
| Armor stand while stepping | fell 10 blocks over 20 steps, landed at −60.0 |
| Husk (hostile AI) idle 5 s | position unchanged |
| Husk while stepping | started moving toward the player (distance 6.0 → 5.947 by step 20) |
| Throughput | ~20 steps/s (5.0 s for 100 steps) |

## Known limitations and open work

- **Players are exempt from the freeze** (`TickRateManager.isEntityFrozen` excludes `Player`), and client ticks keep running. Player movement, input and client prediction are not gated; action application ordering is unproven.
- Throughput is capped by the server's 50 ms tick deadline, because a step still waits for the next scheduled server tick. Faster stepping needs a different mechanism (for example controlled sprinting) and must be re-verified.
- Only one run on one machine. The husk moved late and slowly; a longer mob trajectory and repeated runs are needed before relying on AI timing. Combat, reset, replay determinism, render synchronization and framebuffer capture are untested.
- The Python controller in `scripts/` has no unit tests yet; the Java state machine does.
