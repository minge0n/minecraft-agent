# Lockstep simulation contract (engineering spike)

## Decision: target contract, not an implementation claim

An eventual Minecraft environment step submits exactly one factorized player action, advances exactly one logical simulation tick, synchronizes the result, builds the post-step observation, and waits indefinitely for the next action:

```text
observation_t -> factorized action_t -> exactly one logical tick -> observation_t+1 -> wait
```

Time spent in Python inference must not advance simulation time. A reply must include `step_id`, `tick_before`, and `tick_after` in a diagnostic field, with privileged state available only through an explicitly separate debug/test interface.

**Changed (observation architecture update):** the post-step observation is now the structured visible field of `docs/decisions/observation.md`, built on the server thread from post-step state. RGB framebuffer freshness was previously a Stage 1.5 requirement; it is now deferred with RGB itself. If RGB returns, a server-tick proof alone still does not prove client rendering or capture freshness.

The target Python interface is `obs, info = env.reset(seed=...)` and `obs, reward, terminated, truncated, info = env.step(action)`. It is a target rather than a deployed contract until integrated-server and client-side synchronization are verified. Do not put test-only coordinates, mob health, seed internals, or other privileged information into policy observations.

Alternatives: asynchronous real-time stepping (rejected because inference latency changes gameplay), server-only pause (insufficient for rendering/client input), or advancing multiple ticks per action (may be studied later, not the one-tick contract). Revisit the mechanism based on the actual Minecraft 26.3 tick and render lifecycle, rather than adding an untested pause loop on an arbitrary callback thread.

## Local IPC tradeoffs before selecting a transport

- Localhost TCP is portable between macOS/Linux and Java/Python, supports simple newline-delimited request/response and explicit ordering, but requires port selection and listener binding checks. Bind loopback only; a debug endpoint is not a trusted network service.
- Unix domain sockets avoid an exposed TCP port and inherently scope access to the host, but path length, stale socket cleanup, OS support and Windows portability complicate startup.
- WebSocket offers structured browser-facing framing but adds handshake/dependencies without a browser consumer or a demonstrated need.

For the first verified experiment, prefer loopback TCP with one connection and one in-flight command. This is a proposed transport choice, not evidence that a running Minecraft bridge exists. A request/reply version and per-step monotonic ID are required; explicitly separate the policy observation from privileged `debug` snapshots. Do not declare the transport implemented before a real runtime test.

## Investigation and validation ledger

- Transport for the spike: loopback TCP, one connection, one in-flight request. Line protocol `v1 STATUS|STEP <action>|QUIT` (plus test-only `DEBUG_*`) and `v2 <COMMAND> <json>` for structured observations, factorized actions and reset. Chosen as the smallest portable option; not yet the final environment protocol.
- Proven (automated, full gate passed three times, see `docs/minecraft-spike.md`): with vanilla `ServerTickRateManager` freeze/step plus mixins gating client ticks (`DeltaTracker.Timer.advanceGameTime`) and server player ticks (`tickPlayer`, `isEntityFrozen`), one STEP advances world time, server player, client and local player by exactly one tick each. Nothing advances during 5 s of wall-clock idle. A `FORWARD` action moves the server player on the same step it is submitted. Physics and hostile-mob AI progress only on steps. Mob AI trajectories differed slightly between same-seed runs.
- Proven (v2 observation probe, 4 runs): a factorized `PlayerAction` applied on step N is reflected in step N's structured observation (yaw, pitch, position); occluded, behind-player and behind-wall objects are absent while privileged state proves they exist; removing the wall reveals them on the next step; combat and mining timing, death termination and fresh-world reset behave as described in `docs/minecraft-spike.md`.
- Not proven: replay determinism, throughput above ~10 steps/s, visibility beyond the scripted scene. Render/capture synchronization is deferred with RGB (client views of non-player entities are likely one step behind the server, inferred from source).
- Protocol versioning: `v1` stays as validated and was re-run after the v2 changes. Structured observations and factorized actions use the `v2` request family alongside it.
- Acceleration keeps this contract: `docs/decisions/simulation-throughput.md` removes wall-clock pacing between ticks without skipping or merging ticks, validated by paced/unpaced equivalence tests.
