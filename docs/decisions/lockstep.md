# Lockstep simulation contract (engineering spike)

## Decision: target contract, not an implementation claim

An eventual Minecraft environment step submits exactly one player action, advances exactly one logical simulation tick, synchronizes the result, captures a post-step observation, and waits again. Time spent in Python inference must not advance simulation time. A reply must include `step_id`, `tick_before`, and `tick_after` in a diagnostic field, with privileged state available only through an explicitly separate debug/test interface. A successful server-tick probe alone does not prove client rendering or observation freshness.

The target Python interface is `obs, info = env.reset(seed=...)` and `obs, reward, terminated, truncated, info = env.step(action)`. It is a target rather than a deployed contract until integrated-server and client-side synchronization are verified. Do not put test-only coordinates, mob health, seed internals, or other privileged information into policy observations.

Alternatives: asynchronous real-time stepping (rejected because inference latency changes gameplay), server-only pause (insufficient for rendering/client input), or advancing multiple ticks per action (may be studied later, not the one-tick contract). Revisit the mechanism based on the actual Minecraft 26.3 tick and render lifecycle, rather than adding an untested pause loop on an arbitrary callback thread.

## Local IPC tradeoffs before selecting a transport

- Localhost TCP is portable between macOS/Linux and Java/Python, supports simple newline-delimited request/response and explicit ordering, but requires port selection and listener binding checks. Bind loopback only; a debug endpoint is not a trusted network service.
- Unix domain sockets avoid an exposed TCP port and inherently scope access to the host, but path length, stale socket cleanup, OS support and Windows portability complicate startup.
- WebSocket offers structured browser-facing framing but adds handshake/dependencies without a browser consumer or a demonstrated need.

For the first verified experiment, prefer loopback TCP with one connection and one in-flight command. This is a proposed transport choice, not evidence that a running Minecraft bridge exists. A request/reply version and per-step monotonic ID are required; explicitly separate the policy observation from privileged `debug` snapshots. Do not declare the transport implemented before a real runtime test.

## Investigation and validation ledger

- Transport for the spike: loopback TCP, one connection, one in-flight request, line protocol `v1 STATUS|STEP|QUIT` (plus test-only `DEBUG_*`). Chosen as the smallest portable option; not yet the final environment protocol.
- Proven (automated, repeated three times, see `docs/minecraft-spike.md`): with vanilla `ServerTickRateManager` freeze/step, world logical time and non-player entities stay still for 5 s of wall time and advance exactly one tick per STEP (1 and 100 steps verified), including falling-entity physics and hostile-mob AI. Mob AI trajectories differed slightly between same-seed runs.
- Not proven: player/client-tick gating (players are exempt from freeze), action ordering, render/capture synchronization, reset, replay determinism, throughput above ~20 steps/s. Until those are shown, `env.step()` is not a full lockstep contract.
