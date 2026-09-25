# Minecraft 26.3 lockstep engineering spike

**Scope:** No Minecraft agent training, neural network, reward shaping or policy state exposure. This spike asks whether one Python action can advance exactly one integrated-server logical tick and then return a post-step observation while the game remains paused between steps. A Fabric build is not proof of that contract.

## What must be tested in a running game

| Probe | Required observation | Proof criterion |
| --- | --- | --- |
| One NOOP | `step_id`, `tick_before`, `tick_after` | Exactly one server/world tick, with no active input. |
| Repeated NOOPs | one response per ID | N steps advance exactly N server/world ticks. |
| Idle wall-clock wait | ticks before/after several seconds | Tick count unchanged; keep network/command handling live. |
| Action timing | before/after player state with step IDs | Action affects the intended step, not a later tick. |
| Falling player/entity | privileged position and velocity | Progress only during steps, not idle wall time. |
| Hostile mob | privileged position/distance/AI state | AI progresses during steps but stops while waiting. |
| Combat sequence | player/mob health and positions per ID | Correct attack order, cooldown and damage tick. |
| Reset(seed) | world/episode/player/entity state before/after | Explicitly defined guarantees, without leaking internals to the policy. |
| Replay | per-tick privileged snapshots from same seed/actions | Differences enumerated, not dismissed as deterministic. |
| Render/capture | post-tick frame token or visual state marker | Return occurs after client receives post-step state and renders/captures it. |

These are integration properties: tests against a Python-only stand-in or incrementing an internal counter do **not** establish Minecraft behavior. An observation field carrying only privileged player/mob coordinates is not an RGB observation. In the absence of a verified reset path, a game-launch test must not claim world restoration. Benchmark only after correctness: server simulation-only, rendered stepping, framebuffer readback, and IPC must be timed separately, recording the configuration and steps/s.

## Current evidence and blockers

The existing Fabric skeleton builds for 26.3 with the repository-local Java 25. An opt-in Fabric `ServerTickEvents.START_SERVER_TICK` / `END_SERVER_TICK` trace has now been added: set `MCBOT_TICK_TRACE=1` when launching a legitimate client to log callback counts. The counter is local to this mod instance and does not assert world time, step IDs, or tick causality; no client run or callback log has been validated. The Python/Fabric bridge, reset path and executable Minecraft integration tests do not exist. A successful Gradle build only establishes that the hook compiles and dependencies resolve. Fabric's `ServerTickEvents.START_SERVER_TICK` and `END_SERVER_TICK` are callbacks around a server tick; the client has independent client tick events. An `END_SERVER_TICK` callback does not gate the main tick loop. Blocking the game/server thread for an action risks blocking its task and networking queues; using normal pause semantics may suppress the tick callbacks needed to resume. Client rendering can run at a different cadence from both loops, so a server-tick callback does not identify a completed framebuffer. The precise 26.3 vanilla call chain and client capture point still require direct code/runtime inspection before a lockstep implementation can be claimed.

The intended ordering is: accept one action with a monotonic ID; apply it at the verified pre-tick boundary; advance server/world exactly once; wait for corresponding client state and frame; read framebuffer; return observation and scalar reward with `{step_id,tick_before,tick_after}` in diagnostic `info`; leave future simulation paused. Privileged snapshots used by tests must be isolated from the policy observation. A missing frame, timeout, extra tick or unexpected reset must fail the step rather than quietly return an old frame.

## Validation status (partial checkpoint)

`./scripts/gradle build` passed on the project-local JDK 25, and the existing Stage 1 Python checks passed. `runClient`, integrated-server ticking, rendering, framebuffer capture and Minecraft-level lockstep were not run or measured. The remote repository was empty when this checkpoint was prepared; it does not contain a previous verified Minecraft runtime. No tests in this spike satisfy the ten running-game probes above. **Stage 1.5 remains incomplete**. A source-level 26.3 tick-loop trace and legitimate authenticated game launch are still needed before choosing a control point and implementing tick gating. `MCBOT_TICK_TRACE=1 ./scripts/gradle runClient` is the manual opt-in starting point for tick callback observations; launch/authentication and resulting logs have not yet been demonstrated. Do not infer simulation throughput from a Gradle build or a GridWorld run. No simulation/render/capture/IPC throughput values are available.

The current prototype guarantees only opt-in event logging when a server actually ticks. There is no `env.reset(seed)` guarantee, no `env.step(action)` guarantee, no NOOP implementation, no policy RGB observation, no replay determinism result and no measured throughput. In particular, there is no mechanism to pause either the integrated server or client between external actions.

See [the decision record](decisions/lockstep.md) for IPC tradeoffs and the difference between the target contract and measured guarantees.
