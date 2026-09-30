# Parallel environment workers and development identities

Status: **isolation and identities implemented and smoke-tested with two concurrent workers on 26.3 (macOS arm64).** Orchestration (spawn N, health check, restart) and measured concurrency choices are not implemented; see Implementation and Results.

## Worker isolation

Each worker `env-NNNN` has a unique environment ID, development identity, runtime directory, save directory, IPC port or endpoint, world seed, log directory and recording output path, for example under `.runtime/workers/NNNN/`. No mutable state is shared: options, saves, logs, crash reports and endpoints are per worker. Read-only files obtained through Loom (game jars, libraries, assets) may be shared. No worker uses a personal launcher profile or personal Minecraft installation.

Previous state (superseded by Implementation below): one client in `.runtime/minecraft/client` (fixed Loom `runDirectory`), port from `MCBOT_TICK_PORT`, seed from `MCBOT_WORLD_SEED`, and a random `PlayerNNN` identity per launch.

## Learner separation

```text
Minecraft environment workers -> real transition sequences -> replay storage
    -> Dreamer learner -> updated parameters -> environment workers
```

The learner is not coupled to one Minecraft process. The environment API (`MinecraftClient`, addressed by port) is used identically for one local debug environment and for many training workers.

## Parallelism is measured

More instances do not automatically mean more throughput; Minecraft uses substantial CPU and memory. Benchmark one accelerated instance, several instances per host, learner GPU utilization, environment CPU saturation, memory, IPC, replay and recording overhead, and choose the concurrency empirically. The bottleneck may be simulation, rendering or recording, world-model learning or replay processing.

## Orchestration (later)

When parallel workers become necessary, add a small explicit worker manager instead of manually launched terminals: spawn N workers, assign unique ports and paths, health check, reset, restart failed workers, graceful shutdown, collect per-worker metrics. Not before a single environment is correct and accelerated stepping has been investigated.

## Offline development identities

Workers never depend on interactive Microsoft/Minecraft authentication. The developer owns Minecraft: Java Edition; game jars and assets come only from official development tooling (Fabric Loom). Forbidden: authentication cracking, launcher authentication bypasses, token sharing between workers, account-session automation, unofficial launchers and unofficial binary distribution.

Mechanism, established from 26.3 `net.minecraft.client.main.Main`: the development client accepts `--username` (default `"Player" + currentTimeMillis() % 1000`) and derives the UUID with `UUIDUtil.createOfflinePlayerUUID(username)` unless `--uuid` is given. Probe logs show a different random `PlayerNNN` per launch, so identities are currently nondeterministic and can collide.

Decision: worker `env-NNNN` launches with `--username AgentNNNN` and no `--uuid`, which yields a deterministic vanilla offline UUID per worker. Run metadata records both. Concurrent workers never share an identity.

This is a client identity for the integrated singleplayer server, not dedicated-server `online-mode=false`. Client camera and input semantics are part of the environment, so each worker runs a client. If the architecture changes to separate clients and dedicated servers, revisit authentication and server mode explicitly.

## Launch mechanism

The Loom development client (`runClient`) is acceptable for correctness work. Before large-scale training, compare it with Fabric/Loom production-run tooling on startup cost, memory, CPU, compatibility with accelerated stepping, multi-instance isolation and Linux virtual-display/headless operation. Environment semantics must be identical across launch mechanisms.

## Implementation

- **Per-worker launch.** `./scripts/gradle runClient -Pmcbot.worker=NNNN` (four digits) sets the Loom client `runDirectory` to `.runtime/workers/NNNN/client` and passes `--username AgentNNNN`. Without the property the single-client layout `.runtime/minecraft/client` is unchanged. Game jars, libraries and assets stay in the shared read-only Loom cache (`.runtime/gradle`).
- **Identity.** No `--uuid` is passed, so the vanilla client derives the offline UUID `UUID.nameUUIDFromBytes("OfflinePlayer:AgentNNNN")`: `Agent0000` -> `c577fccb-2a93-3236-bf4e-d10253a2650e`, `Agent0001` -> `e1c8548d-485d-3b49-813b-4b6fbfa47cc4`. No authentication is bypassed and no token exists.
- **Diagnostics.** v2 `STATUS` reports `identity` (username, UUID, game directory) so run metadata can record them.
- **Python.** `launched_client(..., worker=N)` launches worker N; `worker_game_directory(N)` and `worker_username(N)` give the expected path and identity. Each worker gets its own `MCBOT_TICK_PORT`, `MCBOT_WORLD_SEED` and `WorkerRecorder` writing to `runs/<run>/workers/env-NNNN/`.

## Results (macOS arm64, 10 cores, 32 GiB)

`scripts/minecraft-worker-probe.py`, run `runs/minecraft-workers/w1`, passed:

- Two clients started concurrently in 28 s as `Agent0000` (port 47150, seed 1000) and `Agent0001` (port 47151, seed 1001), each with the expected offline UUID and its own game directory; no username, UUID, directory or port was shared.
- Each worker's `saves`, `logs/latest.log` (`Setting user: AgentNNNN`) and `options.txt` stayed inside its own directory; `.runtime/minecraft/client` was not modified.
- Stepping one worker 50 times left the other's world time and client ticks unchanged, in both directions; every step advanced exactly one tick.
- The controlled replay (243 steps) played on both workers at the same time produced identical traces across workers, apart from the characterized per-run `tick_count` offset, at 70 and 66 steps/s unpaced with rendering and recording.
- Each worker recorded its own video (244 of 244 due frames, no drops).

Not covered yet: more than two workers, throughput scaling, memory and CPU per worker, crash and restart handling, and production-run launch tooling.
