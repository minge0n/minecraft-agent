# Parallel environment workers and development identities

Status: **decided direction; not implemented.** Nothing here is built before one environment is correct and accelerated stepping is validated (`docs/roadmap.md`). Until then, avoid assumptions that restrict the project to one global Minecraft process.

## Worker isolation

Each worker `env-NNNN` has a unique environment ID, development identity, runtime directory, save directory, IPC port or endpoint, world seed, log directory and recording output path, for example under `.runtime/workers/NNNN/`. No mutable state is shared: options, saves, logs, crash reports and endpoints are per worker. Read-only files obtained through Loom (game jars, libraries, assets) may be shared. No worker uses a personal launcher profile or personal Minecraft installation.

Current state: one client in `.runtime/minecraft/client` (fixed Loom `runDirectory`), port from `MCBOT_TICK_PORT` (probes use 47123 and 47124), seed from `MCBOT_WORLD_SEED`. Per-worker runtime directories need a launch path that sets the game directory per process; not designed yet.

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
