# Minecraft RL Lab

Research and teaching project toward a language-free, randomly initialized reinforcement-learning agent with a learned recurrent world model (Dreamer-style) for Minecraft 26.3. Perception is simplified into a structured, visibility-limited observation of what the player can currently see; hidden world state, recipes, demonstrations, pretrained models and scripted Minecraft knowledge stay unavailable to the policy.

The repository currently contains reproducible tooling, a completed tabular Q-learning baseline, and a Fabric mod whose lockstep tick gate, structured visible-field observation, factorized actions and fresh-world reset are runtime-verified. It does **not** yet contain a neural network, a Minecraft agent or rewards.

## Setup and commands

On macOS or Linux (arm64/x86_64), install the bootstrap prerequisites: POSIX shell, curl, tar, host Python 3 and Git. Network access is required for downloads. The host Python is used only for checksum verification; project code always uses the local managed interpreter. From a clone:

```sh
./scripts/bootstrap
./scripts/check
./.tools/lefthook run pre-commit
.venv/bin/python -m minecraft_rl.train --episodes 1000 --seed 42
./scripts/gradle build
.venv/bin/python scripts/minecraft-tick-gate-probe.py
.venv/bin/python scripts/minecraft-observation-probe.py
```

Bootstrap downloads SHA-256-verified uv 0.12.19, Temurin 25.0.4.1+1 and Lefthook 2.1.14 into ignored `.tools/`, installs uv-managed CPython 3.12.11 in `.tools/uv-python`, syncs the locked dev environment to `.venv/`, and installs Git hooks when inside a Git checkout. It does not install any global Python package, JDK or Gradle. Re-run bootstrap after dependency changes and commit the updated `uv.lock`. Use `./.tools/uv lock` to update the lock deliberately. `./scripts/check` runs Ruff format check, Ruff lint and pytest; the same quick checks run from the Lefthook pre-commit hook. Training, Fabric builds and Minecraft runtime probes are explicit, not commit hooks. Java/Fabric changes should additionally pass `./scripts/gradle build`.

The Gradle wrapper is committed under `minecraft/fabric-mod/`. Always invoke it through `./scripts/gradle`: the script sets `JAVA_HOME` and `PATH` to the pinned local JDK and `GRADLE_USER_HOME` to `.runtime/gradle`. Loom's client and server run directories point to `.runtime/minecraft/client` and `.runtime/minecraft/server`; no personal Minecraft directory is read. Minecraft files are obtained through normal Loom development tooling. The runtime probe launches the development client unattended; see [the spike ledger](docs/minecraft-spike.md).

The GridWorld run prints the learned Q-table and a greedy policy map and writes metrics into an ignored per-run `runs/gridworld-.../metrics.json`. See [Stage 1](docs/stage1.md) for the exact map, update and interpretation.

## Layout and boundaries

- `src/minecraft_rl/gridworld.py`, `q_learning.py`, `train.py`: Stage 1 tabular baseline.
- `src/minecraft_rl/tick_control.py`: Python client for the validated v1 lockstep probe protocol.
- `src/minecraft_rl/minecraft_interface.py`: versioned policy observation schema (`PolicyObservation`) and factorized `PlayerAction`.
- `src/minecraft_rl/minecraft_client.py`: v2 client (`reset`, `step`, `observe`) and the separate test-only `PrivilegedProbe`.
- `tests/`: deterministic checks for the current behaviors.
- `minecraft/fabric-mod/`: Fabric environment-integration layer: lockstep tick gate, observer mode and test-only probe instrumentation.
- `scripts/`: bootstrap, fast checks, pinned-JDK Gradle entrypoint and the Minecraft runtime probe.
- `docs/`: decisions, roadmap and experiment interpretation.

Design documents:

- [Observation decision](docs/decisions/observation.md): camera-ray structured visible field, categorical IDs, egocentric self state, no recipes or hidden state. RGB is deferred.
- [Lockstep decision](docs/decisions/lockstep.md): one action, one logical tick, post-step observation; transport comparison.
- [Learning roadmap](docs/roadmap.md): Q-learning baseline, then Stage 2A-2F toward a toy Dreamer agent, then Minecraft. DQN is optional, not required.
- [Minecraft spike ledger](docs/minecraft-spike.md): Stage 1.5 completion gate, evidence and open work.
- [Simulation throughput decision](docs/decisions/simulation-throughput.md): unpaced lockstep that keeps one step = one tick; profiling and equivalence tests.
- [Recording decision](docs/decisions/recording.md): mandatory low-cost session video, separate from policy input.
- [Parallel workers decision](docs/decisions/parallel-workers.md): isolated workers and deterministic offline development identities.

Privileged Fabric instrumentation stays outside policy inputs, in separate protocol commands and Python types.

## Version decision and evidence

Minecraft 26.3 is the pinned stable release, not a snapshot. The official [Minecraft release post](https://www.minecraft.net/en-us/article/minecraft-java-edition-26-3) and [Fabric 26.3 note](https://fabricmc.net/2026/09/15/263.html) describe the release; the Mojang version manifest declares Java 25. The Fabric project pins Fabric Loader 0.19.5, Fabric API 0.161.0+26.3 and Loom 1.18.2, with Gradle Wrapper 9.7.0. Official, unobfuscated Minecraft 26.3 names need no separate mappings dependency. Version pins are revisitable deliberately, not updated automatically.

Temurin is acquired from the [Adoptium Temurin 25 release](https://github.com/adoptium/temurin25-binaries/releases/tag/jdk-25.0.4.1%2B1). Bootstrap verifies embedded hashes of the selected release files before extraction; uv and Lefthook release binaries are likewise checked against pinned release hashes. The Gradle wrapper distribution has its own committed SHA-256 pin.

## Git checkpoint

The canonical remote is `https://github.com/minge0n/minecraft-agent.git`, branch `main`. A coherent validated unit is checked, diff-reviewed, committed and pushed; `main` is never force-pushed. Stage 1.5 is complete only when every row of the completion gate in the [spike ledger](docs/minecraft-spike.md) is proven. Keep secrets, game files, worlds, JDK/Python runtimes and model artifacts out of Git.
