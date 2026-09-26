# Minecraft RL Lab

This repository currently contains reproducible development tooling, a minimal Fabric mod, and a small tabular reinforcement-learning experiment. It does **not** contain a Minecraft agent or a Python/Fabric bridge.

## Setup and commands

On macOS or Linux (arm64/x86_64), install the bootstrap prerequisites: POSIX shell, curl, tar, host Python 3 and Git. Network access is required for downloads. The host Python is used only for checksum verification; project code always uses the local managed interpreter. From a clone:

```sh
./scripts/bootstrap
./scripts/check
./.tools/lefthook run pre-commit
.venv/bin/python -m minecraft_rl.train --episodes 1000 --seed 42
./scripts/gradle build
```

Bootstrap downloads SHA-256-verified uv 0.12.19, Temurin 25.0.4.1+1 and Lefthook 2.1.14 into ignored `.tools/`, installs uv-managed CPython 3.12.11 in `.tools/uv-python`, syncs the locked dev environment to `.venv/`, and installs Git hooks when inside a Git checkout. It does not install any global Python package, JDK or Gradle. Re-run bootstrap after dependency changes and commit the updated `uv.lock`. Use `./.tools/uv lock` to update the lock deliberately. `./scripts/check` runs Ruff format check, Ruff lint and pytest; the same quick checks run from the Lefthook pre-commit hook. Training and Fabric builds are explicit, not commit hooks. Use `./.tools/lefthook install` after initializing Git in a non-clone directory. Java/Fabric changes should additionally pass `./scripts/gradle build`.

The Gradle wrapper is committed under `minecraft/fabric-mod/`. Always invoke it through `./scripts/gradle`: the script sets `JAVA_HOME` and `PATH` to the pinned local JDK and `GRADLE_USER_HOME` to `.runtime/gradle`. Loom's client and server run directories point to `.runtime/minecraft/client` and `.runtime/minecraft/server`; no personal Minecraft directory is read. Minecraft files are obtained through normal Loom development tooling. A Fabric `build` is not a game-launch/authentication test.

The GridWorld run prints the learned Q-table (UP, RIGHT, DOWN, LEFT for each state) and a greedy policy map, and writes full-precision values plus metrics into an ignored per-run `runs/gridworld-.../metrics.json` by default. To pick a location, pass `--output runs/my-run/metrics.json`. Pass `--seed` to reproduce exploration and `--episodes` to control training length. Evaluation runs for 100 episodes with deterministic greedy actions and without Q-table updates; it is not a held-out-world-seed evaluation because this toy world is deterministic. Runs record the commit when available (null if there is no commit), seed, environment, architecture, parameters, hyperparameters, steps, duration, reward, and success metrics. See [Stage 1](docs/stage1.md) for the exact map, update and interpretation.

## Layout and boundaries

- `src/minecraft_rl/gridworld.py`: deterministic temporary discrete environment.
- `src/minecraft_rl/q_learning.py`: tabular action-value estimates and update rule.
- `src/minecraft_rl/train.py`: separate training/evaluation loops and run metadata.
- `tests/`: deterministic checks for the current behaviors.
- `minecraft/fabric-mod/`: Fabric integration skeleton only; no IPC or policy access to game state.
- `scripts/`: local bootstrap, fast checks and pinned-JDK Gradle entrypoint.
- `docs/`: design and experiment interpretation.

The future Minecraft policy may receive permitted visual observations and scalar rewards; diagnostic or privileged Fabric instrumentation must stay outside policy inputs. When a bridge is needed, first compare localhost TCP (portable and easy to inspect, requires a port), Unix sockets (local-only, but platform/path constraints), and WebSocket (browser-friendly but added framing/dependency cost). Define a versioned `reset`/`step` exchange with `observation`, `reward`, `terminated`, `truncated` and diagnostic `info`, and review every policy-visible field. No transport is selected or implemented now.

## Version decision and evidence

Minecraft 26.3 is the pinned stable release, not a snapshot. The official [Minecraft release post](https://www.minecraft.net/en-us/article/minecraft-java-edition-26-3) and [Fabric 26.3 note](https://fabricmc.net/2026/09/15/263.html) describe the release; the Mojang version manifest declares Java 25. The Fabric skeleton pins Fabric Loader 0.19.5, Fabric API 0.161.0+26.3 and Loom 1.18.2 in its Gradle properties, with Gradle Wrapper 9.7.0. Its Java bytecode targets version 25. Official, unobfuscated Minecraft 26.3 names do not need a separate Yarn mappings dependency. Version pins are revisitable deliberately, not updated automatically.

Temurin is acquired from the [Adoptium Temurin 25 release](https://github.com/adoptium/temurin25-binaries/releases/tag/jdk-25.0.4.1%2B1). Bootstrap verifies embedded hashes of the selected release files before extraction; uv and Lefthook release binaries are likewise checked against pinned release hashes. The Gradle wrapper distribution has its own committed SHA-256 pin. The lock records Python package resolution; downloads still require availability of the upstream hosts.

## Git checkpoint

The canonical remote is `https://github.com/minge0n/minecraft-agent.git`, branch `main`. A coherent validated unit is checked, diff-reviewed, committed and pushed; `main` is never force-pushed. Stage 1.5 is **not** complete merely because Fabric compiles or one layer is stepped: the [spike validation ledger](docs/minecraft-spike.md) lists what the automated world-tick probe (`.venv/bin/python scripts/minecraft-tick-gate-probe.py`) proves and what remains unproven. Keep secrets, game files, worlds, JDK/Python runtimes and model artifacts out of Git.

## Next milestones

A measured Stage 1.5 contract is required before considering later ML stages. No neural network is trained here. The tiny environment demonstrates the learning loop before function approximation: actions change state, obstacles block movement, reaching the goal yields one reward, and training modifies a table of 16 states × 4 actions. This is temporary scaffolding, not a target observation/action interface. Future stages require an explicit request and evidence for added complexity. For every future reward, ask which useless or reset-farming behavior might maximize it.
