# Mandatory session recording

Status: **decided requirement; not implemented.** The existing probes (`scripts/minecraft-tick-gate-probe.py`, `scripts/minecraft-observation-probe.py`) are non-recording infrastructure tests until the recorder exists. Refines the RGB deferral: rendered frames are required for humans and remain excluded from the policy.

## Principle

Rendered video is a required experiment artifact for human behavioral analysis, never a policy observation.

```text
Minecraft simulation
    -> structured visible observation -> policy
    -> rendered view -> recorder -> video
```

- Lowering video quality does not change the policy observation.
- Encoder lag does not change logical Minecraft timing.
- Recorder failure does not change agent behavior.
- Policy code never reads recorded frames. An RGB-observation experiment would be a separate explicit decision.

## Scope

Every training, evaluation, debugging and integration session that meaningfully runs the environment records automatically, unless the invocation is explicitly marked as a non-recording infrastructure test. One recording per session and episode, failures included; never keep only interesting episodes. Each parallel worker records its own stream; a composite of workers is never the primary artifact. Recordings live under ignored `runs/`, never in Git, and are not deleted automatically without an explicit retention policy.

## Low-cost rendering

Optimize for observability, not visual quality. Initial defaults: render distance 8 chunks, VSync off, fast/minimal graphics, reduced particles and effects, windowed/offscreen-compatible, and the lowest resolution among candidates such as 320x180, 426x240 and 640x360 at which navigation, combat, falls, camera behavior, inventory interactions, crafting attempts, deaths and obvious policy failures stay easy to follow; choose it experimentally. Raise quality only for a concrete debugging or presentation need.

Simulation distance is configured independently and recorded; video cost never reduces simulation semantics. The integrated server derives its view distance from the client render distance, so render distance must stay well above the 32-block sensor reach (8 chunks = 128 blocks) and is recorded as an environment parameter. The mod applies these settings per worker instead of relying on hand-edited shared options.

## Content

The raw video is the agent's first-person camera and an honest view of what Minecraft rendered. It answers what the agent looked at, which actions led to a failure, whether it looped, noticed an entity, fell, and what preceded an advancement or death. Privileged debug visualizations are not drawn into it. A separate annotated debug recording (environment step, logical tick, action, reward, return, advancement event, sensor output, world-model diagnostics) may be added later.

Open question: frames are rendered between client ticks with interpolation, and client views of non-player entities are likely one step behind the server (inferred from source). The tick that a captured frame represents must be defined and measured before claiming frame/tick alignment.

## Simulation time and video time

Simulation ticks, wall-clock execution speed and video playback speed are distinct. The recorder is never the simulation clock and never slows accelerated simulation back to 20 TPS. Policy: capture every N simulated ticks and encode at a fixed playback FPS; candidate rates correspond to 20, 10 or 5 simulated frames per second, possibly higher for combat and fast camera movement, chosen by benchmarking interpretability against cost. Video subsampling never changes the policy step frequency. Metadata records N, playback FPS and the tick of every captured frame; the video preserves simulation order, not wall-clock pacing.

## Asynchronous pipeline

```text
render thread -> bounded frame queue -> recorder/encoder worker -> video file
```

No heavy encoding inside `env.step()`. Measure render, frame readback, queueing, encoding and disk-write cost separately. The queue is bounded; on sustained mismatch the recorder applies deterministic, counted subsampling, never unbounded memory growth and never a silent change of simulation timing. Dropped and subsampled frame counts are metrics.

On encoder failure: mark the recording status as failed in session metadata, keep diagnostics, and continue or terminate according to an explicit configured policy. A session video is never discarded silently.

## Format and encoder

Widely playable output such as MP4/H.264 if a validated encoder is available. The encoder is project-managed or explicitly validated and documented (for example a pinned, checksum-verified binary installed by bootstrap, or a locked Python dependency), not an arbitrary global install. Check codec and encoder licensing (for example, libx264 is GPL) before choosing.

## Metadata, timeline and storage

The monotonic simulation tick is the canonical timeline. Each episode records at least environment/worker ID, session and episode ID, world seed, Git commit, start and end time, simulated ticks, wall-clock duration, mode (training, evaluation, debug), model/checkpoint ID when applicable, recording parameters and recording status. A separate event/action log maps video frame to tick, action, reward, structured observation and checkpoint; transition data is not duplicated inside the video container. Candidate layout (may change if a cleaner one emerges):

```text
runs/<run-id>/metadata.json
runs/<run-id>/workers/env-NNNN/episodes/NNNNNN/video.mp4
runs/<run-id>/workers/env-NNNN/episodes/NNNNNN/metrics.json
```

Storage accounting from the start: video bytes per simulated hour and per episode, encoding bitrate, capture rate. Compression favors research usability and storage efficiency over presentation quality.

## Live monitoring

It must remain possible to watch one or more running workers without changing policy inputs or simulation semantics, but viewing is never required for correctness. A tiled multi-worker preview is desirable later and separate from archival recording.

## Benchmarks

Once accelerated stepping exists, measure one environment in four configurations: simulation only; plus rendering; plus capture; plus encoding. Repeat representative cases with several concurrent workers to identify whether simulation, rendering, capture, encoding, disk I/O or the Dreamer learner dominates.
