# Mandatory session recording

Status: **implemented and runtime-validated on 26.3 (macOS arm64) for a single worker.** See Implementation and Results. Recording is opt-in per invocation today: `scripts/minecraft-recording-probe.py` records; the other probes remain non-recording infrastructure tests. Refines the RGB deferral: rendered frames are required for humans and remain excluded from the policy.

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

Definition (implemented): a captured frame belongs to the client tick it was captured after. It is the first frame drawn after that client tick completed, and `frames.jsonl` records its client tick, world game time and `partial_tick` (interpolation fraction, 0-1). With lockstep the next tick does not start until the next STEP, so interpolation between two ticks shows only already-simulated states. Non-player world state in the frame can lag the server by one step (`docs/decisions/simulation-throughput.md`, client world copy).

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

## Implementation

- **Encoder decision.** PyAV 18.1.0 (locked in `uv.lock`) with the macOS VideoToolbox H.264 hardware encoder `h264_videotoolbox`, MP4 container, NV12, 600 kbit/s, 20 playback FPS. Alternatives: `libx264` (GPL build inside PyAV's wheel), `libvpx-vp9` (BSD, slower, WebM). Chosen by the developer for speed on the current host. Consequence: Linux hosts need a different encoder; not implemented. PyAV's PyPI wheels bundle an FFmpeg build that includes GPL components (libx264, libx265); the project uses them locally and does not redistribute binaries.
- **Capture (mod, render thread).** `FrameRecorder` hooks `Minecraft.renderFrame` just before the frame's command submit. Every client tick is considered once, in the first frame drawn after it; every `every_ticks`-th tick from the start of the episode is due. A due frame is downscaled on the GPU with the vanilla `TRACY_BLIT` screen-quad pipeline into an RGBA texture and copied into one of four readback buffers. The copy completes asynchronously through a GPU fence and is read in a later frame. The render thread never waits for the GPU or the network.
- **Bounded queue and sender.** Read-back frames go into a bounded queue (`queue_frames`); a daemon thread streams them to the recorder over loopback TCP (`FrameStream`, versioned binary format). A frame that finds no free readback buffer, finds the queue full, or arrives after a failure is dropped and counted (`dropped_gpu_busy`, `dropped_queue_full`, `dropped_after_failure`, `missed_unrendered`). Recording never blocks or delays a STEP.
- **Recorder (Python).** `minecraft_rl.recording.Recorder` listens on a loopback port and encodes one episode per connection on a background thread. It writes `video.mp4`, `frames.jsonl` (video frame -> client tick, game time, partial tick) and `metrics.json` (session context, capture counters, encoding summary, `recording_status`). A stream that ends without END, an encoder error or a capture failure marks the episode `failed`; nothing is discarded.
- **Protocol.** v2 `RECORD_START {episode, port, every_ticks, width, height, queue_frames}` returns once the episode's first tick has been considered; v2 `RECORD_STOP` returns the capture counters after every readback has finished and END has been sent. `RESET` and `PACING` with `render_frames=false` are rejected while recording. `STATUS` reports the recorder state. Recording requires rendered frames, so it is incompatible with no-render stepping.
- **Layout.** `runs/<run>/metadata.json` and `runs/<run>/workers/env-NNNN/episodes/NNNNNN/{video.mp4,frames.jsonl,metrics.json}`.

## Results (macOS arm64, seed 12345, unpaced, window framebuffer 1708x960)

`scripts/minecraft-recording-probe.py`, runs `runs/minecraft-recording/{r1,r2-640x360-e1,r3-320x180-e2,final}`. All passed.

- **No effect on steps.** A recorded controlled replay (243 steps) matched an unrecorded one exactly: zero differing policy observations and no unexplained privileged fields, one world tick and one client tick per STEP.
- **Completeness.** Every due frame was written (244 of 244 in the replay, 401 of 401 in each benchmark episode), with strictly increasing, consecutive client ticks and no drops. GPU readback latency was 13-15 ms on average and at most 44 ms. The queue high-water mark was 1-4 frames.
- **Failure isolation.** With the recorder connection closed at start, the mod reported `recording_status=failed` (`Broken pipe`), counted 200 dropped frames, and all 200 steps still advanced exactly one tick.
- **Cost (400 unpaced steps, 426x240, every tick; run `final`):**

| Configuration | Steps/s | Median step ms | p95 step ms |
| --- | --- | --- | --- |
| simulation only (no rendering) | 141.0 | 6.73 | 9.04 |
| + rendering | 62.8 | 16.20 | 22.92 |
| + capture (frames discarded) | 62.1 | 16.47 | 26.78 |
| + encoding | 62.4 | 16.26 | 25.63 |

Rendering costs roughly half of unpaced throughput; capture and encoding add little on top (1.7-2.1 ms encoder time per frame on a separate thread). At 640x360 every tick throughput was about 63 steps/s; at 320x180 every 2nd tick about 58-62 steps/s. The step-rate differences between these settings are within run-to-run variation.

- **Storage.** 426x240 every tick: about 1.2 MB per 400-tick episode, about 264 MB per simulated hour. 640x360: about 269 MB per simulated hour at the same bit rate. 320x180 every 2nd tick: about 133 MB per simulated hour.
- **Default.** 426x240, every tick, 600 kbit/s: the frames stay legible for navigation, combat and camera motion, and cost no more than lower resolutions on this host. Revisit when multiple workers compete for the GPU and encoder.

Not yet covered: multiple concurrent workers, long episodes, Linux encoders, and an explicit retention policy (none exists; nothing is deleted automatically).
