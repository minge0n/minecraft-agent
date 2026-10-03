# Decision: canonical runtime for the Stage 2 toy experiments

Status: decided and measured. It replaces the earlier practice of running each experiment with the PyTorch defaults.

## Problem

The Stage 2F seed-level results in `docs/stage2f.md` (first version) did not reproduce. Seed 6 of the integrated loop succeeded in 50% of real episodes when it ran with the default thread count and in 100% when it ran with one thread. The cause is the order of floating-point additions. PyTorch splits some CPU operations (for example large sums) over its intra-op threads, and the result of a float sum depends on the order of the additions. A difference in the last bit changes a sampled action at some point. The trajectory then diverges, and with it the replay data and the final policy.

Measured on this machine (Apple M5 Pro, 18 cores, macOS 27.0, PyTorch 2.14.0): a sum over 2,097,152 random floats gives `-2260.46435546875` with 1 or 2 intra-op threads and `-2260.464599609375` with 6. A matrix product of the same size gives the same value with every thread count. The small Stage 2C training run gave bit-identical losses with 1 and 6 threads. A short 3-iteration Stage 2F loop gave different world-model and agent weights with 1 and 6 threads, but identical weights in two runs with the same thread count. Load from other processes did not change the result: 4 concurrent copies with 1 thread and 4 with 6 threads each matched their own thread count.

## Decision

`src/minecraft_rl/runtime.py` is the one place that sets the runtime. Every experiment command calls `runtime.configure(seed)` directly after it parses its arguments, before it creates any tensor. The function sets:

1. One intra-op thread (`torch.set_num_threads(1)`).
2. One inter-op thread (`torch.set_num_interop_threads(1)`). The toy experiments run no parallel graph work, so a second inter-op thread cannot help them. Measured: the inter-op count had no effect on the results or on the speed of the Stage 2C run.
3. The seeds of the Python `random` module, NumPy and PyTorch, all equal to the run seed. The experiments also give an explicit `torch.Generator` to every sampling call. The global seeds only protect against a sampling call that has no generator.
4. `torch.float32` as the default dtype.
5. `torch.use_deterministic_algorithms(True)`. On the CPU it has no visible effect for these models. It makes PyTorch raise an error if a later change uses an operation without a deterministic implementation.

`runtime.metadata(device)` writes these settings into the `runtime` field of every `metrics.json`: the seed, both thread counts, the deterministic mode, the dtype, the device, the Python, PyTorch and NumPy versions, the operating system, the CPU type and core count, and the `OMP_NUM_THREADS` and `MKL_NUM_THREADS` variables. The field `canonical` is true only when the seed is set and both thread counts are 1.

PyTorch accepts a new inter-op count only before its first parallel work. `configure` therefore runs at the start of `main`. If a process already did such work, the inter-op count stays as it is, and the metadata shows the count in use.

One thread is also fast enough. The Stage 2C training run took 1.2 s with 1 thread and 1.4 s with 6 threads, because the tensors are too small for parallel work to pay off.

## Scope

The target is stable results for repeated runs on the same documented hardware and software. Results on a different CPU, operating system or library version can differ in the last bits and then diverge. Compare such results with the recorded `runtime` field before you compare the numbers. Results on the MPS or CUDA devices are not covered.

## Evidence

- `tests/test_runtime.py` (part of `./scripts/check`) starts two separate Python processes that train a small world model and agent with the same seed, and requires identical weights.
- Every canonical Stage 2C, 2D, 2E and 2F run for seeds 0-9 ran twice, in separate processes. The `metrics.json` files are identical except for the time stamp, the duration and the file paths: 10 of 10 seeds for each of the four stages.

## Running sweeps with little heat

`python -m minecraft_rl.sweep <experiment> --seeds 0-9 --output-root <dir>` runs one process per seed on a Mac. It supports macOS only. Arguments after `--` go to the experiment, for example `-- --world-model rssm`. The runner limits heat in four ways:

1. It runs one seed process at a time (`--jobs`, default 1).
2. Each process runs under `taskpolicy -b`, the background quality of service of macOS. macOS then runs it on the efficiency cores at a low clock speed.
3. `--duty-cycle` (default 0.5) limits the share of each period of `--period` seconds (default 1) in which the process computes. The runner stops the process group with SIGSTOP for the rest of the period and resumes it with SIGCONT.
4. A thermostat reads the die temperature of the SoC once per period. The value is the highest "PMU tdie" sensor of the IOKit HID event system (`src/minecraft_rl/macos_thermal.py`), which needs no root rights. If the temperature is above `--max-temperature` (default 65 C), or the macOS thermal pressure level is above `--max-thermal-level` (default 0, nominal), the runner stops all seed processes. It resumes them at or below `--resume-temperature` (default 3 C lower), or after `--max-pause` seconds (default 120). The limit exists because other programs can keep the chip warm, and the sweep must not wait forever.

The runner skips a seed whose `metrics.json` exists, so the same command resumes an interrupted sweep. `--rerun` runs such seeds again. The runner deletes an old `metrics.json` before it starts a seed, so a seed that stops early leaves no stale result. If the runner gets SIGINT or SIGTERM, it resumes and then terminates every unfinished seed process.

Next to each `metrics.json`, the runner writes `sweep.json`: the command, the exit code, `jobs`, `duty_cycle`, the period, the thermostat limits, the wall-clock time, the time stopped by the duty cycle, the thermostat pause time and number of pause events, the mean and highest die temperature, the thermal pressure level at start and end, and the thread counts that the experiment recorded.

### Measured heat

Die temperature on the machine above, for an RSSM world-model run with one thread, mean of the last 60 s of a 150 s window:

| Setting | Mean | Highest |
|---|---|---|
| Idle between runs | 49-58 C | - |
| Normal priority, duty cycle 1.0 | 63.5 C | 66.5 C |
| `taskpolicy -b`, duty cycle 1.0 | 56.5 C | 62.8 C |

Background QoS removes most of the heat that one seed process adds. The idle temperature varies by about 9 C with the other programs on the machine, so the remaining heat of a sweep is small next to that variation. A thermostat limit below the idle temperature stops the sweep at every period: with a limit of 56 C, a 300-step run took 265 s instead of 15 s and its mean temperature was not lower. The defaults therefore keep the duty cycle as a fixed reduction and use the thermostat as a guard against high temperatures.

### Evidence for the sweep runner

SIGSTOP and SIGCONT change only when a process computes, not what it computes. Each process uses one thread and seeded generators, so the result must not change. Measured on the machine named in the Problem section:

| Run | Reference (duty cycle 1.0) | Throttled | Result |
|---|---|---|---|
| RSSM `world_model`, seed 0, 1,500 steps | 50 s | 101 s (duty cycle 0.5) | 608 metric fields identical, 131 checkpoint tensors identical |
| RSSM `dreamer_loop`, seed 1, 2 iterations | 79 s | 156 s (duty cycle 0.5) | 573 of 575 metric fields identical (the 2 others are checkpoint paths), world-model and agent weights identical |
| RSSM `world_model`, seed 0, 300 steps, with a 6 s thermal pause | 14 s | 20 s (duty cycle 1.0 and the pause) | 476 metric fields identical |
| RSSM `world_model`, seed 0, 300 steps, thermostat at 56 C | 15 s | 265 s (duty cycle 0.5, 15 thermostat pauses) | 638 metric fields identical |

The comparison ignores only the time stamp, durations and file paths. Other process-level checks:

1. A thermal pause, with a heavy level faked for 6 s, put the seed process into the stopped state (`T`) and back to running (`R`). `sweep.json` recorded 1 event and 6.0 s.
2. SIGTERM and SIGINT to the runner while the seed process was stopped left no process behind.
3. After the interruption, the same command finished the sweep. A second call skipped both seeds, and `--rerun` ran seed 1 again.
