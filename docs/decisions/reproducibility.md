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

## Running sweeps on macOS

`python -m minecraft_rl.sweep <experiment> --seeds 0-9 --jobs 2 --output-root <dir>` runs one process per seed and limits heat in three ways:

1. `--jobs` limits the number of processes that run at the same time (default 2).
2. Each process runs under `taskpolicy -b`, the background quality of service of macOS. The scheduler then gives these processes low priority, uses lower clock speeds and lets other work go first.
3. Before it starts a seed, the runner reads the thermal pressure level of macOS (`notify_get_state` on `com.apple.system.thermalpressurelevel`: 0 nominal, 1 moderate, 2 heavy). While the level is above `--max-thermal-level` (default 0), it waits. `--cooldown <seconds>` adds a pause after each finished seed.

Because each process uses one thread, none of these changes the results (measured for Stage 2C seeds 0 and 1). Arguments after `--` go to the experiment, for example `-- --world-model rssm`.
