# Stage 2A: neural-network sanity check

**Purpose.** Before any recurrent model or world model, prove that the local PyTorch environment trains a randomly initialized network correctly and inspectably: gradients are right, every parameter moves, training converges, checkpoints restore exactly, and device selection works. Not DQN and not reinforcement learning; there is no environment, reward or policy.

## Task and model

**Established behavior.** The task is 4-bit parity: all 16 patterns `b in {0,1}^4`, label `y = (b1 + b2 + b3 + b4) mod 2`. Inputs are encoded as `x = 2b - 1 in {-1, +1}^4`, so the full dataset is a tensor `x` of shape `(16, 4)` and labels `y` of shape `(16,)`. Parity is not linearly separable: a linear model with no hidden layer stays at loss `ln 2 = 0.6931` and 37-56% accuracy after 500 steps for seeds 0-9 (measured once), so solving it demonstrates that the hidden layer learns.

`ParityMLP` (`src/minecraft_rl/parity.py`):

```text
x (16, 4) -> h = tanh(W1 x + b1) (16, 16) -> z = W2 h + b2 (16, 2)
```

Learned parameters: `W1` (16, 4), `b1` (16), `W2` (2, 16), `b2` (2); 114 in total, PyTorch default (Kaiming-uniform) initialization from `torch.manual_seed(seed)`.

Loss: mean cross-entropy over the full batch, `L = -(1/16) sum_i log softmax(z_i)[y_i]`, where `z_i` are the two logits for pattern `i`. Adam, learning rate 0.01, 500 full-batch steps; each step is `zero_grad`, forward, `loss.backward()` (reverse-mode autograd fills `.grad` for all four parameter tensors) and `optimizer.step()`.

## Checks

- **Gradient check.** `torch.autograd.gradcheck` compares the autograd gradient of `L` with respect to all 114 parameters against central finite differences, in float64 on a CPU copy (float32 differences are too coarse).
- **Parameter change.** After the first step every parameter tensor has a nonzero gradient norm and a nonzero update norm.
- **Convergence.** 100% accuracy and final loss below 10% of the initial loss.
- **Checkpoint.** `checkpoint.pt` stores format tag, config, step, model and optimizer state and is loaded with `weights_only=True`. Reloaded outputs are bit-identical; training 20 steps, saving, reloading and training 20 more produces parameters identical to 40 uninterrupted steps (unit test).
- **Device.** `--device auto|cpu|mps|cuda`; an unavailable device is an error, not a silent fallback.

Unit tests (`tests/test_parity.py`) run all of these on CPU in about a second and are part of `./scripts/check`.

## Run

```sh
.venv/bin/python -m minecraft_rl.parity --seed 0 --device cpu --output runs/stage2a/cpu-seed0/metrics.json
```

It prints the parameter shapes, the gradient check, per-parameter first-step gradient and update norms, the loss curve, every prediction and the checkpoint round trip, and writes `metrics.json` (config, seed, device, Python and PyTorch versions, architecture, parameter count, loss curve, Adam moment norms, predictions) plus `checkpoint.pt` under the ignored run directory.

## Results (macOS arm64, Python 3.12.11, PyTorch 2.14.0, NumPy 2.5.3)

| Seed | Gradient check | Loss step 0 | Loss step 50 | Loss step 100 | Final loss | Accuracy | Checkpoint identical |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | pass | 0.733 | 0.428 | 0.0278 | 0.00041 | 100% | yes |
| 1-9 | pass | | 0.136-0.477 | 0.0092-0.0453 | 0.00028-0.00066 | 100% | yes |

Seed 0 on MPS reached the same final loss as CPU to four significant digits (0.000411158 vs 0.000411173); the difference is float32 kernel rounding, so CPU and MPS runs are not bit-identical. MPS took 7.2 s against 1.2 s on CPU for this tiny model, dominated by device startup and per-step launch overhead; CPU is the default here.

## Interpretation and limits

This proves the toolchain and the training mechanics, nothing about sequence memory, dynamics or RL. The dataset is the whole input space, so there is no train/test split and 100% accuracy is memorization of a fixed mapping by design. Next is Stage 2B: a small recurrent model on a toy sequence task (`docs/stage2b.md`).
