# Stage 2B: recurrent-model sanity check

**Purpose.** Prove that a small recurrent network trained from random initialization carries information from an earlier input to a later prediction, the capability the Dreamer recurrent state will need to remember things that are no longer visible. Supervised only; no environment, reward or policy.

## Task

**Established behavior.** Cue recall over a delay. A sequence is

```text
cue, distractor x D, query        (length D + 2)
```

with vocabulary `A=0, B=1, DISTRACTOR=2, QUERY=3`. The cue is A or B with equal probability; the target is the cue, predicted at the query step. For a given D, the sequences for A and for B are identical from step 1 onward, so any prediction better than 50% requires memory of step 0. Training samples a fresh batch of 64 sequences per step with one delay drawn uniformly from 1-10. Evaluation uses both cues at fixed delays 1, 5, 10, 20 and 40; 20 and 40 are longer than any training sequence.

## Model

`CueRecallGRU` (`src/minecraft_rl/cue_recall.py`):

```text
tokens (B, T) -> embedding e_t (B, T, 8) -> GRU h_t (B, T, 16) -> logits z = W h_T + b (B, 2)
```

Here `B` is the batch size and `T = D + 2` the sequence length. The GRU updates its state `h_t` from the previous state and the current input:

```text
r_t = sigmoid(W_ir e_t + b_ir + W_hr h_{t-1} + b_hr)          reset gate
u_t = sigmoid(W_iz e_t + b_iz + W_hz h_{t-1} + b_hz)          update gate
n_t = tanh(W_in e_t + b_in + r_t * (W_hn h_{t-1} + b_hn))     candidate
h_t = (1 - u_t) * n_t + u_t * h_{t-1}                         h_0 = 0
```

`sigmoid` and `tanh` act elementwise and `*` is elementwise multiplication. An update gate `u_t` near 1 copies the previous state forward, which is how the cue can survive many distractor steps. Learned parameters: embedding (4, 8); GRU input weights (48, 8), recurrent weights (48, 16) and two biases (48), each stacking the reset, update and candidate blocks; readout (2, 16) and (2). 1,314 in total.

Loss: cross-entropy of `z` against the cue, only at the query step. `loss.backward()` propagates through the readout and back through every GRU step to the cue embedding (backpropagation through time). Adam, learning rate 0.01, 1,200 steps.

**Control.** `CurrentTokenOnly` (210 parameters) sees only the final token (always QUERY), passed through an embedding, a tanh layer and a readout, and is trained identically. It must stay at 50%; if it did not, the task would leak the answer.

## Checks

- Accuracy and loss at delays 1-40, with the per-cue probability `P(B | cue)`.
- **Cue gradient.** Norm of `d(z_B - z_A) / d(embedding at step 0)`, before and after training: how strongly the prediction depends on the first input through the recurrent state. The logit margin is used instead of the loss, whose gradient vanishes once predictions are confident.
- **State separation.** `||h_t(A) - h_t(B)||` for every step. The inputs are identical after step 0, so any distance is information about the cue held in the state.
- Checkpoint round trip, loaded with `weights_only=True`, and a unit test that resuming from a checkpoint matches uninterrupted training exactly.

Unit tests: `tests/test_cue_recall.py` (about 2 s on CPU, part of `./scripts/check`).

## Run

```sh
.venv/bin/python -m minecraft_rl.cue_recall --seed 0 --device cpu --output runs/stage2b/seed0/metrics.json
```

## Results (macOS arm64, Python 3.12.11, PyTorch 2.14.0, CPU, about 2.5 s per seed)

Seed 0: training loss 0.694 at step 0, 0.0008 at step 50, 0.000014 at step 1,199.

| Delay | Accuracy before | GRU accuracy | P(B given A) | P(B given B) | Cue gradient before -> after | No-memory accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0% | 100% | 0.000 | 1.000 | 4.2e-2 -> 0.59 | 50% |
| 5 | | 100% | 0.000 | 1.000 | 1.5e-2 -> 0.49 | 50% |
| 10 | | 100% | 0.000 | 1.000 | 3.9e-3 -> 0.44 | 50% |
| 20 | | 100% | 0.000 | 1.000 | 2.9e-4 -> 0.37 | 50% |
| 40 | 50% | 100% | 0.000 | 1.000 | 1.7e-6 -> 0.31 | 50% |

State separation at delay 40 rises from 4.1 at the cue step to about 6.6 and stays there through all 40 distractors (7.5 at the query).

Seeds 0-9: every seed reaches 100% at all five delays including 40, the control is exactly 50% for every seed and delay, final training loss 0.000009-0.000035, the delay-40 cue gradient grows from 3e-10-1.7e-6 before training to 0.03-0.31 after, and every checkpoint round trip is identical.

**Interpretation.** Before training, the influence of the cue on the final prediction shrinks sharply with distance (delay 40: about 1e-6), the vanishing-gradient behavior of an untrained recurrent network. Training makes the GRU hold the cue in a stable state direction, so the dependence survives delays four times longer than any it was trained on. The control's exact 50% shows that nothing except the recurrent state carries the answer.

## Limits

One bit of memory, a single distractor token and a supervised target: this shows the mechanism, not a memory capacity or partial-observability result. No actions, no dynamics and no uncertainty are involved yet. Next is Stage 2C: learned dynamics in a tiny environment (`docs/roadmap.md`).
