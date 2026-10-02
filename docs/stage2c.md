# Stage 2C: learned dynamics in a T-maze

**Purpose.** Learn a world model from real transitions of a tiny partially observable environment: given the history of observations and actions, predict the next observation, the reward and whether the episode continues. This is the supervised core of Dreamer's world model, without a stochastic latent, imagination or a policy. The same T-maze is reused in Stages 2D-2F.

## Environment

`TMaze` (`src/minecraft_rl/tmaze.py`), corridor length `L = 4`, step limit 40.

```text
position:   0          1 .. L       L + 1
            cue cell   corridor     junction --LEFT--> left arm
                                             --RIGHT-> right arm
```

- Observations (categorical): `CUE_LEFT=0`, `CUE_RIGHT=1`, `CORRIDOR=2`, `JUNCTION=3`, `ARM=4`. The cue cell shows the cue; every corridor cell looks the same; nothing shows the position.
- Actions: `FORWARD=0`, `LEFT=1`, `RIGHT=2`. `FORWARD` moves one cell toward the junction and stays at the junction. `LEFT`/`RIGHT` bump into the wall and stay, except at the junction, where they enter an arm and terminate with reward `+1` on the cued side and `-1` on the other. All other rewards are 0. Reaching 40 steps truncates.
- The cue is drawn uniformly per episode.

Two transitions are not predictable from the current observation and action alone:

1. **Junction turn reward:** `JUNCTION` looks the same for both cues, so the sign of the turn reward requires remembering the observation from the first step.
2. **Junction arrival:** `CORRIDOR + FORWARD` leads to another `CORRIDOR` or to the `JUNCTION` depending on the position, which requires counting earlier forward moves (bumps do not count).

Continuation is predictable without memory (any turn at a visible junction ends the episode).

## Data

A uniformly random behavior policy collects 512 training episodes (seed `s`) and 256 held-out evaluation episodes (seed `s + 1,000,000`). Seed 0: 8,521 training transitions, mean episode length 16.6, 252 `+1` and 259 `-1` rewards; every evaluation episode terminates with a turn. Episodes are padded to 40 steps with a validity mask `m_t`. The cue is stored with each episode only to break evaluation results down by cue; it is never a model input.

## Model

`WorldModel` (`src/minecraft_rl/world_model.py`). For step `t` of an episode, `o_t` is the observation, `a_t` the action, and the targets are the next observation `o_{t+1}`, the reward `r_{t+1}` and the continuation flag `c_{t+1}` (0 when the step terminated the episode, else 1).

```text
x_t = [onehot(o_t), onehot(a_t)]          (B, T, 8)
h_t = GRU(h_{t-1}, x_t), h_0 = 0          (B, T, 32)   deterministic recurrent state
next-observation logits = W_o h_t + b_o   (B, T, 5)
predicted reward        = w_r h_t + b_r   (B, T)
continuation logit      = w_c h_t + b_c   (B, T)
```

`B` is the batch of 32 episodes and `T = 40` steps. The GRU equations are the ones in `docs/stage2b.md`; here `h_t` summarizes the whole history `o_0, a_0, ..., o_t, a_t`, which is what Dreamer's deterministic state does. Learned parameters: GRU input weights (96, 8), recurrent weights (96, 32) and two biases (96); heads (5, 32)+(5), (1, 32)+(1), (1, 32)+(1). 4,263 in total.

Loss, averaged over valid steps (`N = sum of m_t`):

```text
L = (1/N) sum_t m_t [ -log p(o_{t+1} | h_t)                                  observation cross-entropy
                      + (r_hat_t - r_{t+1})^2                                reward squared error
                      - c_{t+1} log sigmoid(z_t) - (1 - c_{t+1}) log(1 - sigmoid(z_t)) ]   continuation BCE
```

`p(. | h_t)` is the softmax of the observation logits and `z_t` the continuation logit. Padding has `m_t = 0` and contributes neither loss nor gradient. `loss.backward()` reaches every head, then flows back through all GRU steps (backpropagation through time), so the turn-reward error at the junction trains the recurrent weights to keep the cue from step 0. Adam, learning rate 0.003, 5,000 steps of 32 episodes sampled with replacement.

**Control.** The same heads on `h_t = tanh(W x_t + b)` (519 parameters), which sees only the current observation and action and is trained identically. It must fail exactly on the two memory-dependent transitions and match the GRU elsewhere.

## Checks

- Held-out loss per term, next-observation accuracy and continuation accuracy over all valid steps, at training steps 0, 100, 200, 500, 1,000, 2,000, 3,000, 4,000 and 5,000.
- **Junction turn:** reward squared error, reward sign accuracy and the mean predicted reward for each (cue, turn) pair.
- **Junction arrival:** next-observation accuracy on `CORRIDOR + FORWARD` steps that reach the junction.
- Checkpoint round trip (`weights_only=True`) and a unit test that resuming from a checkpoint matches uninterrupted training exactly.

Unit tests: `tests/test_world_model.py` (about 1.6 s on CPU, part of `./scripts/check`): environment transitions and rewards, collected-data consistency, that padding does not affect the loss, a short training run on `L = 1` where the GRU learns both memory-dependent transitions and the control does not, and checkpoint resume.

## Run

```sh
.venv/bin/python -m minecraft_rl.world_model --seed 0 --device cpu --output runs/stage2c/seed0/metrics.json
```

## Results (macOS arm64, Python 3.12.11, PyTorch 2.14.0, CPU, about 14 s per seed alone)

Seed 0, held-out episodes (4,090 transitions, 256 junction turns, 256 junction arrivals), GRU | no-memory control:

| Step | Total loss | Observation accuracy | Turn reward squared error | Turn reward sign accuracy | Junction arrival accuracy |
| --- | --- | --- | --- | --- | --- |
| 0 | 2.409 / 2.478 | 6% / 10% | 1.013 / 1.016 | 48% / 48% | 0% / 0% |
| 200 | 0.172 / 0.293 | 100% / 94% | 1.006 / 0.998 | 54% / 52% | 99% / 0% |
| 1,000 | 0.067 / 0.207 | 100% / 94% | 1.013 / 1.003 | 47% / 48% | 100% / 0% |
| 2,000 | 0.029 / 0.204 | 100% / 94% | 0.415 / 1.004 | 89% / 48% | 100% / 0% |
| 3,000 | 0.001 / 0.204 | 100% / 94% | 0.001 / 1.000 | 100% / 52% | 100% / 0% |
| 5,000 | 0.0001 / 0.203 | 100% / 94% | 0.0002 / 0.996 | 100% / 52% | 100% / 0% |

Mean predicted turn reward at step 5,000 (GRU | control): cue left, turn left `+1.006 | +0.006`; cue left, turn right `-0.996 | +0.084`; cue right, turn left `-0.994 | +0.006`; cue right, turn right `+1.007 | +0.084`. Continuation accuracy is 100% for both models.

Seeds 0-9: every seed ends with 100% held-out observation and continuation accuracy, 100% turn reward sign accuracy, 100% junction arrival accuracy, total loss 0.0001-0.0006 and turn reward squared error 0.0001-0.0045; every checkpoint round trip is identical. The control ends at total loss 0.190-0.203, 94% observation accuracy (it misses exactly the arrivals), 0% junction arrival accuracy, turn reward squared error 0.996-1.017 and sign accuracy 45-53% for every seed. The step at which turn-reward sign accuracy first exceeds 80% varies from 2,000 to 5,000 across seeds.

**Interpretation.** Memory-free transitions are learned within about 200 steps. Junction arrival, which needs a count of forward moves over a few steps, follows at about the same time. The cue-dependent turn reward stays at chance, with the prediction near the average reward 0, for 1,000-4,000 steps, then is learned abruptly. Its signal is one transition per episode, and the information is up to 40 steps back, so the gradient that would make the GRU keep the cue is small until the state starts to separate the cues. Once learned, the held-out error is near zero. The control's chance-level turn reward and zero arrival accuracy show that only the recurrent state carries this information. The late, seed-dependent onset is a warning for later stages: a world model can look nearly converged on the total loss (0.067 at step 1,000) while still being wrong on the only reward that matters, so the task-relevant breakdown is reported separately from the total.

## Limits

One-step prediction with teacher forcing only: the model always reads real observations, so this does not yet measure compounding error in open-loop rollouts. The state is fully deterministic; there is no stochastic latent, no KL term and no uncertainty, which matters once the model must imagine without observations. The environment is deterministic and the data come from a random policy. Next is Stage 2D: imagination and compounding-error measurement on this T-maze (`docs/roadmap.md`).
