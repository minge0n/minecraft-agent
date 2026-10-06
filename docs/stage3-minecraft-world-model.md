# Stage 3: Minecraft RSSM world model smoke test

Purpose: learn short-horizon dynamics from real structured Minecraft transitions with the Stage 2G RSSM (`docs/stage2g.md`), and compare the model against trivial baselines. No actor is trained. No reward exists: the reward design is a separate later milestone.

Status: in progress. This document records the model size selection, the tests and the tiny-overfit test. The held-out results follow in a later section when the training run is complete.

## Data

`src/minecraft_rl/minecraft_collect.py` collected the data under `runs/stage3/data` (ignored by Git). Observation schema `visible-field-v2` (`docs/decisions/observation.md`). The collection policy is the persistent random exploration policy in `src/minecraft_rl/exploration.py`. All sessions have a video.

| Split | Episodes | Transitions | Question |
|---|---|---|---|
| `train` | 25 | 36,794 | Fit data. |
| `eval_episode` | 6 | 9,000 | The same world seeds as `train`, other exploration seeds: unseen trajectories. |
| `eval_seed` | 6 | 9,000 | Held-out world seeds: unseen worlds. |

Splits are per episode. The replay never splits an episode into single transitions, and a window never crosses an episode boundary.

Coverage limit: in this data, XP, armor, absorption and effects never change. The inventory changes in 71 steps of the first 24 training episodes, health in 129, food in 23. The heads for these fields exist and train, but the data cannot show whether they predict change.

## Vocabulary compression

The registries of Minecraft 26.3 are large: 1,286 block types and 1,658 item types. The training data uses few of them. `CompactVocabulary` (`src/minecraft_rl/minecraft_replay.py`) does this:

1. For each family (block, fluid, entity, item, effect), collect the raw ids that occur in the `train` split.
2. Sort them. Compact index `i` is the `i`-th seen raw id.
3. Add one index at the end for unknown. Every raw id that `train` never showed maps to it, in every split.
4. Item raw id 0 and effect raw id 0 mean an empty slot. They are always kept, so compact index 0 still means empty.

The map is a property of the data, not Minecraft knowledge. It comes from `train` only and is the same for every split. Run metadata records its id lists and a hash (`CompactVocabulary.identifier`).

Measured compact sizes, unknown included: block 61, fluid 3, entity 9, item 9, effect 2. Unknown fraction of the targets:

| Split | Block | Fluid | Entity | Item | Effect |
|---|---|---|---|---|---|
| `eval_episode` | 0.000 | 0.000 | 0.002 | 0.003 | 0.000 |
| `eval_seed` | 0.159 | 0.000 | 0.000 | 0.000 | 0.000 |

So 16% of the block rays in the held-out worlds show a block that the training data never showed. A prediction of the unknown class does not identify that block. Evaluation leaves unknown targets out of every ray accuracy and reports them on their own: their fraction, and how often the model predicts the unknown class on them.

## Model

`src/minecraft_rl/minecraft_world_model.py`. The recurrent core is the Stage 2G RSSM with larger sizes: deterministic state h with 256 units (GRU cell), stochastic state z with 16 categorical variables of 16 classes, prior p(z | h), posterior q(z | h, e), KL loss with free nats 1.0. Imagination advances h and samples z from the prior. It never decodes and re-encodes an observation.

Encoder input:

- Rays: each ray of the 25 x 33 grid gets one learned embedding of its ray class (8 values wide) and its distance divided by 32. A ray class joins kind and type into one categorical value: class 0 is no hit, then one class per compact block, fluid and entity type.
- Self state: the 9 scalars (health, max health, absorption, food, air, armor, XP level, XP progress, pitch), normalized as `docs/decisions/observation.md` suggests, a learned embedding of the selected slot, and per slot of the 36 inventory slots, 4 armor slots and the offhand: a learned item embedding, the normalized count and the durability bar. Per effect slot: a learned effect embedding, the amplifier, the normalized timer and a flag for an infinite effect.
- Action: 9 buttons, 2 camera deltas divided by 45 degrees, and a one-hot hotbar choice with one value for "keep".

Decoder output: every field above. The loss of one observation is its negative log likelihood, summed over its elements:

- Cross-entropy for categorical fields: the ray class, selected slot, inventory, armor and offhand items, and effect types.
- Squared error (a Gaussian with unit variance) for the normalized continuous fields.
- Binary cross-entropy for the infinite-effect flag.

A mask removes fields without meaning: the distance of a ray without a hit, and the count, durability, amplifier and timer of an empty slot. Tests make sure that predictions of masked elements do not change the loss. A continuation head predicts that a transition did not end the episode.

## Model size selection

`src/minecraft_rl/minecraft_model_benchmark.py` trains each candidate on the same fixed replay batch (batch 16, sequence length 32, so 33 states per window). It does 2 warm-up updates and 7 timed updates. Canonical runtime: one thread, CPU, float32. The commit was `464ca8f` with the uncommitted changes of this milestone. The command ran in the foreground with default QoS.

The ray-grid layer is the only part that changes:

- `conv`: three strided 3 x 3 convolutions in the encoder, three transposed 4 x 4 convolutions in the decoder.
- `patch`: the grid is cut into non-overlapping 4 x 4 patches. One linear layer, shared by all patches, encodes each patch. In the decoder, a linear layer gives one vector per patch, and one shared linear layer expands it to the 16 rays of the patch. This is a convolution whose kernel size equals its stride.

| Candidate | Ray layer | Encoder / decoder channels | Parameters | Update (s) | Encoder (s) | RSSM core (s) | Decoder (s) |
|---|---|---|---|---|---|---|---|
| A | conv | 32 / 32 | 2,477,981 | 0.604 | 0.139 | 0.013 | 0.446 |
| B | conv | 16 / 16 | 1,855,693 | 0.376 | 0.083 | 0.013 | 0.274 |
| C | conv | 16 / 8 | 1,669,397 | 0.292 | 0.083 | 0.013 | 0.190 |
| D | patch | 16 / 16 | 2,101,421 | 0.177 | 0.024 | 0.013 | 0.134 |
| E | patch | 32 / 32 | 2,892,637 | 0.218 | 0.026 | 0.013 | 0.171 |

The part times are forward and backward passes of one part alone, so they do not add up to the update time. Peak resident memory of the benchmark process was 3.1 GB. Most of that is the replay of all three splits in memory.

Selected: D, `patch` with 16 / 16 channels, 2,101,421 parameters. Reason: it is the fastest, and it passes the tiny-overfit test below. Selection used only computational cost and the tiny-overfit test, never a held-out metric. The configuration is now frozen for this smoke test (`ModelConfig` defaults).

A first draft had 20.0 million parameters because the item output layer used all 1,658 raw item types. The compact vocabulary removed most of them.

## Compute on a Mac: energy, not pauses

The CPU temperature depends on the electrical energy per second. The duty cycle of `minecraft_rl.sweep` pauses a process, but the same energy is used later, so a pause only spreads the heat over a longer time. Two measures make the energy smaller. `src/minecraft_rl/macos_energy.py` reads the CPU energy counter of a process that macOS keeps (`proc_pid_rusage`, `ri_energy_nj`), so the training run records its energy.

Measured with candidate D, batch 16, one thread:

| Configuration | s / update | J / update | Mean CPU power | Share on performance cores |
|---|---|---|---|---|
| Default QoS | 0.188 | 1.21 | 6.45 W | 1.00 |
| Background QoS (`taskpolicy -b`) | 0.654 | 0.28 | 0.43 W | 0.00 |
| `conv` 32 / 32, default QoS | 0.628 | 4.12 | 6.55 W | 1.00 |

Background QoS moves the process to the efficiency cores. An update takes 3.5 times longer but uses 4.3 times less energy, and the power drops by a factor of 15. The patch layer needs 3.4 times less energy per update than the `conv` 32 / 32 layer. Energy per state does not depend on the batch size (0.53-0.55 mJ per state for batch 8, 16 and 32).

Decision: Stage 3 training runs through `minecraft_rl.sweep` with background QoS and `--duty-cycle 1.0`. The thermostat stays on. A duty cycle below 1.0 only adds wall-clock time. In the tiny-overfit run below, the die temperature stayed at or below 61.0 C without a duty-cycle pause.

## Tests

`tests/test_minecraft_replay.py` and `tests/test_minecraft_world_model.py` use small synthetic episodes in the dataset format. They check these properties:

1. A sampled window never crosses an episode boundary, and its time steps follow each other.
2. Window, action and continuation shapes are correct, and continuation is 0 only on the terminal transition.
3. Vocabulary compression is deterministic and does not depend on the episode order.
4. Known ids map back to the raw id, empty slots stay at index 0, and unseen ids map to unknown.
5. A held-out split uses the training vocabulary.
6. Encoder and decoder shapes are correct for both ray layers.
7. Masks remove fields without meaning, and changing masked predictions does not change the loss.
8. The ray-class loss targets the joint class.
9. Prior and posterior outputs, losses and KL are finite, and every parameter group receives a gradient.
10. The prior depends on the action.
11. Imagination calls neither the encoder nor the decoder.
12. A checkpoint saves and loads the same model.
13. Training that stops and continues from saved model, optimizer and generator states gives the same parameters as training without a stop.
14. The evaluation reports persistence, frequency and action-shuffle results and all horizons.

`tests/test_macos_energy.py` checks the energy counter. `./scripts/check`: 201 tests pass.

## Tiny-overfit test

Question: can the full pipeline (encoder, RSSM, decoder, losses, optimizer) fit a small fixed data set? Data: 8 fixed training windows of 32 transitions. Batch 8, Adam, learning rate 1e-3, gradient-norm clip 1000, seed 0. Run: `runs/stage3/tiny-overfit-v2`. The metrics are one-step prior predictions on the same 8 windows: the model reads 12 real steps as context, then predicts each next observation from the prior with the real action.

Two earlier attempts set the training defaults. With learning rate 3e-4 and gradient-norm clip 100, 600 updates reached a ray accuracy of 0.851. The gradient norm was 400-1,500, so the clip of 100 made every step smaller. The self-state heads got too little signal: the predicted health fell from 20 to 10 inside a window while the real health stayed at 20. The defaults changed to learning rate 1e-3 and clip 1000 before any held-out run.

| Metric (8 training windows) | Update 0 | Update 2000 | Persistence | Frequency |
|---|---|---|---|---|
| Ray accuracy, known targets | 0.027 | 0.949 | 0.914 | 0.392 |
| Ray accuracy where the ray class changed | 0.020 | 0.602 | 0.000 | 0.228 |
| Ray NLL where the ray class changed (nats) | 4.27 | 1.03 | | |
| Ray distance error, rays with a hit (blocks) | 4.96 | 2.08 | 1.47 | |
| Pitch error (degrees) | 7.93 | 1.24 | 1.27 | |
| Health error | 23.6 | 0.32 | 0.00 | |
| Food error | 20.8 | 0.13 | 0.00 | |
| Selected slot accuracy where it changed | 0.0 | 1.0 | 0.0 | 0.0 |

The training loss fell from 3,574 to 103. The ray-class term fell from 3,451 to 98, and inventory items from 79.1 to 0.014. With the wrong action of another window, the ray accuracy fell from 0.949 to 0.901, the accuracy on changed rays from 0.602 to 0.437, and the pitch error rose from 1.24 to 2.68 degrees. The model uses the action.

Result: passed. The model reaches 0.60 accuracy on rays that change, which persistence cannot predict. The ray distance error is still higher than persistence: the squared error is small next to the summed ray-class loss. This is a known weak point, not an overfit failure.

Latent diagnostics at update 2000: mean KL 0.08 nats, 99.4% of the states below the free nats, 5 of 16 variables active, prior-posterior agreement 0.999. On 8 windows the model barely needs z. The held-out run must show whether z carries more information on the full data.

Cost: 2,000 updates in 666 s over two processes, 0.33 s per update on the efficiency cores, 351 J of CPU energy (0.18 J per update). One evaluation of the 8 windows takes 0.4 s.

## Commands

```
.venv/bin/python -m minecraft_rl.minecraft_model_benchmark --output runs/stage3/benchmark.json
.venv/bin/python -m minecraft_rl.sweep minecraft_world_model_train --seeds 0 \
    --output-root runs/stage3/tiny-overfit-v2 --duty-cycle 1.0 --time-budget 540 -- \
    --tiny-windows 8 --batch 8 --updates 2000 --eval-at 0,500,1000,2000
```

If the time budget ends first, run the same command again. It continues from the saved state with the same results.
