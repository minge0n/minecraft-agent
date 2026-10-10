# Stage 3: Minecraft RSSM world model smoke test

Purpose: learn short-horizon dynamics from real structured Minecraft transitions with the Stage 2G RSSM (`docs/stage2g.md`), and compare the model against trivial baselines. No actor is trained. No reward exists: the reward design is a separate later milestone.

Status: first held-out run and the follow-up on loss balance complete. The follow-up selected and froze the objective G (section "Follow-up: balance of the loss components"). The model is still not ready for actor training: it uses the action only weakly, and pitch prediction stays far behind persistence.

First held-out run: complete, with a mixed result. The model predicts changed rays better than every baseline. It predicts unchanged content and pitch worse than persistence, and it uses the action only weakly. The likely cause is the scale of the loss terms (section "Diagnosis"). The model is not yet good enough for actor training.

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

## Held-out run

Run: `runs/stage3/train-3000`, seed 0, commit `07ccf4b`, the frozen model D. Batch 16, sequence length 32, Adam 1e-3, clip 1000. Staged budget: evaluation at 0, 100, 300, 1,000, 2,000 and 3,000 updates. Each evaluation uses 128 fixed windows per split, with 12 context steps and 21 scored transitions per window (2,688 transitions). Imagination ran at 1,000 and 3,000 updates on windows of 12 context and 20 imagined steps.

The run stopped after 3,000 updates because the held-out gain became small. Accuracy on changed rays in `eval_episode` was 0.357 at 1,000, 0.351 at 2,000 and 0.360 at 3,000 updates.

The same command with `--updates 1000` and the run with `--updates 3000` gave identical metrics at 0, 100, 300 and 1,000 updates, over different process boundaries. This confirms the resume determinism on the real workload.

### One-step prediction at 3,000 updates

The model reads the context with the posterior, then predicts each next observation from the prior with the real action. "Changed" metrics use only the elements whose target differs from the last real observation. Persistence is 0 on them by definition.

| Metric | Split | Model | Shuffled actions | Persistence | Frequency |
|---|---|---|---|---|---|
| Ray accuracy, known targets | `train` | 0.779 | 0.776 | 0.855 | 0.342 |
| | `eval_episode` | 0.761 | 0.760 | 0.855 | 0.306 |
| | `eval_seed` | 0.639 | 0.638 | 0.839 | 0.298 |
| Ray accuracy where the class changed | `train` | 0.375 | 0.365 | 0.000 | 0.269 |
| | `eval_episode` | 0.360 | 0.354 | 0.000 | 0.265 |
| | `eval_seed` | 0.279 | 0.275 | 0.000 | 0.229 |
| Ray NLL where the class changed (nats) | `eval_episode` | 2.26 | 2.30 | | |
| | `eval_seed` | 3.42 | 3.45 | | |
| Ray distance error, hits (blocks) | `eval_episode` | 3.03 | 3.04 | 1.22 | |
| | `eval_seed` | 3.63 | 3.62 | 1.22 | |
| Pitch error (degrees) | `eval_episode` | 7.25 | 7.36 | 1.46 | |
| | `eval_seed` | 12.15 | 12.08 | 1.45 | |
| Health error where health changed | `eval_episode` | 1.51 | 1.56 | 0.54 | |
| | `eval_seed` | 2.75 | 2.76 | 0.83 | |

Unknown targets in `eval_seed`: 10.6% of the scored rays (the 16% in the data section counts block rays only). The model predicted the unknown class on 0% of them. The training data never contains the unknown class, so the model cannot learn to predict it.

Continuation: no evaluation window contains a death, so the continuation head is untested on terminal transitions. On non-terminal transitions it predicts continue with p = 0.9999.

### Action conditioning

The action shuffle replaces the actions of each window with those of another window. At one step, the effect is small. In `eval_episode`, accuracy on changed rays falls from 0.360 to 0.354 and the NLL rises from 2.26 to 2.30. The prior on the full trained model changes little with the action: the mean KL between the priors after the real and a shuffled action was 0.014 nats at 1,000 updates.

The effect grows with the horizon (next section). At horizon 10 in `eval_episode`, accuracy on changed rays is 0.231 with the real actions and 0.195 with shuffled actions, and the pitch error is 10.9 against 12.4 degrees. So the model uses the action, but weakly. Negative result: pitch follows from the previous pitch and the camera action, and persistence alone has an error of 1.5 degrees, yet the model error is 7-12 degrees.

### Event subsets (`eval_episode`, 3,000 updates)

Events with at least 20 transitions in the 2,688 scored transitions: camera turn (2,580), forward (1,255), attack (1,232), jump (538), sneak (300), use (273), many rays change (1,672), entity rays change (240). Too few examples: hotbar selection (12), health change (13), inventory change (4), food change (0), termination (0), no movement and no camera (0). The exploration policy almost always turns the camera, so no "still" subset exists.

At 1,000 updates, accuracy on changed rays was 0.35-0.37 in every event subset of `eval_episode`. The real and the shuffled action differed by at most 0.004 in each subset. In `eval_seed`, the subset "use" was weakest (0.178). Per-event metrics at all update counts are in `metrics.json`.

### Imagination

The model filters 12 real steps, then rolls the prior forward with the 20 real future actions, without decoding and re-encoding. Values at 3,000 updates, `eval_episode`:

| Horizon | Ray accuracy (model / shuffled / persistence) | Changed rays (model / shuffled) | Pitch error, degrees (model / shuffled / persistence) | Ray distance error, blocks (model / persistence) |
|---|---|---|---|---|
| 1 | 0.779 / 0.778 / 0.859 | 0.356 / 0.351 | 7.8 / 7.9 / 1.6 | 3.09 / 1.19 |
| 5 | 0.674 / 0.662 / 0.685 | 0.289 / 0.273 | 9.0 / 9.8 / 6.4 | 3.23 / 3.00 |
| 10 | 0.573 / 0.540 / 0.581 | 0.231 / 0.195 | 10.9 / 12.4 / 10.7 | 3.62 / 4.27 |
| 20 | 0.485 / 0.458 / 0.488 | 0.206 / 0.177 | 13.5 / 15.6 / 14.7 | 4.01 / 5.38 |

`eval_seed`, same columns:

| Horizon | Ray accuracy | Changed rays | Pitch error | Ray distance error |
|---|---|---|---|---|
| 1 | 0.651 / 0.651 / 0.843 | 0.302 / 0.296 | 11.4 / 11.5 / 1.6 | 3.60 / 1.17 |
| 5 | 0.566 / 0.558 / 0.644 | 0.256 / 0.234 | 12.3 / 12.9 / 6.4 | 3.80 / 3.31 |
| 10 | 0.479 / 0.475 / 0.539 | 0.212 / 0.201 | 13.2 / 12.9 / 10.7 | 4.10 / 4.90 |
| 20 | 0.381 / 0.376 / 0.439 | 0.148 / 0.142 | 15.1 / 14.7 / 14.5 | 4.62 / 6.23 |

Error grows with the horizon for the model and for persistence. Persistence error grows faster. In `eval_episode` the model matches persistence on ray accuracy from horizon 5 on, beats it on ray distance from horizon 10 on, and keeps 0.21 accuracy on changed rays at horizon 20. In `eval_seed` the model stays below persistence on ray accuracy at every horizon.

### Latent diagnostics

| Split | Raw KL (nats) | Below free nats | Active variables | Prior entropy | Posterior entropy | Prior-posterior agreement |
|---|---|---|---|---|---|---|
| `train` | 6.55 | 2.9% | 16 / 16 | 18.9 | 12.6 | 0.572 |
| `eval_episode` | 6.93 | 6.2% | 16 / 16 | 17.9 | 11.5 | 0.574 |
| `eval_seed` | 8.04 | 9.0% | 16 / 16 | 18.2 | 11.7 | 0.542 |

The maximum entropy of z is 44.4 nats. 86-94 of the 256 latent classes occur. The posterior carries 6-8 nats more than the prior, so the free nats of 1.0 almost never apply. On the full data, z carries information, unlike in the tiny-overfit test. The higher KL in `eval_seed` agrees with its unseen worlds. Free nats stay at 1.0. The latent shows no collapse that would need another setting.

### Diagnosis

The gradient on the encoder and RSSM core comes almost only from the ray-class term. At 1,000 updates, on one batch, the gradient norm from each loss term was:

| Term | Loss | Gradient norm on encoder and core |
|---|---|---|
| Ray class | 822 | 223 |
| Ray distance | 15.5 | 12.5 |
| Inventory items | 0.62 | 1.10 |
| Selected slot | 1.89 | 0.95 |
| Scalars (with pitch) | 0.046 | 0.066 |

The ray-class loss sums 825 categorical terms, while pitch is one squared error of a value divided by 90. The pitch error of 7 degrees costs (7 / 90)^2 = 0.006 in the loss. The objective is the correct likelihood under unit-variance Gaussians, but these variances do not fit the data: a change of 1 degree is important and a unit variance makes it free. The same holds for the ray distance. Hypothesis, not yet tested: a smaller fixed variance (a larger weight) for the continuous fields, or a mean instead of a sum over the rays, fixes pitch and copy behavior and makes the model use the action more. That changes the training objective, not the frozen architecture. It is the next experiment, not part of this run.

### Compute

| Item | Value |
|---|---|
| Update time, background QoS, efficiency cores | 0.66 s (800 sequence steps per second) |
| Update time, default QoS (benchmark) | 0.177 s |
| CPU energy, 3,000 updates with evaluations | 1,091 J (0.36 J per update) |
| Wall-clock time | 2,131 s over 4 processes |
| Die temperature | mean 53.7 C, max 59.2 C, one thermostat pause of 14 s |
| One evaluation, 128 windows, one split | 9 s (imagination included) |
| Peak memory of the benchmark process | 3.1 GB |

Collection throughput is separate: the collector reached 93 environment steps per second with rendering and video (`docs/minecraft-spike.md`).

### Not done

- Render-distance benchmark for training workers (8, 6, 4). It needs Minecraft runs and does not affect offline training. It stays open.
- Events that the data barely contains: inventory, health, food, hotbar selection and death. Their heads train, but no metric for them has enough examples.

## Follow-up: balance of the loss components

Question: does the summed ray-class loss crowd out the other fields, and does a loss that weighs each field by its meaning instead of by its element count improve continuous prediction and action use? The model, data, splits, seed 0, batch, sequence length, optimizer, learning rate, clip and free nats stayed as in the held-out run. Only the objective changed.

### Original imbalance

Under the original objective ("sum"), each field adds the sum of its element losses per observation. The ray class adds 825 cross-entropy terms, the pitch one squared error in units of 90 degrees. `src/minecraft_rl/minecraft_gradients.py` measures the gradient of each loss group alone, with its training weight, on one fixed batch, at three shared places: the ray encoder, the RSSM core (GRU cell, action layer, prior, posterior) and the decoder input s = [h, z]. Under "sum" at 1,000 updates:

| Component | Loss | Ray encoder | RSSM core | s = [h, z] |
|---|---|---|---|---|
| Ray class | 787 | 118.4 | 118.9 | 6.64 |
| Ray distance | 13.3 | 7.1 | 7.7 | 0.27 |
| Self state | 2.19 | 0.67 | 0.71 | 0.023 |
| Pitch (part of self state) | 0.017 | 0.048 | 0.049 | 0.002 |
| Inventory | 0.89 | 1.34 | 1.13 | 0.040 |
| Continuation | 0.0004 | 0.001 | 0.001 | 0.0001 |
| KL | 11.9 | 5.7 | 7.2 | 0 |

The ray class moves the RSSM core 170 times more than the self state and 2,400 times more than the pitch.

### Objective variants

`Objective` in `src/minecraft_rl/minecraft_world_model.py` sets a reduction, a fixed weight per loss group (`LOSS_GROUPS`: ray class, ray distance, self state, inventory, plus continuation and KL) and a `component_scale` on every reconstruction component and the continuation loss. The KL keeps its Stage 2G form with free nats 1.0.

| Variant | Reduction | Weights | Description |
|---|---|---|---|
| A | `sum` | none | The original log likelihood. Identical to the held-out run. |
| B | `ray_mean` | none | As A, but the ray class is the mean over the rays. |
| C | `semantic_mean` | none | Each term is the mean over its own valid elements. Each of the 9 scalars is its own component. |
| D | `semantic_mean` | KL 1/825 | As C, with the KL scaled down by the ray count. |
| E | `semantic_mean` | scale 825 | As C, with every component scaled up by the ray count. |
| G | `group_mean` | scale 825 | As E, then each loss group is the mean of its valid components. |

The continuous targets keep their fixed game ranges (`docs/decisions/observation.md`): ray distance / 32, health, max health and absorption / 20, food / 20, air / 10, armor / 20, pitch / 90, XP level as log1p(level) / log1p(30), XP progress as it is, stack count as log1p(count) / log1p(64), durability as the 13-step bar, effect amplifier / 4, effect seconds as log1p(seconds) / log1p(600) with a flag for an infinite effect. No new transform was needed: every field already lies in [0, 1] or [-1, 1].

A component with no valid element in the batch adds nothing. An empty armor slot, for example, has no durability component.

### Staged budget

All variants ran 500 updates first. Only variants without an early failure continued: E and G to 1,000 updates, G to 3,000. A is the held-out run itself. A rerun of A reproduced its metrics at 1,000 updates bit for bit, so the earlier 3,000-update run of A is the baseline at 3,000 updates.

`eval_episode`, one-step prior prediction:

| Variant | Updates | Ray accuracy | Changed rays | Ray distance (blocks) | Pitch (degrees) | Latent KL | Active variables |
|---|---|---|---|---|---|---|---|
| A | 500 | 0.580 | 0.309 | 3.71 | 8.04 | 4.3 | 16 |
| B | 500 | 0.319 | 0.250 | 2.63 | 8.18 | 1.5 | 16 |
| C | 500 | 0.349 | 0.269 | 3.81 | 10.62 | 0.8 | 16 |
| D | 500 | 0.326 | 0.269 | 3.82 | 9.59 | 0.2 | 4 |
| E | 500 | 0.414 | 0.265 | 3.78 | 10.19 | 1.4 | 6 |
| E | 1,000 | 0.421 | 0.268 | 3.66 | 10.97 | 0.8 | 6 |
| G | 1,000 | 0.705 | 0.353 | 3.38 | 7.61 | 4.3 | 16 |
| A | 1,000 | 0.714 | 0.357 | 3.46 | 7.64 | 4.4 | 16 |

B, C and D failed early. They share a cause: once the ray class is a mean, its gradient is 825 times smaller, and the whole loss falls from about 3,600 to about 4-18. Adam divides each gradient by its running root mean square plus eps = 1e-5. With losses this small, the second-moment estimate of 15-90% of the parameters in the prior, the GRU cell, the self-state encoder and the decoder grid was below eps (measured on the optimizer state at 500 updates), so Adam stopped scaling those gradients up and the parts learned slowly. D also scaled the KL down and its latent collapsed (4 of 16 variables active, KL 0.24 nats). The ray accuracy of B, C and D at 500 updates was 0.32-0.35, the same as their posterior reconstruction, so the decoder had not learned the scene.

E kept the ray class at its old scale and raised every other component by 825. It then failed in a different way. The selected-slot cross-entropy (9 slots, entropy 2.15 nats) became the largest self-state term, its training loss fell from 2.2 to 0.4 nats, and the self-state gradient on the RSSM core was 2,425 against 523 for the ray class. The latent fell to 6 active variables, and the one-step changed-ray accuracy stayed at 0.27. Negative result: per-component means give every small field the weight of the whole ray grid, and a field with many classes and almost no change then dominates.

G fixed that by averaging the components inside each group. It reached the same values as A at 1,000 updates (0.353 against 0.357 on changed rays) with 16 active variables.

### Gradient balance under G

| Component | Update 1,000 core | Update 3,000 loss | Ray encoder | RSSM core | s = [h, z] |
|---|---|---|---|---|---|
| Ray class | 145.3 | 555 | 129.3 | 194.3 | 8.97 |
| Ray distance | 7.2 | 11.6 | 9.6 | 8.1 | 0.32 |
| Self state | 52.8 | 4.9 | 12.8 | 26.9 | 2.18 |
| Pitch (part of self state) | 2.4 | 0.82 | 1.7 | 2.0 | 0.11 |
| Inventory | 17.9 | 3.7 | 9.3 | 12.1 | 0.44 |
| Continuation | 0.22 | 0.025 | 0.10 | 0.14 | 0.005 |
| KL | 6.8 | 16.2 | 4.2 | 6.6 | 0 |

The ratio of ray class to self state on the RSSM core falls from 170 under A to 7 at 3,000 updates under G, and from 2,400 to 99 for the pitch. G removes the dominance that came only from tensor size.

### Result at 3,000 updates: A against G

| Metric | Split | A | G | Persistence |
|---|---|---|---|---|
| Ray accuracy | `eval_episode` | 0.761 | 0.761 | 0.855 |
| | `eval_seed` | 0.639 | 0.625 | 0.839 |
| Changed rays | `eval_episode` | 0.360 | 0.360 | 0.000 |
| | `eval_seed` | 0.279 | 0.271 | 0.000 |
| Changed-ray NLL (nats) | `eval_episode` | 2.263 | 2.246 | |
| | `eval_seed` | 3.419 | 3.447 | |
| Ray distance (blocks) | `eval_episode` | 3.03 | 3.00 | 1.22 |
| | `eval_seed` | 3.63 | 3.40 | 1.22 |
| Pitch (degrees) | `eval_episode` | 7.25 | 7.91 | 1.46 |
| | `eval_seed` | 12.15 | 10.08 | 1.45 |
| Health | `eval_episode` | 1.08 | 0.80 | 0.003 |
| | `eval_seed` | 2.02 | 1.70 | 0.004 |
| Food | `eval_episode` | 1.38 | 1.10 | 0.000 |
| | `eval_seed` | 1.66 | 1.37 | 0.003 |

Action shuffle at one step, `eval_episode`, changed rays: A 0.360 against 0.354, G 0.360 against 0.352. Changed-ray NLL: A 2.263 against 2.299, G 2.246 against 2.289. The pitch error does not depend on the action under either objective: with shuffled actions it is 7.36 (A) and 7.87 (G) degrees.

Imagination at 3,000 updates: changed-ray accuracy and pitch error (degrees), each with real / shuffled actions.

`eval_episode`:

| Horizon | A changed rays | G changed rays | A pitch | G pitch | Persistence pitch |
|---|---|---|---|---|---|
| 1 | 0.356 / 0.351 | 0.343 / 0.338 | 7.8 / 7.9 | 9.0 / 9.0 | 1.6 |
| 5 | 0.289 / 0.273 | 0.303 / 0.276 | 9.0 / 9.8 | 9.3 / 9.9 | 6.4 |
| 10 | 0.231 / 0.195 | 0.248 / 0.213 | 10.9 / 12.4 | 10.0 / 11.2 | 10.7 |
| 20 | 0.206 / 0.177 | 0.217 / 0.180 | 13.5 / 15.6 | 12.5 / 14.6 | 14.7 |

`eval_seed`:

| Horizon | A changed rays | G changed rays | A pitch | G pitch | Persistence pitch |
|---|---|---|---|---|---|
| 1 | 0.302 / 0.296 | 0.274 / 0.272 | 11.4 / 11.5 | 11.3 / 11.4 | 1.6 |
| 5 | 0.256 / 0.234 | 0.210 / 0.203 | 12.3 / 12.9 | 12.0 / 12.8 | 6.4 |
| 10 | 0.212 / 0.201 | 0.176 / 0.163 | 13.2 / 12.9 | 12.2 / 13.1 | 10.7 |
| 20 | 0.148 / 0.142 | 0.142 / 0.111 | 15.1 / 14.7 | 12.7 / 14.6 | 14.5 |

In `eval_episode`, G is better than A on changed rays from horizon 5 on, and on pitch from horizon 10 on. Under both objectives, real actions beat shuffled actions at every horizon from 5 on. The gap on changed rays is about the same (0.027-0.037 for G, 0.016-0.036 for A). In `eval_seed`, G is worse than A on changed rays at every horizon, but better on pitch from horizon 5 on, and its action gap at horizon 20 is larger (0.031 against 0.006). Ray accuracy over all rays is 0.01-0.04 lower under G at every horizon.

Latent diagnostics at 3,000 updates, `eval_episode` (`eval_seed`): A raw KL 6.93 (8.04) nats, below free nats 6.2% (9.0%), 16 active variables, prior entropy 17.9, posterior entropy 11.5, agreement 0.574, 86 classes used. G raw KL 6.72 (7.54), below free nats 0.4% (2.7%), 16 active variables, prior entropy 18.3, posterior entropy 12.5, agreement 0.556, 76 classes used. The latent stays healthy under G.

Cost: G used 0.36 J per update, the same as A, and 0.78 s per update on the efficiency cores.

### Why the pitch did not improve

The hypothesis was that the pitch error came from its small loss weight. G raised that weight by about 50 times, and the pitch error stayed at 7-8 degrees in `eval_episode`. The weight was not the main cause.

A linear probe shows where the pitch is lost. It fits pitch from features on 96 training windows and tests on 48 others. The features are those of the self-state encoder, the encoder output e, and h and z:

| Model | Self-state features | Encoder output e | h | z | Mean predictor |
|---|---|---|---|---|---|
| Untrained | 0.15 | 10.0 | 11.5 | 10.8 | 11.0 |
| A, 1,000 updates | 1.11 | 16.6 | 8.7 | 9.2 | 11.0 |
| G, 1,000 updates | 1.76 | 16.5 | 8.9 | 9.5 | 11.0 |

Probe error in degrees. The self-state features hold the pitch almost exactly. After the last encoder layer, which joins 1,008 ray features with 256 self-state features into 256 values with an ELU, a linear probe cannot recover the pitch anymore. The pitch then cannot reach z through the posterior, and the decoder predicts it from h, from earlier steps. Under both objectives, 60-72% of the encoder outputs sat in the flat negative part of the ELU (below -0.95). This is a hypothesis about the frozen encoder, not a proven cause. Testing it changes the architecture, which this experiment keeps fixed.

### Selected objective

Selected: G, `group_mean` with component scale 825 (the ray count). Reasons:

1. It removes the dominance that came only from the element count (ray class to self state on the RSSM core: 7 against 170).
2. It keeps changed-ray prediction (0.360 in `eval_episode`, as A) and the latent diagnostics.
3. It improves ray distance, health and food in both held-out splits at one step, pitch in `eval_seed` at one step, and changed rays and pitch in `eval_episode` imagination from horizon 5 or 10 on.
4. Each weight has one meaning: every loss group counts once, as a whole ray grid, whatever the number of its fields.

Negative findings that the selection does not hide:

- Pitch in `eval_episode` got worse (7.25 to 7.91 degrees), and the pitch error stays 5 times the persistence error.
- Ray accuracy and changed rays in `eval_seed` fell by about 0.01.
- Action use did not get clearly stronger. At one step, the gap between real and shuffled actions is small under both objectives (changed rays 0.008 under G, 0.006 under A). In imagination, the gap is about the same under G and A in `eval_episode`. The pitch error does not depend on the action at one step.
- In `eval_seed`, G predicts changed rays in imagination worse than A at every horizon (0.176 against 0.212 at horizon 10).

The objective is frozen as G for the next Minecraft world-model experiment: `--reduction group_mean --component-scale rays`. The training default stays `sum` so that the runs of this document remain reproducible.

## Commands

```
.venv/bin/python -m minecraft_rl.minecraft_model_benchmark --output runs/stage3/benchmark.json
.venv/bin/python -m minecraft_rl.sweep minecraft_world_model_train --seeds 0 \
    --output-root runs/stage3/tiny-overfit-v2 --duty-cycle 1.0 --time-budget 540 -- \
    --tiny-windows 8 --batch 8 --updates 2000 --eval-at 0,500,1000,2000
```

Held-out run:

```
.venv/bin/python -m minecraft_rl.sweep minecraft_world_model_train --seeds 0 \
    --output-root runs/stage3/train-3000 --duty-cycle 1.0 --time-budget 570 -- \
    --updates 3000 --eval-at 0,100,300,1000,2000,3000 --imagine-at 1000,3000
```

Objective variant G (follow-up):

```
.venv/bin/python -m minecraft_rl.sweep minecraft_world_model_train --seeds 0 \
    --output-root runs/stage3/objective-G --duty-cycle 1.0 --time-budget 540 -- \
    --reduction group_mean --component-scale rays --updates 3000 \
    --eval-at 0,500,1000,3000 --imagine-at 1000,3000 --pause-at 500,1000
```

`--pause-at` stops the run after the named update counts, so that a staged budget can be inspected. The same command continues it.

If the time budget ends first, run the same command again. It continues from the saved state with the same results.
