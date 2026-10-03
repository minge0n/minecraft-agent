# Stage 2F: integrated Dreamer-style toy agent

**Purpose.** Combine the Stage 2C world model, Stage 2D model-error measurement and Stage 2E actor-critic into one loop that learns from its own experience: the policy collects real data, the world model learns from it, and the policy learns in the world model's imagination. Stage 2E showed that a policy trained against a model built from random-policy data can exploit rewards the model invents; this stage tests whether collecting real data with the current policy corrects that.

## Loop

`run` (`src/minecraft_rl/dreamer_loop.py`), T-maze with `L = 4`, 40-step limit:

```text
replay <- 64 uniform-random-policy episodes
repeat 12 iterations:
    world model: 1,000 updates on the whole replay              (Stage 2C losses)
    actor-critic: 200 updates in imagination from every replay step  (Stage 2E losses)
    evaluate in the real maze (greedy and sampled, 256 episodes, held-out seed)
    collect 32 real episodes with the current policy (sampled actions)
    measure the world model's error on them before training on them  (Stage 2D metrics)
    replay <- replay + these episodes
```

The world model, actor, critic and both Adam optimizers persist across iterations; nothing is reset. The policy acts on `s_t = [h_{t-1}, onehot(o_t)]` with `h` built from real observations, exactly as in Stage 2E evaluation. The actor-critic configuration is Stage 2E's except the entropy coefficient (below). Parameters: world model 4,263, actor-critic 5,124. There are no new learned components or losses in this stage; what is new is the data flow, in which the world model's training distribution follows the policy.

**Model error under the policy's own behavior.** Each new batch is first used as held-out data for the current world model: one-step accuracy, junction-turn reward sign accuracy and open-loop imagined-return error at horizons 1, 5, 10, 20, all on behavior the model has not trained on yet. The real-maze evaluation also reports the imagined-minus-real discounted return of the policy from the same start states.

## Entropy coefficient

Canonical measurement (one thread, `docs/decisions/reproducibility.md`), seeds 0-9 after 12 iterations: with the Stage 2E coefficient `eta = 0.01`, 5 of 10 seeds reach 100%, three end at 50% and two at 0%. With `eta = 0.03`, 10 of 10 seeds reach 100%. With the smaller coefficient, the policy commits early to one turn direction or to never reaching the junction. It then stops producing the data that would correct it. `eta = 0.03` keeps the policy stochastic long enough to try both turns in both cue states. This is a measured, task-independent exploration setting, not task knowledge. It is the Stage 2F default (`--entropy`).

Superseded: the first version of this section reported 4 of 10 (`eta = 0.01`) and 8 of 10 (`eta = 0.03`). Those runs used the PyTorch default of 6 intra-op threads on this machine, and their seed-level outcomes depend on the thread count. The direction of the effect is the same in both measurements.

## Checks

Unit tests `tests/test_dreamer_loop.py` (about 3 s): policy-collected episodes are consistent transitions, and a short `L = 1` loop (4 iterations) grows the replay as expected, measures model error on every new batch, reaches 100% greedy and more than 80% sampled real success with imagined and real returns within 0.2, and reloads both checkpoints to the same real performance.

## Run

```sh
.venv/bin/python -m minecraft_rl.dreamer_loop --seed 0 --device cpu --output runs/stage2f/seed0/metrics.json
```

## Results (macOS arm64, Python 3.12.11, PyTorch 2.14.0, CPU, canonical runtime, about 60 s per seed)

All numbers in this section come from the canonical runtime: one intra-op and one inter-op thread (`docs/decisions/reproducibility.md`). Each seed ran twice in separate processes, and both runs gave identical metrics for every seed.

Seed 0, `eta = 0.03`:

| Iteration | Real transitions | Greedy success (cue left / right) | Sampled success | Imagined minus real return | New-data turn-reward sign accuracy |
| --- | --- | --- | --- | --- | --- |
| 0 | 1,335 | 50% (0% / 100%) | 50% | +0.66 | 56% |
| 1 | 1,611 | 50% (0% / 100%) | 50% | +0.68 | 66% |
| 3 | 2,435 | 50% (0% / 100%) | 50% | +0.40 | 65% |
| 4 | 2,720 | 50% (100% / 0%) | 50% | +0.77 | 31% |
| 5 | 3,244 | 50% (0% / 100%) | 65% | +0.20 | 100% |
| 6 | 3,528 | 100% | 100% | -0.01 | 100% |
| 11 | 4,848 | 100% | 100% | -0.01 | 100% |

The world model starts with too little data to predict the turn reward on the policy's own episodes (56% sign accuracy). The policy is optimistic about one cue (imagined minus real return +0.66). It goes there and collects real outcomes. The accuracy of the model on new policy data rises until both cues are solved at iteration 6. After that, the policy turns correctly every time, all collected episodes succeed, and imagined and real returns agree within 0.02.

Seeds 0-9 after 12 iterations, `eta = 0.03`:

- 10 of 10 seeds reach 100% greedy and sampled real success with no wrong turns. The mean length is 6.0-8.5 (shortest possible: 6). The first iteration with 100% is 2-9. The runs use 4,848-12,360 real transitions in total. Imagined minus real return at the end: -0.010 to +0.006.
- At iteration 0, the imagined-minus-real gap is +0.52 to +2.90 for every seed. The Stage 2E exploitation appears in every run at first. Policy data removes it in every seed.
- Seed 7 is the slowest: it first reaches 100% at iteration 9 and needs 12,360 real transitions, twice the median. It is also the seed with the largest iteration-0 gap (+2.90).
- With `eta = 0.01`: 5 of 10 seeds reach 100%, three end at 50% and two at 0%, none with a wrong turn. The 50% and 0% seeds stop before the junction for one or both cues. Their final imagined-minus-real gap is at most 0.01, so their models are accurate and the failure is exploration.

Superseded: the first version of this section reported 8 of 10 seeds at 100% for `eta = 0.03`, with seeds 6 and 7 at 50%, and called the two failures an exploration local optimum. Those runs used 6 intra-op threads. In the canonical runtime, seeds 6 and 7 reach 100% (at iterations 3 and 9). The conclusion that these seeds fail by their nature is withdrawn: with 6 threads the same seeds fail, with 1 thread they succeed, so the outcome of single seeds is sensitive to last-bit floating-point differences. The failure pattern itself is real and is still visible with `eta = 0.01`.

**Interpretation.** The integrated loop works on the toy task. An agent that starts from random initialization, without demonstrations or task knowledge, learns a world model from its own experience. It learns behavior only in the imagination of that model, and it solves the partially observable cue-memory task in the real environment for 10 of 10 seeds. Real data collection with the current policy corrects model exploitation: the large early imagined-versus-real gap closes in every seed. The outcome of a single seed is sensitive to small numerical differences, so a seed count is a weak measure for this loop. Reports of this loop give per-seed results under the canonical runtime.

## Limits

The world model is still deterministic: no stochastic latent, no KL term, no uncertainty-aware imagination, and the actor learns by REINFORCE through argmax observation feedback rather than through dynamics gradients. These are the explicit RSSM components the roadmap requires before a Minecraft world model. Exploration is entropy only. With `eta = 0.01`, half of the seeds stop exploring one branch, and single-seed outcomes depend on small numerical differences. The task has one bit of memory and deterministic transitions, the evaluation uses the same maze length as training, and the run is short. Minecraft training additionally needs the environment track's step 8 at the scale trained (`docs/roadmap.md`).
