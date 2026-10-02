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

With the Stage 2E coefficient `eta = 0.01`, the policy often collapsed early onto one turn direction or onto never turning, and then stopped producing the data that would correct it: 4 of 10 seeds reached 100% after 12 iterations (final success 0%, 50% or 100%). Raising it to `eta = 0.03` keeps the policy stochastic long enough to try both turns in both cue states: 8 of 10 seeds reach 100%. This is a measured, task-independent exploration setting, not task knowledge; it is the Stage 2F default (`--entropy`).

## Checks

Unit tests `tests/test_dreamer_loop.py` (about 3 s): policy-collected episodes are consistent transitions, and a short `L = 1` loop (4 iterations) grows the replay as expected, measures model error on every new batch, reaches 100% greedy and more than 80% sampled real success with imagined and real returns within 0.2, and reloads both checkpoints to the same real performance.

## Run

```sh
.venv/bin/python -m minecraft_rl.dreamer_loop --seed 0 --device cpu --output runs/stage2f/seed0/metrics.json
```

## Results (macOS arm64, Python 3.12.11, PyTorch 2.14.0, CPU, about 70 s per seed with five seeds in parallel)

Seed 0, `eta = 0.03`:

| Iteration | Real transitions | Greedy success (cue left / right) | Sampled success | Imagined minus real return | New-data turn-reward sign accuracy |
| --- | --- | --- | --- | --- | --- |
| 0 | 1,335 | 50% (0% / 100%) | 50% | +0.66 | 56% |
| 1 | 1,611 | 50% (0% / 100%) | 50% | +0.68 | 66% |
| 3 | 2,435 | 50% (0% / 100%) | 50% | +0.40 | 65% |
| 4 | 2,720 | 50% (100% / 0%) | 50% | +0.77 | 31% |
| 5 | 3,215 | 50% (0% / 100%) | 67% | +0.20 | 100% |
| 6 | 3,494 | 100% | 100% | -0.01 | 100% |
| 11 | 4,820 | 100% | 100% | +0.01 | 100% |

The world model starts with too little data to predict the turn reward on the policy's own episodes (56% sign accuracy). The policy is optimistic about one cue (imagined minus real return +0.66), goes there, collects real outcomes, and the model's accuracy on new policy data rises until both cues are solved at iteration 6. After that the policy turns correctly every time, all collected episodes succeed, and imagined and real returns agree within 0.06.

Seeds 0-9 after 12 iterations, `eta = 0.03`:

- 8 of 10 seeds reach 100% greedy and sampled real success with no wrong turns and mean length 6.0-6.5 (shortest possible: 6), first at iteration 2-7, using 4,820-6,937 real transitions in total. Imagined minus real return at the end: -0.013 to +0.019 for every seed.
- At iteration 0 the imagined-minus-real gap is +0.52 to +2.90 for every seed: the Stage 2E exploitation appears in every run at first. Policy data removes it in all seeds, including the two that do not solve the task.
- Seeds 6 and 7 end at 50%: correct for one cue; for the other, the greedy policy stops advancing in the corridor and turns into the wall until truncation, so it never reaches the junction (sampled success also 50%). Their models are accurate (gap +0.02 and -0.004), so this is not model exploitation. For seed 6, at the junction after that cue, the policy would also turn the wrong way (`LEFT` with probability 0.999) while the model correctly predicts `-1.01` for it: the actor never sees that state in imagination or reality because it does not get there. 24 iterations instead of 12 leave both seeds unchanged. This is an exploration local optimum of the actor.
- With `eta = 0.01`: 4 of 10 seeds reach 100%, one ends at 50% with half the episodes taking the wrong turn and its imagined return 0.79 above the real return, three at 50% and two at 0%.

**Interpretation.** The integrated loop works on the toy task: an agent starting from random initialization, without demonstrations or task knowledge, learns a world model from its own experience, learns behavior entirely in that model's imagination, and solves the partially observable cue-memory task in the real environment for 8 of 10 seeds. Real data collection with the current policy is what corrects model exploitation: the large early imagined-versus-real gap closes in every seed. The remaining failures are exploration failures (a confident accurate model of a policy that stopped trying), which better data coverage of the policy's own states does not fix; entropy pressure helps, as measured, but does not guarantee coverage.

## Limits

The world model is still deterministic: no stochastic latent, no KL term, no uncertainty-aware imagination, and the actor learns by REINFORCE through argmax observation feedback rather than through dynamics gradients. These are the explicit RSSM components the roadmap requires before a Minecraft world model. Exploration is entropy only; 2 of 10 seeds stay at 50%. The task has one bit of memory and deterministic transitions, the evaluation uses the same maze length as training, and the run is short. Minecraft training additionally needs the environment track's step 8 at the scale trained (`docs/roadmap.md`).
