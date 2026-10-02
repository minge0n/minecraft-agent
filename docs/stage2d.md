# Stage 2D: imagination and compounding error

**Purpose.** Roll the Stage 2C world model forward without the environment and measure how imagined trajectories drift from real ones as the horizon grows. A low one-step (teacher-forced) loss does not show that the model is usable for planning or for training a policy in imagination; open-loop error does. This stage also measures what happens when the actions come from a policy unlike the one that produced the training data, which is what a learning policy will do.

## Open-loop rollout

`imagine` (`src/minecraft_rl/imagination.py`), from a real start step `s` of an episode:

```text
h_s       = GRU over the real (o_0, a_0), ..., (o_s, a_s)          (B, 32)
o_hat_t+1 = argmax of the next-observation logits of h_t           (B,)
h_t+1     = GRU(h_t, [onehot(o_hat_t+1), onehot(a_t+1)])           t = s .. s + k - 2
outputs   = next-observation logits, reward and continuation logit of h_s .. h_{s+k-1}
```

After `s` the model never sees a real observation: its input is its own most likely prediction. Actions are the real episode's next actions, so imagined and real trajectories are directly comparable step by step. The step-`s` prediction equals the teacher-forced prediction (tested). No parameters are learned in this stage; it evaluates the trained Stage 2C model. Feeding back the argmax is a deterministic choice for a model without a stochastic latent; Stage 2F revisits this with an RSSM.

## Metrics

For every start `s` and horizon `k` in 1, 5, 10, 20 where the real episode still runs at transition `s + k - 1` (pairs pooled over starts and episodes):

- `observation_accuracy`: imagined `o_{s+k}` equals the real one. `trajectory_accuracy`: all of `o_{s+1} .. o_{s+k}` do.
- `return_absolute_error`: `|sum_{j=1..k} (r_hat_{s+j} - r_{s+j})|` over the real episode's remaining steps, the error a policy would see in an imagined k-step return.
- `continuation_accuracy` of the flag at step `s + k`.
- `junction_turn_sign_accuracy` and absolute error on steps that are real junction turns (the only nonzero rewards), and `non_turn_reward_absolute_error` on all other steps (real reward 0).

## Experiment

Seeds `s`: training and evaluation data, initialization and minibatches as in `docs/stage2c.md`. Three models, the Stage 2C configuration and 5,000 steps each:

- `gru_uniform_data`: the Stage 2C GRU, trained on 512 uniform-random-policy episodes.
- `no_memory_uniform_data`: the Stage 2C no-memory control, same data.
- `gru_mixed_data`: the same GRU trained on 512 episodes, 256 of the uniform set and 256 from a **shifted** behavior policy that picks `FORWARD` with probability 0.8 and each turn with 0.1 (data seed `s + 3,000,000`).

Two held-out evaluation sets of 256 episodes: uniform policy (seed `s + 1,000,000`) and shifted policy (seed `s + 2,000,000`). The shifted policy reaches the junction sooner and then waits there: seed 0 has a mean of 5.1 and a maximum of 26 steps at the junction per episode, against 1.5 and 7 for the uniform evaluation set and a maximum of 6 in the uniform training set.

Unit tests: `tests/test_imagination.py` (about 1.5 s): first-step equality with teacher forcing, that real observations after the start are never read, that the real actions are followed, horizon clipping, pair counting, and a short `L = 1` training where the GRU's open-loop rollouts are exact and the control's turn-reward sign stays near chance.

## Run

```sh
.venv/bin/python -m minecraft_rl.imagination --seed 0 --device cpu --output runs/stage2d/seed0/metrics.json
```

## Results (macOS arm64, Python 3.12.11, PyTorch 2.14.0, CPU, about 28 s per seed alone)

Ranges over seeds 0-9 after 5,000 steps.

| Evaluation | Model | k | Observation acc. | Return abs. error | Non-turn reward abs. error | Turn sign acc. |
| --- | --- | --- | --- | --- | --- | --- |
| uniform | GRU, uniform data | 1 | 1.00 | 0.002-0.008 | 0.002-0.007 | 1.00 |
| uniform | GRU, uniform data | 20 | 1.00 | 0.014-0.073 | 0.002-0.010 | 1.00 |
| uniform | no memory | 1 | 0.94 | 0.061-0.079 | 0.001-0.019 | 0.45-0.53 |
| uniform | no memory | 20 | 0.48-0.62 | 0.19-0.45 | 0.000-0.019 | 0.44-0.65 |
| shifted | GRU, uniform data | 1 | 1.00 | 0.007-0.033 | 0.003-0.024 | 0.96-1.00 |
| shifted | GRU, uniform data | 10 | 1.00 | 0.035-0.38 | 0.004-0.085 | 0.92-1.00 |
| shifted | GRU, uniform data | 20 | 1.00 | 0.09-1.59 | 0.005-0.18 | 0.57-1.00 |
| shifted | no memory | 20 | 0.00-0.37 | 0.22-0.51 | 0.002-0.016 | 0.35-0.61 |
| shifted | GRU, mixed data | 1 | 1.00 | 0.003-0.005 | 0.003-0.005 | 1.00 |
| shifted | GRU, mixed data | 20 | 1.00 | 0.019-0.11 | 0.001-0.014 | 1.00 |

Imagined continuation accuracy is 1.00 at every horizon for both GRUs and falls to 0.62-0.92 at k = 20 for the control. Every GRU imagined observation trajectory is exactly right at every horizon on both evaluation sets (trajectory accuracy 1.00). k = 20 pools few pairs on the shifted set because its episodes are short (seed 0: 69 pairs, 13 of them turns), so its spread is the noisiest entry.

Seed 0, k = 20 return error on the shifted set during training (uniform-data GRU): 0.33 at step 200, 0.69 at 1,000, 2.00 at 3,000, 1.22 at 5,000, while on the uniform set it falls from 0.35 to 0.02. The mixed-data GRU reaches 0.04 on both.

**Interpretation.**

- **Compounding.** In this deterministic environment the GRU's own predictions are exact, so feeding them back does not corrupt the observation stream; on in-distribution data the error grows only through the small per-step reward residuals, roughly linearly (0.002-0.008 at k = 1, 0.014-0.073 at k = 20). The no-memory control compounds visibly: its imagined trajectory goes wrong at the first junction arrival it cannot predict and stays wrong, so observation accuracy halves by k = 20.
- **Distribution shift.** With actions from a different policy, the uniform-data GRU still imagines every observation correctly, but its reward predictions on steps with real reward 0 drift as the agent waits at the junction for longer than anything seen in training. The error accumulates over the horizon to an imagined 20-step return off by up to 1.6, more than a whole `+1` reward, in several seeds, and it grows with training on the uniform set. This is the failure that matters for Stage 2E: an actor trained in imagination would be scored by rewards that do not exist.
- **Data coverage fixes it here.** The same model, budget and data size with half the episodes from the shifted policy imagines both evaluation sets almost exactly. This is the remedy the roadmap prescribes (better data coverage, not task-specific code) and the reason Stage 2F must keep collecting real experience with the current policy.
- One-step accuracy alone would have hidden all of this: the uniform-data GRU has 100% teacher-forced accuracy on the shifted set.

## Limits

The environment is deterministic and small, so observation compounding is essentially absent for a good model; compounding will be larger with stochastic transitions and a stochastic latent. Rollouts reuse the real episode's actions instead of choosing them, and the argmax feedback discards model uncertainty. The shifted policy is hand-picked to probe junction waiting; a learned policy can find other unvisited regions. Next is Stage 2E: an actor-critic trained on imagined trajectories, evaluated in the real T-maze (`docs/roadmap.md`), done in `docs/stage2e.md`.
