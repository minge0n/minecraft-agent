# Stage 2G: RSSM world model in the toy loop

Purpose: replace the deterministic GRU world model of Stages 2C-2F with an explicit recurrent state-space model (RSSM). An RSSM splits the state into a deterministic part and a stochastic part, and trains the stochastic part with a KL penalty. The roadmap requires it before Minecraft. This stage checks that the architecture trains stably, that its latent carries information, and that it works in the imagination, actor-critic and integrated loops of Stages 2D-2F.

The T-maze of these stages is deterministic. Nothing in it needs a stochastic latent, so this stage cannot show the main advantage of an RSSM. It tests the architecture and its stability. Stage 2H (`docs/stage2h.md`) tests a stochastic environment.

## Architecture

`src/minecraft_rl/rssm.py`. Sizes: hidden size H = 32, V = 8 latent variables with K = 4 classes each. The model has 11,463 parameters (GRU world model: 4,263).

```text
h_t = GRUCell(h_{t-1}, [z_{t-1}, onehot(a_{t-1})])      h_0 = 0          (32)
prior      p(z_t | h_t)            = MLP(h_t)             logits (8, 4)
posterior  q(z_t | h_t, o_t)       = MLP(h_t, onehot(o_t)) logits (8, 4)
z_t: one sample per variable, as 8 one-hot vectors      (8 x 4 = 32)
heads on s_t = [h_t, z_t] (64): observation o_t, reward r_t, continuation c_t
```

Terms:

- `h_t` is the deterministic state. A GRU cell updates it from the previous state, the previous latent and the previous action.
- `z_t` is the stochastic state: 8 categorical variables, each with 4 classes.
- The prior predicts `z_t` from `h_t` alone, before the model sees `o_t`. Imagination uses the prior.
- The posterior predicts `z_t` from `h_t` and the observation `o_t`. Training and acting on real data use the posterior.

An observation reaches `h` only through `z`. So everything that the model must remember about an observation, for example the cue, must pass through a posterior sample.

Sampling uses the Gumbel-max trick with an explicit `torch.Generator`. In training, the straight-through estimator passes gradients through the sample: the forward value is the one-hot sample, and the backward gradient is that of the class probabilities.

## Loss

The loss is computed per valid state of a sequence. It is the negative evidence lower bound:

```text
loss = cross_entropy(o_k) + mse(r_k) + bce(c_k)
     + KL(sg(q) || p)                      prior part
     + max(free_nats, KL(q || sg(p)))      posterior part
```

`sg` stops the gradient. Both KL parts have the same value, KL(q || p), summed over the 8 variables. The prior part trains only the prior toward the posterior. The posterior part trains only the posterior toward the prior. Both scales are 1. With free nats F, the posterior part is constant while the KL is below F, so the posterior gets no KL gradient there. The prior part has no threshold, so the prior always learns to predict the posterior.

In the T-maze, only the cue step carries information that the prior cannot predict. One bit of cue costs at least ln 2 = 0.69 nats of KL there.

## Free-nats ablation

The first RSSM runs used F = 1.0. About 99% of the states had a KL below 1.0, so they gave the posterior no KL gradient. The question was whether this threshold stops useful latent learning. The ablation changed only F: 0.0, 0.1 and 1.0, seeds 0-2. All other settings stayed the same.

The value 0.1 is useful because it lies between the later-step KL (about 0.015) and the cue-step KL (0.3-0.9). With F = 0.1, the later steps are masked and the cue step is not.

T-maze world model after 5,000 updates, mean over seeds 0-2 [min, max]:

| Measurement | F = 0.0 | F = 0.1 | F = 1.0 |
|---|---|---|---|
| Raw KL at the cue step (nats) | 0.50 [0.35, 0.63] | 0.41 [0.40, 0.43] | 0.87 [0.86, 0.90] |
| Raw KL at later steps | 0.015 | 0.015 | 0.003 |
| KL term in the loss, after the threshold | 0.085 | 0.165 | 1.054 |
| States below the threshold (masked) | 0% | 92% | 99.9% |
| Cue steps below the threshold | 0% | 0% | 100% |
| Prior entropy at later steps | 1.51 | 0.65 | 0.40 |
| Posterior entropy | 1.74 | 0.95 | 0.56 |
| Active variables (mean KL above 0.01), of 8 | 1.3 | 1.3 | 1.7 |
| Classes used, of 32 | 26.7 | 26.3 | 27.7 |
| Prior-posterior agreement | 0.975 | 0.988 | 0.990 |
| Variables that code the cue | 5.3 | 2.7 | 2.7 |
| Observation accuracy | 0.991 | 0.992 | 0.999 |
| Turn-reward sign accuracy | 0.76 [0.47, 0.93] | 0.77 [0.47, 0.96] | 0.85 [0.57, 0.99] |
| Continuation accuracy | 1.000 | 1.000 | 1.000 |

The signal T-maze of Stage 2H shows the same pattern more clearly (details in `docs/stage2h.md`). With F = 0.0, the cue is spread over 2-4 latent variables, and 69% of the prior samples at the signal step are a mixed code that no real signal produces. Then the imagined reward after an imagined right signal is only -0.38 instead of -1.

Answers to the questions of the ablation:

1. Lower free nats did not increase the information in the latent. With F = 0.0, the KL penalty pushes the cue to fewer nats (0.50 against 0.87). The cue is then coded weakly, in more variables.
2. Latent use did not improve. The number of active variables and classes is the same for all three values. Each model uses 1-2 of 8 variables, which matches the one bit of hidden information in the task.
3. The prior learned the immediate signal distribution with every value (Stage 2H).
4. The link from the sampled signal to the later reward is strongest with F = 1.0 and weakest with F = 0.0.
5. No value made training unstable or made the latent collapse.

The two observations from the first runs are therefore not related in the expected way. The 99% masked share at F = 1.0 does not stop useful learning. The prior part of the KL has no threshold, and the cue step is the only state that needs KL. The weak link from the signal to the reward in the first signal T-maze probe came from a short run (1,500 updates) and from a probe that measured the cue step at the wrong state.

Decision: F = 1.0 stays the canonical value. It gives the clearest cue code, the best link from the signal to the reward and the most accurate predictions, and it is the existing default. F = 0.0 and 0.1 remain ablation results. Three seeds is a small sample, so the ranking of 0.1 and 1.0 on some measurements is uncertain.

## Imagination

Imagination in latent space advances `h` with the actions and samples `z` from the prior. It decodes no observation. The actor and the critic act on `s_t = [h_t, z_t]`. For evaluation, the RSSM also runs with mode latents as a diagnostic. In this mode, each variable takes its most likely class instead of a sample. Training never uses it.

## Run

```sh
.venv/bin/python -m minecraft_rl.world_model --seed 0 --world-model rssm --output runs/stage2g/rssm/world_model/seed0/metrics.json
.venv/bin/python -m minecraft_rl.sweep dreamer_loop --seeds 0-9 --time-budget 420 --output-root runs/stage2g/rssm/dreamer_loop -- --world-model rssm
```

With `--time-budget`, the sweep stops after the budget, and the same command continues it (`docs/decisions/reproducibility.md`). All results below come from the canonical runtime (one intra-op and one inter-op thread). The RSSM runs used jobs 1 and duty cycle 0.5. One RSSM world-model run takes about 150 s of computation. The GRU runs are the canonical Stage 2C-2F runs. A new GRU world-model run of seed 3 under the current code gave the same values in every field that both versions record.

## Results, seeds 0-9

### World model (Stage 2C setup)

| Measurement | GRU | RSSM |
|---|---|---|
| Observation accuracy | 1.000 | 0.999 |
| Turn-reward sign accuracy, mean / median | 1.000 / 1.000 | 0.897 / 0.990 |
| Turn-reward sign accuracy per seed | 1.00 in every seed | 0.98 0.57 0.99 0.99 1.00 0.47 1.00 0.98 1.00 1.00 |
| Turn-reward mean squared error, mean / median | 0.001 / 0.001 | 0.227 / 0.038 |
| Continuation accuracy | 1.000 | 1.000 |

The no-memory control predicts the turn reward at chance (0.50). In 8 of 10 seeds, the RSSM learns the cue-dependent turn reward. In seeds 1 and 5, it does not learn it within 5,000 updates (0.57 and 0.47). Its observation prediction is still exact there. The GRU learns it in every seed.

RSSM latent, mean over 10 seeds [min, max]:

- Raw KL at the cue step: 0.90 nats [0.79, 0.99]. Later steps: 0.004.
- KL term in the loss after the threshold: 1.056. States below the free-nats threshold: 99.4%.
- Prior entropy: 4.87 at the cue step, 0.36 at later steps. Posterior entropy: 0.57 (maximum 11.09).
- Active variables: 1.5 of 8. Classes used: 28 of 32. Prior-posterior agreement: 0.990.
- Cue intervention. Here the latent at every state that shows a cue is sampled from the prior instead of the posterior. The turn-reward sign accuracy then drops from 0.898 to 0.494, which is chance. So the cue reaches the reward only through `z`. If only the first cue step is replaced, the accuracy is 0.731. The remaining information comes from later visits to the cue cell: 65-68% of the random-policy episodes see the cue more than once.

### Imagination (Stage 2D setup)

Open-loop return error at horizon 20, mean over 10 seeds [max]:

| Model and training data | Uniform policy | Shifted policy (`FORWARD` 0.8) |
|---|---|---|
| GRU, uniform data | 0.036 [0.073] | 0.773 [1.585] |
| GRU, mixed data | 0.045 [0.063] | 0.052 [0.108] |
| RSSM, uniform data | 0.163 [0.448] | 0.431 [1.376] |
| RSSM, mixed data | 0.234 [0.385] | 0.369 [0.756] |

The observation accuracy at horizon 20 is 1.000 for every model. On data of its own behavior policy, the RSSM has a 4-5 times larger return error than the GRU, because each sampled rollout carries latent noise. Under the shifted policy, uniform-data RSSMs show the Stage 2D effect less strongly than uniform-data GRUs (0.43 against 0.77). Mixed training data does not remove it for the RSSM as it does for the GRU. In 3 of 10 RSSM seeds, mixed data gave a model that did not learn the turn reward (turn sign accuracy 0.46-0.59).

### Actor-critic in imagination (Stage 2E setup)

Final real evaluation, 256 episodes:

| Measurement | GRU | RSSM | RSSM, mode latents |
|---|---|---|---|
| Greedy success per seed | 0.50, then 1.00 in seeds 1-9 | 0.97 0.00 0.99 0.98 0.99 0.00 0.98 0.96 0.98 0.98 | 1.00, except 0.00 in seeds 1 and 5 |
| Greedy success, mean / median | 0.950 / 1.000 | 0.783 / 0.979 | 0.800 / 1.000 |
| Greedy real return, mean | 0.950 | 0.767 | - |
| Greedy imagined minus real return, mean / median | 0.098 / 0.006 | 0.093 / 0.034 | 0.073 / 0.009 |
| Sampled success, mean / median | 0.949 / 1.000 | 0.782 / 0.979 | 0.800 / 1.000 |

A uniform random policy succeeds in about 50% of episodes. The two failing RSSM seeds are the two seeds whose world model did not learn the turn reward (seeds 1 and 5). Their policies never turn: all episodes end at the step limit. Their imagined return is still positive (0.29 and 0.48). This is model exploitation, and the imagined-versus-real gap shows it. In seed 0 of the GRU, the same gap (+0.93) shows the known Stage 2E exploitation.

In the 8 successful RSSM seeds, greedy success is 0.96-0.99 with normal latent samples and 1.00 with mode latents. So the remaining 1-4% of failures come from latent sampling noise during evaluation, not from a weak policy.

### Integrated loop (Stage 2F setup, 12 iterations)

| Measurement | GRU | RSSM |
|---|---|---|
| Greedy success per seed | 1.00 in every seed | 0.50 1.00 1.00 1.00 0.98 1.00 0.99 1.00 1.00 1.00 |
| Greedy success, mean / median | 1.000 / 1.000 | 0.946 / 0.996 |
| Sampled success, mean / median | 1.000 / 1.000 | 0.948 / 0.996 |
| Greedy imagined minus real return at the end, mean [min, max] | -0.002 [-0.010, 0.006] | 0.027 [-0.021, 0.084] |
| Imagined minus real at iteration 0, range | +0.52 to +2.90 | +0.24 to +2.48 |
| Real transitions in total | 4,848-12,360 | 4,314-9,822 |
| Turn-reward sign accuracy on new policy data, last iteration | 1.000 | 0.997 |

The integrated loop removes the actor-critic failures of seeds 1 and 5: policy data corrects their world models, as in Stage 2F. In 9 of 10 RSSM seeds, the final policy succeeds in 98-100% of episodes. The early imagined-versus-real gap closes in every seed.

RSSM seed 0 ends at 50%: it solves the right cue (100%) but never turns at the left cue. Its world model predicts the turn reward on new policy data correctly from iteration 2 (100% sign accuracy). The episodes it collects with the left cue end at the step limit without a turn. So the model is not wrong about the reward of the data it sees. The policy does not explore the left turn. This is an exploration failure of one seed. This stage changes no exploration setting (entropy coefficient 0.03, as in Stage 2F).

The mode-latent diagnostic gives the same result as normal sampling in the integrated loop (seed 0 at 50%, the other seeds at 100%).

On new policy data at the end of the loop, the RSSM latent has a cue-step KL of 0.84 nats, 98.6% of states below the free-nats threshold, 1.5 active variables and a prior-posterior agreement of 0.989.

## Interpretation

The RSSM trains stably in every run of this stage. Its latent carries the cue in 1-2 variables, the prior and posterior agree on 99% of the variables, and the cue intervention shows that the cue reaches the reward only through the latent. Imagination in latent space works, and the actor-critic and the integrated loop learn on it.

In this deterministic task, the RSSM is worse than the GRU on every measurement:

- It learns the turn reward in fewer seeds (8 of 10 against 10 of 10 after 5,000 updates).
- It has a larger open-loop return error on data of its own policy.
- It reaches 100% success in fewer seeds of the integrated loop.

This is expected. The stochastic latent adds sampling noise and a KL cost, and this task has nothing stochastic to model. It is not evidence against the RSSM for Minecraft.

## Limits

- Three seeds for the free-nats ablation. The decision rests on clear differences in the cue code and the signal-to-reward link, not on small differences.
- The actor gradient is REINFORCE with a critic baseline, as in Stages 2E and 2F. Dreamer's dynamics backpropagation through the world model is a later, separate milestone.
- RSSM seed 0 of the integrated loop shows an exploration failure. This stage adds no exploration method. If the failure appears again in later stages, it needs its own measured experiment.
- One RSSM seed is about 7 times slower than one GRU seed on this CPU (about 150 s against 20 s of computation for 5,000 updates). The cause is the per-step loop of the RSSM (`filter`) and the 16 latent samples per prediction.
