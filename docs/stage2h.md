# Stage 2H: uncertain futures in the signal T-maze

Purpose: test whether a world model can represent a future that is random, and keep each imagined future consistent with its own random outcome. The question is not only whether a model predicts a 70/30 signal correctly. It is whether a sampled signal also produces the correct later reward. Stage 2H compares the deterministic GRU world model of Stages 2C-2F with the RSSM of Stage 2G (`docs/stage2g.md`) on the same data.

## Environment

`src/minecraft_rl/signal_tmaze.py`. The signal T-maze has the observations and actions of the T-maze, corridor length 4 and a step limit of 40.

1. The episode starts on a plain cell that looks like the corridor. The start shows no cue.
2. The first `FORWARD` move reaches the signal cell. Only then does the environment draw the signal: `CUE_LEFT` with probability p = 0.7, else `CUE_RIGHT`.
3. Plain corridor cells follow, then the junction.
4. A turn toward the signalled side gives +1. The other turn gives -1.

So the same observable history (start, `FORWARD`) leads to two different futures, and the later reward depends on which one happened. The true values are:

- P(signal left) = 0.7.
- A left turn gives +1 after a left signal and -1 after a right signal.
- The expected left-turn reward over both signals is 2p - 1 = +0.4. P(reward > 0) is 0.7.
- The lowest possible negative log likelihood (NLL) of the signal is its entropy, 0.611 nats.

Randomness comes only from the `torch.Generator` that `reset` gets. The drawn signal is privileged state, for evaluation only.

## Why deterministic prediction is not enough

A deterministic world model can predict the probability of the next observation. That covers the first step. In imagination it must feed back one observation, its most likely one, so it imagines one future only. After the start it always imagines the more likely left signal, and with it a left-turn reward of +1. The expected reward that it imagines is then +1 instead of +0.4. A policy trained in that imagination always turns left, and loses in 30% of the episodes.

An RSSM samples the signal from its prior. Each imagined future gets its own signal. To be correct, the model must also keep the later reward consistent with the sampled signal: +1 after a sampled left signal, -1 after a sampled right signal.

## Measurements

`src/minecraft_rl/stochastic_world.py`. Both models train with 5,000 updates on the same 512 random-policy episodes. The evaluation uses 256 held-out episodes and a probe of 2,000 open-loop rollouts. Each probe rollout starts at the start cell and takes `FORWARD` to the junction, then `LEFT`.

- Signal probability: the one-step predicted P(signal left) after (start, `FORWARD`), against the true 0.7 and against the held-out signal frequency.
- Signal NLL: the mean -log P(real signal) over held-out episodes whose first action is `FORWARD` (80-97 episodes per seed).
- Imagined signal: the fraction of probe rollouts that imagine a left signal or a right signal.
- Conditional reward: the imagined left-turn reward given the imagined signal, and its share of positive values.
- Observed signal: the imagined left-turn reward after the model reads a real left or right signal. This separates two failures. The model can fail to sample the signal, or it can fail to carry a known signal to the reward.
- Expected return: the mean imagined left-turn reward over all rollouts, against the true +0.4.
- RSSM latent: KL and free-nats share, the prior samples at the signal step classified as the left code, the right code or a mixed code, and the turn-reward sign accuracy when the signal-step latent comes from the prior.

## Run

```sh
.venv/bin/python -m minecraft_rl.stochastic_world --seed 0 --output runs/stage2h/seed0/metrics.json
.venv/bin/python -m minecraft_rl.sweep stochastic_world --seeds 0-9 --output-root runs/stage2h/stochastic_world
```

All results come from the canonical runtime (one intra-op and one inter-op thread), jobs 1 and duty cycle 0.5, with the canonical RSSM configuration (free nats 1.0). One seed takes about 180 s of computation for both models. Seeds 0-2 are the free-nats 1.0 runs of the Stage 2G ablation. Seeds 3-9 ran later with a diagnostic that also replaces every later visit to the signal cell.

## Results, seeds 0-9

| Measurement | Truth | GRU, mean [min, max] | RSSM, mean [min, max] |
|---|---|---|---|
| One-step P(signal left) | 0.70 | 0.691 [0.61, 0.75] | 0.671 [0.62, 0.71] |
| Absolute error against 0.7 | 0 | 0.022 | 0.032 |
| Signal NLL, held out | 0.611 at best | 0.595 [0.56, 0.68] | 0.625 [0.57, 0.69] |
| Imagined left signal | 0.70 | 1.000 in every seed | 0.693 [0.64, 0.73] |
| Imagined right signal | 0.30 | 0.000 in every seed | 0.307 [0.27, 0.36] |
| Imagined E[left-turn reward] | +0.40 | +1.003 [+0.95, +1.05] | +0.379 [+0.23, +0.46] |
| Absolute error of E[reward] | 0 | 0.603 | 0.064 [max 0.174] |
| P(reward > 0) | 0.70 | 1.000 | 0.729 |
| Reward given imagined left signal | +1 | +1.003 | +0.885 (median +0.947) |
| Reward given imagined right signal | -1 | never imagined | -0.794 (median -0.952) |
| P(reward > 0) given imagined right signal | 0 | never imagined | 0.103 (0.000 in 9 seeds) |
| Reward given observed left signal | +1 | +1.003 | +0.882 |
| Reward given observed right signal | -1 | -0.943 | -0.671 |
| Turn-reward sign accuracy, held out | - | 1.000 | 0.961 (median 0.988) |

The held-out signal frequency of each seed lies between 0.64 and 0.76 (mean 0.727), because 80-97 episodes reach the signal cell. The NLL below the entropy bound of 0.611 for some GRU seeds comes from this sample noise, not from a better model.

## GRU behavior

The GRU predicts the signal probability as well as the RSSM, and its one-step NLL is slightly better. Its turn reward after a real signal is also correct. But in imagination it commits to the more likely signal in 100% of the rollouts. It imagines a left-turn reward of +1.00 where the true expectation is +0.40. Its expected-return error is 0.60 in every seed. The GRU therefore represents the uncertainty only for one step. It cannot carry it into an imagined future.

## RSSM behavior

In 9 of 10 seeds, the RSSM represents the uncertain future and keeps the dependency on the hidden outcome:

- Prior samples at the signal step give the left code in 70% and the right code in 28% of the rollouts. 2.4% are mixed codes: in seed 0, 24% of the samples combine its two cue-coding variables, and in the other 9 seeds the cue lives in one variable.
- After a sampled right signal, the imagined left-turn reward is negative in 100% of the rollouts in 9 seeds (median -0.95). After a sampled left signal, it is positive in 100% of the rollouts (median +0.95).
- The imagined expected reward is within 0.06 of the true +0.40 in 7 seeds.

The latent behaves as in Stage 2G. The KL at the signal step is 0.53 nats [0.47, 0.62], which is close to the 0.61 nats of the true signal entropy. Of all states, 99% are below the free-nats threshold. Each model uses 1-2 of 8 variables, and the prior and posterior agree on 99.7% of the variables. If the signal-step latent comes from the prior, the turn-reward sign accuracy drops from 0.96 to 0.81. If every visit to the signal cell does (seeds 3-9), it drops to 0.60. The remaining accuracy above chance (0.5) comes from the turn direction alone: a left turn is right in 70% of the episodes, and a model that ignores the signal reaches 0.705 on the held-out turns.

## Negative findings

- RSSM seed 6 did not learn the link from the signal to the reward. It predicts the turn reward from the turn direction alone: +0.29 for every left turn and -0.38 for every right turn, for both signals. Its sign accuracy (0.73) is the no-signal baseline (0.705). Its prior still samples the signal correctly (64/36). This seed matches the marginal distributions but lost the dependency, the failure that this stage was designed to detect.
- The RSSM carries a known signal to the reward less exactly than the GRU. After a real right signal, its imagined left-turn reward is -0.67 against -0.94 for the GRU. After a real left signal, it is +0.88 against +1.00. The RSSM rollouts after an observed signal also sample later latents from the prior, and part of their reward prediction drifts toward the mean.
- With free nats 0 (Stage 2G ablation, seeds 0-2), the RSSM spread the cue over 2-4 variables, and 69% of its prior samples were mixed codes. The imagined reward after a right signal was then only -0.38. A stochastic latent alone is not enough. The cue must sit in few variables, because the prior samples each variable independently.
- The GRU has the better one-step signal NLL (0.595 against 0.625). The RSSM advantage exists only in imagination over several steps.

## Policy results

This stage trains no policy on the signal T-maze. The expected-return error predicts what an imagination-trained policy would see. The GRU imagines that a left turn is always safe (+1.00 against the true +0.40). The RSSM imagines the correct mixture in 9 of 10 seeds. A policy experiment on this task, with the same actor-critic, is left for later.

## Interpretation

The stochastic latent of the RSSM represents the uncertain future and preserves the dependency between the hidden random outcome and its later consequence in 9 of 10 seeds. The deterministic GRU predicts the next-step probability correctly but cannot keep the uncertainty in imagination, so its imagined expected return is wrong by 0.6 in every seed. Stage 2G did not show this advantage, because its task has no random event. It does not remove the failure modes. One seed lost the dependency while it kept the correct marginal distribution, and the free-nats ablation shows that the latent code must be compact for the prior to sample consistent outcomes.

## Limits

- One random event per episode, with two outcomes. Minecraft has many correlated random events.
- Random-policy training data only. Policy data can make one outcome rare in the data.
- The probe uses one fixed action sequence. The conditional reward is measured for the left turn only.
- Seeds 0-2 have no measurement of the every-visit signal intervention, because they ran before it existed.
