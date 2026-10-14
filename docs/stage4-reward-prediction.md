# Stage 4: reward prediction on the Minecraft RSSM

Question: why did the reward head of the first actor experiment (`docs/stage4-minecraft-actor.md`) predict reward that the game does not pay, and does learning the reward together with the recurrent dynamics fix it?

Answer: **no.** The reward targets were correctly aligned. A head on states that carry the whole episode now ranks held-out rewarded steps much better than before. But it still predicts about 100 times the real reward on the old actor's episodes, and the reward-aware partial unfreeze does not fix that while it makes the world model slightly worse. The stop conditions were met, so no actor was trained.

Code: `src/minecraft_rl/minecraft_reward_model.py`, `src/minecraft_rl/minecraft_reward_train.py`. Run: `python -m minecraft_rl.minecraft_reward_train --output runs/stage4/reward-fix`, CPU only, no game process. Results are in `runs/stage4/reward-fix/`.

## Data

- Training: random exploration episodes 0-12 of `runs/stage4/random`, 13,000 transitions, 74 rewarded.
- Held out: episodes 13-15, 3,000 transitions, 13 rewarded (3 first-step seen, 8 later seen, 1 broken, 1 acquired).
- Exploit set: the 16 real episodes of the old actors (`eval-actor0`, `eval-actor200`, `eval-actor1000`, `eval-onpolicy1000`), 16,000 transitions, 76 rewarded. Evaluation only; no stage trains on it.

With 13 held-out positive steps, every held-out number below has a large error. One step more or less changes a recall by 0.08.

## 1. Reward alignment

`reward[t]` is the reward of action a_t, from o_t into o_{t+1}. The replay window takes observations o_s..o_{s+L} and actions a_s..a_{s+L-1}; the filter gives s_s..s_{s+L}; the old head was trained on s_{s+1}..s_{s+L} against `reward[s..s+L-1]`. So s_{t+1} predicts `reward[t]`, in training, in evaluation and in imagination (`imagine` takes the reward of s_{t+1} after action a_t). No index is shifted. A test fixes this pairing (`test_reward_t_is_paired_with_the_state_after_action_t`).

One mismatch exists, but it is not the cause. The old head was trained and evaluated on states that restart h at 0 every 32 steps, while the actor acts with h carried over the whole episode. The old head on both kinds of state:

| Old head | Held out, windows of 32 | Held out, whole episode | Exploit, windows of 32 | Exploit, whole episode |
|---|---|---|---|---|
| Predicted reward per step | -0.0005 | -0.0009 | 0.021 | 0.019 |
| Real reward per step | 0.00013 | 0.00013 | 0.00014 | 0.00014 |

The false reward on the exploit episodes is the same for both. All new runs use whole-episode states: h_0 = 0 at the first observation, carried over every step, reset at the next episode.

## 2. Does the frozen RSSM keep the episode history?

Probe: for each of the 8 block types seen in at least 5 training episodes, a label says "this type was in view earlier in the episode and is not in view now". A logistic regression on h is fitted on training states and scored on held-out states (ROC AUC, 0.5 is chance). The same probe on the current observation embedding e is the no-memory control.

| Probe input | Mean AUC (8 types) |
|---|---|
| h, frozen RSSM | 0.81 |
| e, current observation only | 0.84 |
| h, after the partial unfreeze | 0.80 |

h does not beat the control. The control is high because in a fixed world the current view and the time in the episode tell much about what was seen before. So this probe gives no evidence that h stores which block types the episode saw, and the partial unfreeze did not change that.

## 3. Tiny overfit

A reward head on the frozen whole-episode states of one training episode (6 rewarded steps), 1,000 updates: positive MAE 0.0000, average precision 1.0, precision and recall 1.0 at threshold 0.005, no false positive. The head can fit known positive steps.

## 4. Frozen versus partially unfrozen

- A. Frozen: the canonical RSSM, a new reward head on whole-episode states, 3,000 updates.
- B. Reward-aware: the GRU cell, the action layer, the prior network and the posterior network learn (774,016 parameters, `TRAINABLE_MODULES`). The encoder, decoder and continuation head stay frozen. Loss: the canonical objective G with pitch scale 128 (unchanged) plus 825 times the reward loss, so the reward counts as one more loss group. Each chunk starts from the stored whole-episode h of the current model, refreshed every 100 updates. 1,000 updates, learning rate 1e-4 for the core, 593 s.

Both use the reward loss of the first experiment (targets times 100, rewarded steps weigh 20 times more) and 32-step chunks, half of them centered on a rewarded step. Evaluation uses the original distribution, not a balanced set.

| | Old head | A. Frozen | B. Reward-aware |
|---|---|---|---|
| Held out, positive MAE (zero baseline 0.030, 13 steps) | 0.031 | 0.020 | 0.018 |
| Held out, zero MAE (2,987 steps) | 0.0016 | 0.0011 | 0.0043 |
| Held out, average precision (chance 0.004) | 0.010 | 0.16 | 0.12 |
| Held out, precision / recall at 0.005 | 0 / 0 | 0.05 / 0.31 | 0.008 / 0.38 |
| Held out, false-positive rate at 0.005 | 0.021 | 0.024 | 0.22 |
| Held out, precision / recall at 0.02 | - | 0.25 / 0.23 | 0.07 / 0.23 |
| Training, average precision (74 steps) | - | 0.71 | 0.39 |

By held-out event type, mean predicted reward:

| Event (count) | Real | A. Frozen | B. Reward-aware |
|---|---|---|---|
| Seen at step 0 (3) | 0.03-0.05 | 0.043 | 0.048 |
| Seen later (8) | 0.01 | 0.0006 | 0.0053 |
| Broken (1) | 0.05 | -0.0012 | 0.0035 |
| Acquired (1) | 0.10 | 0.0054 | 0.0026 |
| Zero, a paid type in view again (2,754) | 0 | 0.0002 | 0.0043 |

The gain over the old head comes almost only from the step-0 reward, which every episode pays for the start view. Later novelty, breaking and acquiring stay near zero on held-out episodes in both runs. The reward-aware run raises the prediction everywhere, on repeats as much as on later novelty, which is not memory.

## 5. The old exploit episodes

Predicted reward per step on the old actors' real episodes (real reward about 0.00015):

| Episodes | Old head | A. Frozen | B. Reward-aware |
|---|---|---|---|
| Untrained actor | -0.0013 | 0.0008 | 0.0056 |
| Actor, 200 updates | 0.0098 | 0.019 | 0.0015 |
| Actor, 1,000 updates | 0.035 | 0.027 | 0.0091 |
| Actor after the on-policy round | 0.033 | 0.020 | 0.033 |
| All 16 episodes | 0.019 | 0.017 | 0.012 |

On all 16 exploit episodes, the false-positive rate at 0.005 is 0.37 for the old head, 0.35 for A and 0.42 for B. Almost all of the false reward is on zero-reward steps where a paid type is in view again (15,299 steps, mean prediction 0.018 for A). These states are far from the random-exploration states: on held-out random episodes the same kind of step gets 0.0002. The head extrapolates on views and actions that the random policy rarely produced, such as spinning and constant `use`. B lowers the false reward for two of the four actors and keeps it for the others. Neither run reduces it substantially.

## 6. The world model after the partial unfreeze

Stage 3 held-out splits, 128 windows each, the same evaluation as Stage 3:

| | eval_episode, canonical | eval_episode, B | eval_seed, canonical | eval_seed, B |
|---|---|---|---|---|
| Changed-ray accuracy | 0.358 | 0.343 | 0.279 | 0.255 |
| Ray accuracy | 0.750 | 0.724 | 0.630 | 0.607 |
| Ray distance error (blocks) | 2.85 | 2.97 | 3.40 | 3.54 |
| Pitch error (degrees) | 3.74 | 3.99 | 5.69 | 6.02 |
| KL (nats) | 8.2 | 9.7 | 10.1 | 12.4 |
| Active latent variables | 16 | 16 | 16 | 16 |
| Changed rays, imagination horizon 10 | 0.244 | 0.234 | 0.220 | 0.178 |

The latent did not collapse, but every observation metric got worse, by 4-19%. B is a cost without a gain.

## Result

- Reward alignment: correct. No timing bug.
- Episode history: no evidence that h stores which types were seen, frozen or after the partial unfreeze.
- The frozen head on whole-episode states ranks held-out rewarded steps better than the old head (average precision 0.16 against 0.010), but this rests on the step-0 reward of 3 held-out episodes.
- Stop conditions met: false reward on the exploit episodes stays near 100 times the real reward (0.012-0.017 against 0.00014), and the partial unfreeze makes the observation dynamics worse. No actor was trained; the canonical world model stays canonical; the reward-aware checkpoint is an experiment artifact only.

## The blocker

Reward-model generalization outside its data. Rewards after step 0 are rare (61 in 13,000 training steps), all come from random exploration, and the head pays reward on actor states that random exploration never visited. Episode memory may also be missing, but it is not the first failure: the false reward is on states where nothing new is in view.

The next decision is a reward design and data question, not a model change: either a reward that depends less on episode memory, or reward data from the states the actor visits, with the head checked on them before any actor update.
