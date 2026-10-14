# Stage 4: reward head with actor data, and one actor retry

Question: does adding the recorded actor episodes to the reward-head data fix the false reward that the first actor exploited (`docs/stage4-reward-prediction.md`)?

Answer: **only for the states in the data.** On held-out actor episodes the new head predicts 0.0007 reward per step against 0.016 for the old one, and it keeps detecting rewarded steps. But the first actor trained on it found a new behavior that the head had never seen, and the head again paid false reward there (45 times the real reward). Real return fell below random exploration. Decision: **redesign the reward.**

Code: `actor_data` stage of `python -m minecraft_rl.minecraft_reward_train --output runs/stage4/reward-fix`, then `minecraft_actor_train` and `minecraft_play`. Results: `runs/stage4/reward-fix/actor_data.json`, `runs/stage4/actor-r3/`, `runs/stage4/eval-actor-r3/`.

## Setup

- World model: the canonical Stage 3 RSSM, frozen. The partially unfrozen checkpoint is not used.
- States: whole-episode posterior states, h_0 = 0 at each episode start.
- Head, loss, sampling and budget: as the frozen head A of the previous follow-up (3,000 updates, targets times 100, rewarded steps weigh 20 times more, half of the 32-step chunks centered on a rewarded step).
- Split by whole episode, fixed before training: episode 3 of each actor run is held out.

| | Episodes | Transitions | Rewarded steps |
|---|---|---|---|
| Random training | 13 | 13,000 | 74 |
| Actor training (episodes 0-2 of 4 runs) | 12 | 12,000 | 55 |
| Random held out (episodes 13-15) | 3 | 3,000 | 13 |
| Actor held out (episode 3 of 4 runs) | 4 | 4,000 | 21 |

Head A trains on random data only; head B on random plus actor training data. The held-out counts are small, so a recall moves by 0.05-0.08 per step.

## Reward prediction

| | First actor head | A: random only | B: random + actor |
|---|---|---|---|
| **Random held out** | | | |
| Positive MAE (zero baseline 0.030) | 0.031 | 0.020 | 0.019 |
| Zero MAE | 0.0016 | 0.0011 | 0.0015 |
| Average precision (chance 0.004) | 0.010 | 0.16 | 0.26 |
| Precision / recall at 0.005 | 0 / 0 | 0.05 / 0.31 | 0.05 / 0.46 |
| False-positive rate at 0.005 | 0.021 | 0.024 | 0.041 |
| Predicted / real reward per step | -0.0009 / 0.00013 | 0.0003 / 0.00013 | 0.0007 / 0.00013 |
| **Actor held out** | | | |
| Positive MAE (zero baseline 0.036) | 0.041 | 0.034 | 0.029 |
| Zero MAE | 0.021 | 0.018 | 0.0013 |
| Average precision (chance 0.005) | 0.017 | 0.009 | 0.20 |
| Precision / recall at 0.005 | 0.007 / 0.48 | 0.007 / 0.48 | 0.05 / 0.29 |
| Precision / recall at 0.02 | 0.008 / 0.48 | 0.010 / 0.43 | 0.44 / 0.19 |
| False-positive rate at 0.005 | 0.38 | 0.33 | 0.027 |
| Predicted / real reward per step | 0.019 / 0.00019 | 0.016 / 0.00019 | 0.0007 / 0.00019 |

Positive events after step 0 (mean prediction, share called at 0.005):

| Held-out event (count) | Real | A | B |
|---|---|---|---|
| Random: seen later (8) | 0.01 | 0.0006, 0% | 0.0031, 38% |
| Random: broken (1), acquired (1) | 0.05, 0.10 | -0.001, 0.005 | -0.0003, 0.005 |
| Actor: seen later (11) | 0.01 | 0.011, 18% | 0.0020, 18% |
| Actor: broken (5) | 0.05 | 0.050, 80% | -0.0004, 0% |
| Actor: acquired (1) | 0.10 | -0.001 | -0.0007 |

Step-0 novelty is detected by A and B in every held-out episode (7 of 7). After step 0, B detects 38% of the random "seen later" steps and 18% of the actor ones, and no broken or acquired step. A's 80% on actor breaking comes with a 33% false-positive rate, so it is not detection.

## False reward on the old exploit behaviors

Zero-reward steps of the held-out actor episodes, mean prediction (false-positive rate at 0.005):

| Behavior (steps) | First actor head | A | B |
|---|---|---|---|
| Camera turn of 10 degrees or more (2,299) | 0.020 (0.31) | 0.021 (0.32) | 0.0005 (0.02) |
| Jump or use held (2,587) | 0.016 (0.24) | 0.018 (0.27) | 0.0005 (0.02) |
| A paid block type in view again (3,855) | 0.019 (0.39) | 0.017 (0.34) | 0.0006 (0.03) |

On these behaviors, B removes the false reward without a rule for any behavior.

## Does the latent state carry the novelty history?

For every rewarded step after step 0, the distance from its state to the nearest zero-reward state of another episode, divided by the median distance between two consecutive states of one episode:

| Held out | Steps | Median ratio | Share at most 1 | Share at most 2 |
|---|---|---|---|---|
| Random | 10 | 1.18 | 0% | 90% |
| Actor | 17 | 1.28 | 18% | 100% |

A rewarded state is about as far from some unrewarded state as one step is from the next. With the memory probe of the previous follow-up (h does not beat the current observation), this says the state does not separate "new" from "seen before" clearly. For 18% of the actor steps an unrewarded state is closer than one step. The head can learn which views pay in the data it has, but not the history rule behind them.

## Actor retry

One retry, as allowed: head B, the canonical RSSM, the fixed world and spawn, the existing actor-critic, 200 updates (`runs/stage4/actor-r3`, 55 s). Four real episodes of 1,000 steps with video (`runs/stage4/eval-actor-r3`).

| | Random exploration (16) | First actor, 200 updates (4) | Retry, 200 updates (4) |
|---|---|---|---|
| Real return | 0.171 (sd 0.088) | 0.080 | 0.078 (0.07-0.08) |
| Block types seen | 8.0 | 8.0 | 7.75 |
| Blocks broken, items acquired | 1.06, 0.31 | 0, 0 | 0, 0 |
| Imagined return (last 50 updates) | - | 1.24 | 0.28 |
| Head on its own real states, per step | - | 0.0098 | 0.0035 |
| Real reward per step | 0.00017 | 0.00008 | 0.00008 |

The imagined return is lower than in the first run, but the head again predicts 45 times the real reward on the new actor's own states, with a false-positive rate of 0.22 at 0.005. Real return equals the step-0 reward only: the actor never breaks a block.

Behavior from the action log: forward, sneak and sprint 100% of the time, never jump or attack, use 2%, and a steady turn of about 9,500 degrees per episode (14 degrees per step). Entropy fell from 11.5 to 2.2 nats in 200 updates; every factor except yaw is nearly fixed. It is camera spinning again, now with sneaking instead of jump and use. Sneak alone is not new to the head: it is held in 10% of the random steps and 22-48% of the steps of two training actors. The new part is the fixed combination, forward, sneak and sprint held with no jump, attack or use: 53 of the 25,000 training steps have it (none in random data), and the retry holds it on almost every step while turning. The head again paid false reward on states that its data barely covers, the same failure as before in a new place.

Video: frames were decoded from the recorded MP4 files with AVFoundation and compared to the random and first-actor videos (64 x 36, one frame per 0.5 s, not watched by a person). The retry videos change less between frames than the random ones (0.065-0.070 against 0.102-0.106 mean pixel change), consistent with slow sneaking while turning in place, and show sky in 53-58% of the image, the level of the start view. No death, no dark frames.

Stop condition met: the actor learned camera spinning again. No further training.

## Decision: redesign the reward

Better data coverage fixes the false reward only on the states that the data covers. Each new actor finds states outside it, and the novelty reward gives the head no rule that transfers: whether a step pays depends on the episode history, and the state does not separate that history.

Recommended smaller redesign, not implemented here:

1. Remove seen novelty (`block_seen_novelty`, 0.01) from the scalar reward. It pays mostly at step 0, needs long episode memory, and is the reward that spinning chases.
2. Keep the reward components that change the HUD-visible self state: item acquisition and importance, which show in the inventory the agent observes. A head can predict them from the change in o_t to o_{t+1} without episode memory, apart from the "first time" rule.
3. Make "first time this episode" checkable from the observation where possible, or report those components separately, so their prediction error is measured on its own.
4. Keep breaking novelty only if the same test shows it can be predicted on held-out actor data.

## Heat

This follow-up adds `src/minecraft_rl/chunk_gate.py` and `minecraft_play --background-game` (macOS). The game runs with the background policy on the efficiency cores. Before the first observation and before every step, a privileged chunk query checks that every sensor-range chunk is loaded and entity-ticking and every simulation chunk is entity-ticking. If not, the gate waits while the world is frozen, and after 0.5 s moves the game to default priority until the chunks are ready. The policy never sees the query, and the gate never changes what a step does.

Measured on the actor-retry evaluation (4 episodes, 4,000 steps, recording on): the gate waited before 73 of 4,004 checks, 42 s in total, longest 6.4 s; it boosted 11 times for 22 s. The game used 153 J (0.038 J per step), 0.1 J of it on the performance cores; the wall time was 195 s (20.5 steps per second). The earlier default-priority measurement was 0.076 J per step at 142 steps per second (`docs/decisions/simulation-throughput.md`). This is one run; the chunk readiness that the gate checks was not compared to a default-priority run of the same episodes.
