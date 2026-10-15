# Stage 4: block-breaking diagnostic

Question: can the Minecraft Dreamer actor learn to break a real block with the existing camera and attack controls?

Answer: **no, not in this run.** After 200 actor updates the actor looks down at the block but stops pressing attack (0.1% of steps) and breaks nothing. Random persistent actions break 5 blocks in 6 episodes. The reward predictor is useful after continued world-model training, and imagination predicts reward for held attack. The measured blocker is **actor action persistence**: a break needs 15 consecutive attack ticks, the actor samples attack independently every step from a nearly uniform start, and with that start no imagined 15-step rollout completes a break. The actor learned that attack costs a little predicted reward and never found the long hold.

`reward-v1`, `reward-control-v1` and `reward-pitch-v1` are unchanged.

## Block and start

World seed 100000 with the fixed spawn: the player stands at x 192.5, y 69, z 112.5 on **sand**, facing south, pitch 0. The block under the player and the sand around it are in reach when the player looks down. Sand is weak and breaks by hand. No custom world.

## Mining semantics (probe, test-only scripted attack, never training data)

| Measurement | Result |
|---|---|
| Held attack ticks to break the sand, empty hand | 15 (at pitch 30, 45, 50, 60, 70 and 90) |
| Releasing attack for 5 ticks after 10 ticks, then holding again | 15 more ticks: progress resets |
| Pitch wobble of ±3 or ±10 degrees while holding | still 15 ticks while the crosshair stays on the same block; a wobble that moves it to another block resets progress |
| Item | the sand item is in the inventory 9 ticks after the break |
| Observation after the break | the center ray's distance grows by 0.9 blocks (sand gone, the block below now hit) |
| Attack at pitch 0 for 100 ticks | 0 breaks (nothing in reach) |

## Diagnostic reward and actions

`src/minecraft_rl/minecraft_reward_break.py`:

```text
reward-break-v1:  reward[t] = 1.0 * (blocks broken by the player in step t)
```

From the privileged block-break event (`StepEvents.blocks_broken`), stored as `privileged_breaks` per step, never in an `obs_*` array. Every break pays, the first and the tenth alike; attack presses, turns and unfinished attempts pay nothing.

Restricted action space for this diagnostic: `BreakAgent` (a `RestrictedAgent`, the existing actor and critic) chooses pitch and attack; every other factor stays at no-op. No yaw, no movement, no `mine_block()` action.

## Data

The existing datasets have too few breaks: the 16,000 random steps of `runs/stage4/random` have 17 steps with a first break of a block type (`reward-v1` novelty), and the total number of breaks was not recorded. A random baseline with independent per-step pitch choices broke nothing in 12 episodes of 100 steps: its pitch rarely stayed on one block for 15 ticks. `RandomBreakPolicy` therefore keeps its choices persistent, as the exploration policy does: a pitch choice is held for 10 steps on average, and attack starts with probability 0.1 per step and is held for 20 steps on average. No choice depends on the observation; it is action exploration, not a mining routine.

`runs/stage4/break-random2`: 30 episodes of 200 steps, 6,000 transitions, **19 breaks** in 11 episodes, all sand, unique types 1. The breaks came after 15-53 attack ticks, 14 of 19 at pitch 88-90.

## Reward prediction

The existing reward head on whole-episode states (rare-event loss of the first actor experiment: positive weight 20, half the chunks centered on a break). The fixed split (24 training, 6 held-out episodes) leaves only 2 held-out breaks, so the main measure is 5-fold cross-validation over whole episodes: every break is scored by a head that never trained on its episode.

| 5-fold cross-validation, 19 breaks in 6,000 steps | Canonical RSSM | Continued RSSM |
|---|---|---|
| Average precision (chance 0.003) | 0.019 (6 times chance) | 0.105 (33 times chance) |
| Mean prediction on breaks / other steps | 0.21 / 0.024 | 0.48 / 0.012 |
| Precision / recall at 0.3 | 0.02 / 0.21 | 0.13 / 0.58 |
| False-positive rate at 0.3 | 0.028 | 0.013 |
| Other steps by current attack run: 0, 1-14, 15-29, 30+ | 0.006, 0.020, 0.059, 0.054 | 0.002, 0.016, 0.018, 0.021 |

The canonical frozen RSSM does not represent the break: its one-step prior predicts the center ray getting 1.7 blocks shorter on break steps, where the real ray gets 0.9 blocks longer. But h does carry the attack history: a ridge probe on h recovers the length of the current attack run on in-reach states (R^2 0.67, against 0.15 from the observation alone).

### Continued world-model training

Allowed by the plan when the frozen model lacks the dynamics: the same architecture, the canonical objective (G, pitch 128), Adam 1e-3, 1,000 updates, each batch 8 windows of the 24 new training episodes and 8 windows of the Stage 3 training data (`runs/stage4/break-continued/world_model.pt`). Held-out episodes were never trained on. Stage 3 regression on its held-out splits:

| | eval_episode, canonical | continued | eval_seed, canonical | continued |
|---|---|---|---|---|
| Changed-ray accuracy | 0.358 | 0.357 | 0.279 | 0.257 |
| Ray accuracy | 0.750 | 0.749 | 0.630 | 0.618 |
| Ray distance error (blocks) | 2.85 | 3.24 | 3.40 | 3.32 |
| Pitch error (degrees) | 3.74 | 3.20 | 5.69 | 4.21 |
| Active latent variables | 16 | 16 | 16 | 16 |
| Changed rays, imagination horizon 10 | 0.244 | 0.265 | 0.220 | 0.201 |

Within the same seed the dynamics hold; on the unseen seed they lose 2 points of changed-ray accuracy. No collapse. This checkpoint is a diagnostic artifact; the canonical model is unchanged.

The reward head on the continued model passed the gate (cross-validated average precision at least 10 times chance).

## Imagination and horizon

From 200 in-reach start states looking down, with the continued model and its head:

| Imagined actions | Predicted reward over 15 steps | Over 25 steps |
|---|---|---|
| Attack held every step | +0.30 (57% of rollouts reach a step of 0.3 or more within 25) | +1.10 |
| No attack | -0.11 | -0.19 |
| Attack with probability 0.5 per step (the untrained actor) | -0.10 | - |
| Attack with probability 0.8 per step | +0.11 | - |
| Attack with probability 0.95 per step | +0.21 | - |

The predicted reward rises from step 8 of a held attack and is largest after step 15, so the break fits within the horizon of 15, at its end. The horizon was not changed. A policy that attacks with probability 0.5 per step, the untrained actor's, almost never holds 15 ticks (longest run 3.3 on average) and gets slightly negative imagined reward, because the head gives attack-without-break steps a small negative value.

## Actor

The existing actor-critic, 200 updates, 4,824 start states. Imagined return rose from -0.34 to +0.51 and imagined reward per step from -0.003 to +0.001. Attack entropy fell from 0.69 to 0.01 nats, toward **no attack**; pitch entropy from 1.61 to 0.24, toward ±10 alternation.

## Real evaluation

Six episodes of 200 steps each, world seed 100000, fixed spawn, start pitch 0, video, continued world model for the actors, game on the efficiency cores behind the chunk gate.

| | Random persistent actions | Untrained actor | Actor, 200 updates |
|---|---|---|---|
| Blocks broken (6 episodes) | 5 | 0 | 0 |
| Episodes with a break / with an item | 3 / 3 | 0 / 0 | 0 / 0 |
| Real return per episode | 0.83 | 0 | 0 |
| Attack share of steps | 70% | 50% | 0.1% |
| Longest attack run (per episode) | 23-62 | 6-9 | 0-1 |
| Attack runs of 15 or more | 22 | 0 | 0 |
| Attack ticks before each break | 15, 15, 28, 41, 53 | - | - |
| Pitch at the end | -90 to 72 | -27 to 67 | 70-90 |
| Pitch actions -10 / -3 / 0 / 3 / 10 | 21 / 25 / 18 / 16 / 20% | about 20% each | 47 / 1 / 0 / 1 / 52% |

The trained actor turns to look down (70-90 degrees, the pitch where 14 of 19 data breaks happened) and then stops attacking. The imagined return is positive while the real return is zero: the imagined positive reward comes from rare imagined attack holds in rollouts and from the critic, not from what the trained policy does.

Video: frames were decoded from the MP4 files with AVFoundation into statistics (64 x 36, one frame per 0.5 s); no person watched them. The trained actor's frames show sky in 6% of the image (looking down) and change little (0.022 mean pixel change per 0.5 s, against 0.089 for random), consistent with looking at the ground without breaking it.

## Result and blocker

- The Actor did not learn to break a block. Random persistent actions break more.
- Not the blocker: the data (19 breaks were enough to learn a reward head 33 times chance after continued training), the reward head (it predicts the break and the reward rises with a held attack), the horizon (a held attack's predicted reward peaks within 15 steps).
- World-model failure was real for the frozen canonical model (it predicts the wrong ray change on a break) and was fixed enough by 1,000 updates of ordinary continued training on the new transitions, at a cost of 2 points of changed-ray accuracy on an unseen seed.
- **Blocker: actor action persistence.** A break needs 15 consecutive attack ticks. The actor samples attack independently each step, starts at 50%, and every imagined step of attack without a break costs a little predicted reward. With a 0.5 attack rate a given 15-tick hold has probability 0.5^15, about 1 in 33,000, so the gradient sees attack's small cost and not its delayed payoff, and the policy moved to never attacking.

The next decision is how the actor can hold an action long enough to discover delayed reward, with an exploration change the developer approves, without a scripted mining skill.
