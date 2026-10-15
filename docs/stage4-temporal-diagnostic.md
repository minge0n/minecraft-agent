# Stage 4: why the temporal attack actor overestimates its return

Question: the temporal attack actor (`docs/stage4-temporal-attack.md`) breaks real blocks, but its imagined return is far above its real return. Which part causes this: the reward head, the RSSM imagination or the critic?

Answer: **all three overestimate, and the critic contributes the most to the reported number.** Over the 32-tick imagination horizon, the RSSM is the larger error: it imagines sandstone mining as sand mining. The reward head on real states adds a smaller error. The critic then learns from these inflated imagined rewards and adds most of the gap through its bootstrap value. One bounded correction (1,000 RSSM updates and a new reward head) fixed the sandstone dynamics but made the reward head much worse. The corrected pair is not suitable for training and is not promoted.

Nothing here retrains the actor, changes `reward-break-v1`, the horizon or the controller. Code: `src/minecraft_rl/minecraft_temporal_diagnostic.py` (measurements, `runs/stage4/temporal-diagnostic/diagnostic.json`) and `src/minecraft_rl/minecraft_temporal_correction.py` (the correction, `runs/stage4/temporal-correction/correction.json`).

## Block 39 and block 115

The registry of the experiment's environment (`DEBUG_REGISTRY`, `runs/stage4/temporal-diagnostic/registry.json`) and the Minecraft 26.3 game data:

| Block id | Name | Hardness | Correct tool needed for a drop | Loot |
|---|---|---|---|---|
| 39 | `minecraft:sand` | 0.5 | no | the sand item (item id 88) |
| 115 | `minecraft:sandstone` | 0.8 | yes, a pickaxe (`requiresCorrectToolForDrops`, tag `mineable/pickaxe`) | the sandstone item, only with the correct tool |

Block 115 is sandstone, not stone. Vanilla mining progress per tick is speed / hardness / 30 with the correct tool and / 100 without it. With an empty hand (speed 1), sand needs 1 / (1 / 0.5 / 30) = 15 ticks. Sandstone needs 1 / (1 / 0.8 / 100) = 80 ticks, and it drops nothing.

Measured on the recorded lives of the trained actor (center ray on the same block, attack held):

| | Breaks (4 lives) | Uninterrupted mining ticks at the break | Ticks between consecutive breaks | Inventory change 15 ticks after the break |
|---|---|---|---|---|
| Sand | 8 | 19-21 | 25-29 | +1 |
| Sandstone | 39 | 81-83 | 87 (some 101-153) | 0 |

The mining time and the break interval differ. A sandstone break needs 81-83 continuous ticks on the block, near the vanilla 80. The other 4-6 ticks of the 87-tick interval are the fall onto the next block, before the center ray settles on it. The sand mining count of 19-21 includes the first hold that starts before the player looks straight down. The missing sandstone drop is vanilla behavior: an empty hand is not the correct tool.

The untrained temporal actor broke 19 sand blocks and no sandstone. In every recorded life all attack runs are whole 20-tick holds, except the run cut by the tick budget. So the controller phase of each tick is rebuilt exactly from the recorded attack runs.

## Method

All comparisons use the 4 recorded lives of the trained actor (`runs/stage4/temporal-eval/trained`, 4,000 ticks), the frozen continued world model, its reward head and the trained actor and critic. The start states are every second tick, 1,936 in total, with at least 32 recorded ticks after them. Each start is the real posterior state. The imagination uses the recorded physical actions of the next 32 ticks, 4 latent samples per start. Starts are grouped by the block under the center ray at the start.

- Real: the recorded breaks (`reward-break-v1`).
- Posterior: the reward head on the real posterior states of the same 32 ticks.
- Imagined: the reward head on the RSSM prior rollout with the same actions.

The difference between posterior and real is the reward-head error. The difference between imagined and posterior is the RSSM error.

## Reward head on real posterior states

| | Sand | Sandstone |
|---|---|---|
| Real breaks | 8 | 39 |
| Predicted reward on the break tick, mean | 0.89 | 0.019 |
| Largest prediction within 2 ticks of the break, mean | 1.06 | 0.029 |
| Predicted reward on non-break ticks, mean | 0.063 | 0.019 |
| Non-break ticks | 172 | 3,582 |

The head detects sand breaks and misses sandstone breaks. It gives sandstone mining a flat 0.017 per tick whatever the mining time (1-14, 15-29, 30-79 and 80+ ticks: 0.017, 0.017, 0.017, 0.016). Over a whole life it predicts 20.0-22.0 reward for 11-12 real breaks.

## Imagination by horizon

Reward sums over the first k ticks, mean over starts:

| Start on | k | Real | Posterior head | Imagined | Center ray class correct (imagined) | Center ray distance error (blocks) |
|---|---|---|---|---|---|---|
| Sand (103 starts) | 1 | 0.039 | 0.083 | 0.027 | 97% | 0.48 |
| | 5 | 0.194 | 0.452 | 0.198 | 89% | 0.50 |
| | 10 | 0.388 | 0.924 | 0.533 | 81% | 0.53 |
| | 20 | 0.777 | 1.829 | 1.425 | 61% | 0.59 |
| | 32 | 1.136 | 2.576 | 2.547 | 38% | 0.67 |
| Sandstone (1,826 starts) | 1 | 0.010 | 0.017 | 0.007 | 8% | 1.38 |
| | 5 | 0.052 | 0.085 | 0.043 | 6% | 1.19 |
| | 10 | 0.104 | 0.171 | 0.115 | 5% | 1.04 |
| | 20 | 0.209 | 0.344 | 0.531 | 3% | 0.83 |
| | 32 | 0.337 | 0.557 | 1.623 | 1% | 0.91 |
| All (1,936 starts) | 32 | 0.381 | 0.665 | 1.671 | 3% | 0.90 |

The RSSM does not represent sandstone. Its posterior decodes the center ray on sandstone mostly as `minecraft:grass_block` (768 of 948 ticks in life 0) and sometimes as sand. In imagination it almost never predicts the sandstone class. After 20 imagined ticks of held attack on sandstone, the imagined reward rises the way a sand break does: from 0.53 at k = 20 to 1.62 at k = 32, against a real rise from 0.21 to 0.34. The training data of the world model had 2,028,926 sand ray hits and 20,475 sandstone ray hits, and no sandstone break.

Horizon limit: a sandstone break needs 81-83 continuous ticks, more than the horizon of 32. A rollout that starts at a decision tick can never contain a whole sandstone break. It contains one only if the start state is already 49 or more ticks into the mining. The horizon was not changed.

## Returns from matching states

The same 1,936 starts, the same 32 ticks, discount 0.99 per tick:

| | Real | Posterior head | Imagined |
|---|---|---|---|
| Reward sum over 32 ticks | 0.381 | 0.665 | 1.671 |
| Discounted reward over 32 ticks | 0.327 | 0.573 | 1.350 |
| Critic bootstrap at tick 32 | 5.41 (on the real state) | - | 6.78 (on the imagined state) |
| Discounted reward + 0.99^32 x bootstrap | 4.25 | - | 6.26 |

The actor's own imagined lambda return (lambda 0.95, its own actions, from the same starts) is 5.80. The critic value on the real states is 5.65. The real discounted return of the rest of the life from the same states is 1.15 (starts with at least 460 recorded ticks after them, 1,084 states).

Error split, for the discounted reward of the 32-tick horizon (1.350 imagined against 0.327 real, gap 1.02):

- Reward head on real states: 0.573 - 0.327 = 0.25 (24%).
- RSSM imagination: 1.350 - 0.573 = 0.78 (76%), almost all from sandstone starts (1.31 against 0.48).

The critic is a separate and larger error: 5.65 on real states against 1.15 real. Its bootstrap term, 0.725 x 6.78 = 4.9, is most of the imagined lambda return. The critic learns from imagined returns, so the inflated imagined rewards also inflate its targets. A high critic value is not a predicted break: on real sandstone states it is 5.3, while the real head predicts 0.019 per tick.

## Main cause

The reported overestimate (5.8 imagined against 1.15 real) comes mostly from the critic value. The critic's error is learned from the reward-and-dynamics error within the horizon, and that error is mostly the RSSM imagining sandstone mining as sand mining. Sandstone is almost absent from the world model's data, needs 80 mining ticks with no drop, and is the block that the trained actor mines 39 of its 47 breaks on.

## Correction: one bounded RSSM continuation and one new reward head

The RSSM error was the largest within-horizon error, so the plan allowed one continuation of the RSSM (`minecraft_temporal_correction`):

- RSSM: the continued block-breaking model, 1,000 more updates, the same architecture and canonical objective (G, pitch 128), Adam 1e-3, clip 1000. Each batch has 8 windows from the training lives and 8 from the Stage 3 training data. Training lives are lives 0 and 1 of the trained and of the untrained temporal actor. Lives 2 and 3 of both are held out. 21 minutes on the efficiency cores.
- Reward head: a new head on the new latent states with the existing `reward-break-v1` loss (positive weight 20, half the chunks centered on breaks, 3,000 updates). The data are the 4 training lives and the 24 random block-breaking training episodes: 28 episodes with 49 breaks.

Held-out lives 2 and 3 of the trained actor (968 starts):

| | Original | Corrected |
|---|---|---|
| Imagined center ray class on sandstone, k = 32 | 1% | 100% |
| Imagined center ray distance error on sandstone, k = 32 (blocks) | 0.89 | 0.35 |
| Head on real states, sand break tick | 0.88 | 0.67 |
| Head on real states, sandstone break tick | 0.017 | 0.28 |
| False reward per non-break tick, sand / sandstone | 0.047 / 0.019 | 0.26 / 0.18 |
| Predicted reward per life (real 12) | 20.4, 20.2 | 188, 166 |
| Discounted reward over 32 ticks: real / posterior / imagined | 0.34 / 0.56 / 1.35 | 0.34 / 4.91 / 4.41 |

Stage 3 evaluation on the same windows (128 per split):

| | eval_episode original | corrected | eval_seed original | corrected |
|---|---|---|---|---|
| Ray accuracy | 0.765 | 0.775 | 0.600 | 0.625 |
| Changed-ray accuracy | 0.361 | 0.362 | 0.260 | 0.281 |
| Ray distance error (blocks) | 3.21 | 2.80 | 3.38 | 3.38 |
| Pitch error (degrees) | 3.49 | 3.03 | 4.36 | 3.85 |
| KL mean / active variables / classes used | 9.08 / 16 / 170 | 9.31 / 16 / 170 | 11.39 / 16 / 185 | 11.79 / 16 / 191 |
| Changed rays, imagination k = 10 | 0.296 | 0.265 | 0.207 | 0.211 |
| Changed rays, imagination k = 20 | 0.237 | 0.209 | 0.157 | 0.179 |

The RSSM part of the correction worked: it now imagines sandstone, and Stage 3 one-step metrics do not regress. Imagination on `eval_episode` regresses by 3 points of changed-ray accuracy at k = 10 and k = 20, and `eval_seed` improves.

The new reward head failed: it predicts about 15 times the real reward on held-out lives. A sandstone break follows 80 mining ticks that look the same in the observation. The observation shows no mining progress, and the rare-event loss gives each positive 20 times the weight. The head therefore spreads the break over the whole mining run: 0.18 per tick, about 20 / 80 of a break. The break is only predictable from a count of mining ticks in the recurrent state, and the new latent does not make that count readable for the head.

Decision: the corrected checkpoint and head stay experimental artifacts in `runs/stage4/temporal-correction`. They are not promoted and not used for training. The original continued model and head stay unchanged. No further correction was tried.

## Next step toward items

Breaking a block, producing a dropped item, collecting it and getting a useful resource are different events. Sand gives all four by hand. Sandstone gives only the first: without a pickaxe it drops nothing. A reward for breaks therefore pays for work that produces nothing. A mining-progress count that the observation does not show is also hard to predict.

Recommendation: the next diagnostic reward pays for an item that enters the inventory, which the HUD-like self state shows, rather than for a break. Before that, the compact vocabulary must contain the items that the agent can collect. The current vocabulary has no index for the sand item (88), so the world model sees collected sand as an unknown item.

## Checks

- `tests/test_minecraft_temporal_diagnostic.py`: the rebuilt controller phase, the mining-tick count and the per-tick discount.
- `./scripts/check` passes. No Minecraft code changed for the diagnostic. The registry was read once from the experiment's environment.
