# Stage 4: pitch-control diagnostic

Question: can the existing Minecraft Dreamer loop (frozen canonical RSSM, reward head, actor-critic in imagination) learn one action-conditioned control task when the reward depends only on a value that the policy observes?

Answer: **yes, for this task.** After 200 actor updates the real camera reaches the target pitch within 3 steps in every evaluation episode and stays near it: mean pitch error 7.1 degrees against 57.8 for random pitch actions and 28.7 for the untrained actor. The imagined reward per step (-0.104) is close to the real one (-0.079), so this is not reward exploitation.

This is a basic control test. It does not show navigation, mining, crafting or progression. `reward-v1` and `reward-control-v1` are unchanged.

## Reward and action space

`src/minecraft_rl/minecraft_reward_pitch.py`:

```text
reward-pitch-v1:  reward[t] = -|pitch(o_{t+1}) - 30| / 90
```

The reward of transition t reads only the next observation, the pitch field of the HUD-like self state. Best value 0 at +30 degrees (looking down), -1.33 at -90 degrees. No position, yaw, event or memory.

Restricted action space, for this diagnostic only: `PitchOnlyAgent` is the existing actor and critic network. It samples only the pitch factor (-10, -3, 0, 3, 10 degrees per step); every other factor stays at its no-op value, so the player does not move, turn, press a button or change the slot. Only the pitch head gets a gradient. The camera action semantics and the production action schema are unchanged.

## Data

Existing episodes are relabeled from their recorded `obs_pitch`; no new data was collected for the reward head, and no original reward array was changed. Split by whole episode: the last 3 episodes of each random dataset and the last episode of each actor dataset are held out.

| | Episodes | Transitions |
|---|---|---|
| Training (random 23, actor 14 episodes) | 37 | 32,500 |
| Held out random (`random` 13-15, `control-random` 9-11) | 6 | 4,500 |
| Held out actor (episode 3 of 5 actor runs) | 5 | 5,000 |

Pitch coverage over all 42,000 transitions: 2,875 next observations in 25-35 degrees, 2,320 in 20-25, 1,876 in 35-60. The target and its neighborhood are covered, mostly by the earlier actor runs.

## Reward prediction

The existing reward head on whole-episode states of the frozen canonical RSSM, 3,000 updates, targets times 10, no positive weighting (the reward is dense). The exact reward from the observed pitch is the reference and has error 0.

| | Held out random (4,500) | Held out actor (5,000) |
|---|---|---|
| Model MAE | 0.039 | 0.036 |
| Zero baseline MAE | 0.331 | 0.272 |
| Mean baseline MAE | 0.124 | 0.153 |
| Correlation | 0.93 | 0.96 |
| Predicted / real per step | -0.329 / -0.332 | -0.265 / -0.272 |

By the pitch of the next observation, predicted against real mean reward:

| Pitch | Random held out | Actor held out |
|---|---|---|
| -90 to -10 | -0.50 / -0.52 (1,078) | -0.62 / -0.63 (350) |
| -10 to 10 | -0.33 / -0.34 (2,316) | -0.34 / -0.34 (1,745) |
| 10 to 25 | -0.17 / -0.15 (941) | -0.16 / -0.15 (1,380) |
| 25 to 35 | -0.10 / -0.03 (148) | -0.07 / -0.03 (576) |
| 35 to 50 | -0.13 / -0.07 (17) | -0.09 / -0.11 (349) |
| 50 to 90 | - | -0.35 / -0.43 (520) |

The prediction rises toward the target from both sides and peaks in the 25-35 range, so closer states get a higher reward. Near the target the head is too low by about 0.05: it smooths the peak.

Spurious reward at a fixed pitch: within 2-degree pitch bins, the prediction spreads by 0.04-0.05 (median standard deviation), against 0.006 for the true reward. So yaw, movement and view change the prediction a little, about a tenth of the reward range between the far and the near pitch ranges.

## Actor

The existing actor-critic in imagination (horizon 15, lambda returns, REINFORCE with a critic baseline, entropy 0.003), 200 updates, 32,537 start states, 50 s. Imagined reward per step rose from -0.33 to -0.10; the imagined lambda return fell from -3.2 to -9.3 because the critic's value estimate of a 15-step negative return grew with training, not because the policy got worse. Pitch-factor entropy fell from 1.61 to 0.14 nats.

## Real evaluation

Four episodes of 100 steps each, world seed 100000, fixed spawn, start pitch 0 (30 degrees from the target), video, game on the efficiency cores behind the chunk gate.

| | Random pitch actions | Untrained actor | Actor, 200 updates |
|---|---|---|---|
| Real return per episode | -64.2 | -31.9 | -7.9 |
| Real reward per step | -0.642 | -0.319 | -0.079 |
| Reward head on the real states, per step | -0.599 | -0.297 | -0.118 |
| Final pitch error (degrees) | 63.2 | 36.0 | 7.5 |
| Mean pitch error | 57.8 | 28.7 | 7.1 |
| Steps within 5 degrees | 0% | 10% | 40% |
| Pitch actions -10 / -3 / 0 / 3 / 10 | 21 / 18 / 22 / 21 / 17% | 15 / 22 / 23 / 22 / 18% | 48 / 0 / 0 / 0 / 52% |
| Other buttons or yaw | none | none | none |

Per-episode final errors of the trained actor: 10, 10, 0, 10 degrees; mean errors 6.4-8.2. The untrained actor already does better than uniform random because its near-uniform logits drift; one of its episodes ended at 90 degrees.

Behavior from the action log, by the pitch before the action:

| Pitch before | 0-20 | 30 | 40 | 50 |
|---|---|---|---|---|
| Action +10 / -10 | 15 / 0 | 154 / 3 | 38 / 151 | 0 / 38 |

The actor turns down by 10 degrees per step until it reaches 30, then alternates between 30 and 40: at 30 it almost always turns down, at 40 mostly up. The choice depends on the observed pitch, so it is closed-loop control, not a fixed action sequence. It never uses the small steps (-3, 0, +3) that would hold 30 exactly; the coarse ±10 oscillation costs about 5 degrees of mean error.

Video: frames were decoded from the MP4 files with AVFoundation into image statistics (64 x 36, one frame per 0.5 s); no person watched them. The trained actor's frames show sky in 5-16% of the image (looking down) against 67-90% for random pitch actions, and change less between frames (0.035 against 0.07).

## Result

- The reward head predicts the observable pitch reward on held-out random and held-out actor episodes, and on the new evaluation states (error 0.05-0.08 per step).
- The trained actor moves the real camera to the target and keeps it within 10 degrees, far better than random pitch actions and the untrained actor.
- Imagined and real reward agree within 0.04 per step. No exploit was found.

The existing Dreamer actor loop can learn a simple action-conditioned control task in Minecraft when the reward is a function of the current observation.

## The next blocker

This task gives the actor a reward that the frozen world model already represents well (the pitch, which Stage 3 weighted 128 times). The earlier failures had rewards that the state does not represent: episode novelty and position. The next blocker is a reward for real progress, such as mining or acquiring an item, that the frozen state represents as well as it represents the pitch.
