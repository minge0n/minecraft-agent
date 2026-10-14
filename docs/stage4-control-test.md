# Stage 4: movement control test

Question: can the existing Dreamer actor learn a real physical control task in Minecraft when the reward depends only on the current transition, not on episode-wide novelty?

Answer from this run: **the test stopped before the actor.** The reward head on the frozen canonical RSSM fits the movement reward on training episodes but not on held-out episodes: held-out error 0.090 against 0.104 for predicting zero, and steps that move toward the target get a mean prediction of 0.007 against a real 0.115. The stop rule says no actor training when the reward predictor cannot learn this simpler signal, so no actor was trained.

This is a diagnostic test, not a progression reward. `reward-v1` is unchanged and still recorded on every step.

## The diagnostic reward `reward-control-v1`

`src/minecraft_rl/minecraft_reward_control.py`:

```text
target   = p_0 + 20 * facing(yaw_0)            (horizontal x, z)
d(p)     = |(p.x, p.z) - target|
reward_t = 1.0 * (d(p_t) - d(p_{t+1}))
```

- p_t is the server player position after step t (p_0 after RESET), yaw_0 the start yaw. In world seed 100000 with the fixed spawn, the start is x 192.5, z 112.5, yaw 0 (south), so the target is x 192.5, z 132.5. A probe of 200 forward-sprint steps from the spawn moved 55 blocks along +z without dying, so the target is reachable on foot.
- Moving toward the target is positive, away is negative, standing still or turning in place is zero.
- The sum over an episode is d(p_0) - d(p_T): any loop that returns to a place earns zero in total. No arrival bonus, no memory of earlier positions.
- "Reached" (minimum distance 1.5 blocks or less) is an evaluation metric only.

Positions and yaw come from the existing privileged `DEBUG_PLAYER` request. `minecraft_play` stores them as `privileged_position` and `privileged_yaw` arrays, one row per observation. No `obs_*` array and no model input contains them; `observation_tensors` reads `obs_*` arrays only. Tests check forward and backward progress, standing still, turning in place, loops, identical transitions, independence from earlier visits, and the separate storage.

## Data

The old episodes have no positions, so a small new dataset was collected: random exploration, 12 episodes of 500 steps, world seed 100000, fixed spawn, video (`runs/stage4/control-random`, game on the efficiency cores behind the chunk gate). Episodes 0-8 train, 9-11 are held out.

| | Episodes | Steps | Toward (> 0.01) | Away (< -0.01) | No progress | Turning 10 degrees or more, no progress |
|---|---|---|---|---|---|---|
| Training | 9 | 4,500 | 1,646 | 1,843 | 1,011 | 6 |
| Held out | 3 | 1,500 | 536 | 738 | 226 | 0 |

Random exploration: mean final distance 28.7 blocks (sd 10.8) from a start of 20, net progress -8.7 (sd 10.8), 2 of 12 episodes end closer than they started, none comes nearer than 9.7 blocks. A step moves 0.14 blocks on average. Fast turning is rare in this data (0.8% of steps).

## Reward prediction

The existing reward head (same network as before), on whole-episode posterior states of the frozen canonical RSSM. Movement rewards are dense and signed, so the head trains without positive weighting and without positive-centered chunks; scale 10 maps one sprint step (0.28 blocks) to about 3. 3,000 updates.

| | Training (4,500 steps) | Held out (1,500 steps) |
|---|---|---|
| Model MAE | 0.024 | 0.090 |
| Zero baseline MAE | 0.090 | 0.104 |
| Mean baseline MAE | 0.092 | 0.104 |
| Correlation with the real reward | 0.97 | 0.56 |
| Sign right on moving steps | 94% | 68% |
| Toward steps: predicted / real | 0.081 / 0.100 | 0.007 / 0.115 |
| Away steps: predicted / real | -0.129 / -0.129 | -0.083 / -0.128 |
| No-progress steps: predicted / real | -0.004 / 0.000 | -0.013 / 0.000 |

On held-out steps with no progress, the head predicts -0.004 when the camera turns by 3 degrees or more (83 steps) and -0.018 otherwise (143 steps): turning in place is not rewarded.

Held-out episode returns, predicted against real: -29.4 / -29.5, -7.8 / -10.4, -23.5 / +7.5. The one held-out episode that moved toward the target (+7.5) is predicted as a loss (-23.5).

Learning curve on the same held-out episodes:

| Training episodes | Held-out MAE | Zero baseline | Correlation | Toward steps predicted |
|---|---|---|---|---|
| 3 | 0.123 | 0.104 | 0.47 | -0.029 |
| 6 | 0.095 | 0.104 | 0.45 | 0.023 |
| 9 | 0.090 | 0.104 | 0.56 | 0.007 |

More episodes help a little, but the head does not separate "toward" from "away" on new episodes.

## Old spinning episodes

The first actor episodes (`eval-actor200`, `eval-actor1000`, `eval-actor-r3`, 12 episodes) have no positions, so their real control reward is unknown. Physically they turned in place most of the time. The head predicts -0.053 per step on them (-0.060 on turning steps, -0.019 on the others): negative, not the near-zero that standing still deserves, and not positive. So spinning would not be rewarded, but the head is not calibrated there either.

## Why the head fails: the state does not carry heading or distance

The reward needs the heading relative to the target and the distance to it. Ridge-regression probes from training to held-out states (1,503 states, R^2, 1 is perfect, 0 is no better than the mean):

| Input | sin(yaw) | cos(yaw) | Distance to target |
|---|---|---|---|
| h | 0.13 | 0.51 | 0.19 |
| z | 0.29 | 0.45 | 0.05 |
| e (current observation only) | 0.46 | 0.49 | 0.19 |

Absolute yaw and position are deliberately not in the observation (`docs/decisions/observation.md`). The agent can only infer them from what it sees and from the integrated camera actions. The recurrent state h knows less about the heading than the current view does (0.13 against 0.46 for sin), and almost nothing about the distance to a point 20 blocks away. The RSSM was trained to predict the next ray grid, not to integrate yaw deltas over 500 steps, and the frozen objective gives it no reason to.

## Result

- Implemented: `reward-control-v1`, privileged position recording in `minecraft_play`, control metrics per episode, a training and probe script (`src/minecraft_rl/minecraft_control_train.py`).
- Stop condition met: the reward head does not predict movement toward the target on held-out episodes. No actor was trained, so there are no actor returns, no imagined returns and no actor videos.
- The random-exploration videos of this dataset were recorded (`runs/stage4/control-random/video`) and not inspected; they are not needed for a decision that stops before the actor.

## The single blocker

The frozen RSSM state does not represent where the agent is or which way it faces relative to the start. The movement reward is a function of exactly that, so no reward head on this state can predict it. This is a world-model representation limit, not a reward-design or actor problem, and it also limits any future reward that depends on where the agent is.

The next decision belongs to the developer: either a reward that depends only on what changes in the current view (no hidden position), or a measured world-model change, which the current freeze forbids.
