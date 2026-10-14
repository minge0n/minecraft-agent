# Stage 4: the first Minecraft actor smoke test

Question: can an actor learn Minecraft behavior that earns more real reward than random exploration?

Answer from this run: **not yet.** The full Dreamer loop runs in Minecraft, but no trained actor earned more real reward than the random exploration policy. The single largest blocker is the reward predictor: it does not tell rewarded states from unrewarded ones on new data, so the actor optimizes imagined reward that the real game does not pay.

## Setup

- World: `normal`, seed 100000 (the first Stage 3 training world), fixed spawn (`RESET` with `fixed_spawn`: the top of the world spawn column, facing south). 1,000 steps per episode, no death in any episode.
- World model: the frozen canonical Stage 3 RSSM (`runs/stage3/tradeoff-P128/seed0/world_model.pt`, objective G with pitch weight 128, 3,000 updates). `load_world_model` refuses any other config or objective. It never changes in this stage.
- Reward: `reward-v1` (`docs/decisions/reward.md`), computed live by `EpisodeReward` and stored per step.
- Data: `minecraft-episodes-v2` (`src/minecraft_rl/minecraft_dataset.py`) adds the scalar reward, the components, `events.jsonl`, and the action schema, reward version, policy and return in the metadata. v1 episodes still load.
- Actions: the physical factors of `player-action-v3` as 9 categorical factors (`ActionSpace`): movement 9, jump, sneak, sprint, attack and use 2 each, yaw 7 (-30 to 30 degrees), pitch 5 (-10 to 10 degrees), hotbar 10. No abstract action: the frozen world model has no input for it.
- Reward head: an MLP r(s_t) on s = [h, z] of the frozen RSSM, trained on posterior states. Targets are multiplied by 100, and nonzero steps weigh 20 times more, because only 0.5% of steps have a reward.
- Actor-critic (`src/minecraft_rl/minecraft_actor.py`): one actor head per factor and a critic on s; imagination horizon 15 from posterior states of real steps; lambda returns (discount 0.99, lambda 0.95); REINFORCE with a critic baseline, advantages divided by their mean absolute value, entropy weight 0.003; Adam 3e-4. `actor_critic.lambda_returns` is reused.
- Acting: `Filter` runs one RSSM posterior step per real step.
- Commands: `python -m minecraft_rl.minecraft_play` (collect or evaluate, with video) and `python -m minecraft_rl.minecraft_actor_train` (reward head, then actor checkpoints at 0, 200 and 1,000 updates).

## Data and reward

Random exploration, 16 episodes of 1,000 steps (`runs/stage4/random`):

| | Per episode |
|---|---|
| Return | 0.171 (sd 0.088, range 0.06-0.33) |
| Block types seen | 8.0 |
| Block types broken | 1.06 |
| Item types acquired | 0.31 |
| Important items | 0.06 (one log in 16 episodes) |
| Advancements | 0 |
| Deaths | 0 |

87 of 16,000 steps have a reward. Every episode earns its seen novelty at step 0, from the start view. All rewards are finite, the components sum to the scalar, events match the rewards, and novelty restarts each episode.

## Reward prediction

Held-out episodes 13-15 (13 rewarded steps), mean absolute error:

| Steps | Reward head | Zero baseline | Mean baseline |
|---|---|---|---|
| Rewarded (training episodes, 74) | 0.021 | 0.032 | 0.031 |
| Rewarded (held out, 13) | 0.030 | 0.030 | 0.030 |
| Unrewarded (held out) | 0.0018 | 0 | 0.0001 |

The head fits the rewarded training steps, but on held-out episodes it predicts almost nothing for them (mean 0.0002 against 0.030). It ranks a held-out rewarded step above an unrewarded one in 74% of pairs. A second head trained with the actor's own 12 episodes added (127 rewarded steps) ranks 59% on the same held-out steps.

The likely cause is the novelty itself: the reward of a step depends on whether the block or item type was new in the episode, and that history is not in the state that the head reads. The posterior state shows the block that is seen or broken now, but not whether it was seen before. So rewarded and unrewarded states with the same view look alike, and 87 positive examples are too few to learn the difference.

## Actor results

Real Minecraft, 4 episodes of 1,000 steps each, same world and spawn. Imagined return is the mean lambda return over the last 50 actor updates.

| Policy | Real return | Seen | Broken | Acquired | Imagined return | Entropy (sum of 9 factors, nats) |
|---|---|---|---|---|---|---|
| Random exploration (16 episodes) | 0.171 (sd 0.088) | 8.0 | 1.06 | 0.31 | - | - |
| Untrained actor | 0.138 (sd 0.019) | 7.5 | 1.25 | 0 | 0.0 | 11.5 |
| Actor, 200 updates | 0.080 (sd 0.000) | 8.0 | 0 | 0 | 1.24 | 6.56 |
| Actor, 1,000 updates | 0.145 (sd 0.130) | 7.0 | 0.5 | 0.25 | 4.55 | 1.38 |
| Actor after one on-policy round, 1,000 updates | 0.200 (sd 0.062) | 7.5 | 1.5 | 0.25 | 3.99 | 1.73 |

Imagined against real reward per step: the 1,000-update actor's imagined reward is 0.041 per step; the reward head predicts 0.033 per step on the real states of its own episodes; the real reward is 0.00015 per step. The imagined return rises from 0 to 4.6 while the real return stays at the random level. This is model exploitation through the reward head, the failure that Stage 2E already flagged.

## Behavior

From the action logs, the positions and the recorded videos (frame-difference statistics, not watched by a person):

- **Untrained actor:** uniform factors, so each button is pressed half the time, with fast camera jitter (12 degrees of yaw per step against 3 for random exploration) and the view tilted down (mean pitch 37 degrees).
- **200 updates:** collapsed onto one pattern: jump and use 99% of the time, never attack, forward and right, and a steady turn of about 10,000 degrees per episode (spinning). It never breaks a block, so its return is the seen novelty of the start view (0.08 in all 4 episodes).
- **1,000 updates:** a steady turn the other way (about 13,500 degrees per episode, 20 degrees per step), attack half the time, use 82%, no jump. One episode broke two blocks and picked up a log (0.37); the other three got only the start view.
- **After the on-policy round:** forward-right 88% of the time, attack 95%, turning. Return 0.13-0.30 per episode, mean 0.20 against 0.17 for random exploration. With 4 episodes and a standard deviation of 0.06-0.09, this difference is not reliable evidence.

The spinning is the reward-hacking pattern that `docs/decisions/reward.md` predicted for seen novelty. Here the reward head drives it, not the real reward: about half of the seen novelty comes at step 0 (4-6 block types), the rest from 2-5 types later in the episode, and the spinning actors saw no more types than random exploration (8.0, 7.0 and 7.5 against 8.0).

## On-policy correction

The 12 actor episodes were added to the training data, a new reward head and a new actor were trained (`runs/stage4/actor-r2`), and the new actor was evaluated. Its real return rose from 0.145 to 0.200 and its behavior became more useful (attacking while moving instead of using), but it is still within the noise of random exploration, and the reward head's held-out ranking did not improve (0.74 to 0.59).

## Result

- The full loop works in Minecraft: data with reward, a reward head on the frozen RSSM, actor-critic in imagination, a one-step posterior filter, real evaluation with video, one on-policy correction. Every step was one tick; no death and no invalid action occurred.
- No actor clearly beats random exploration in real reward.
- Stop condition met: imagined return rose strongly while real return did not, and the 200-update actor collapsed onto one action pattern. Training was not extended.

## The blocker and the smallest fix

The reward predictor cannot predict novelty reward from the current state, because novelty depends on episode history that the state does not carry, and the positive examples are too few (87 in 16,000 steps).

The smallest fix is not a new memory. Options in order of size:

1. Learn the reward with the world model instead of on top of a frozen one (reward head trained jointly), so h can learn to keep the history that predicts reward. This changes the frozen world model, so it needs a decision.
2. Collect more rewarded transitions in the fixed world, for example longer episodes, so that the head sees more than 87 positive examples.
3. A conservative actor objective: penalize imagined reward that the head is unsure about (for example the spread of an ensemble of reward heads).

Do not extend actor training before the reward head ranks held-out rewarded steps clearly above unrewarded ones.

## Heat

The game process uses about 11 W and the trainer about 0.4 W (`docs/decisions/simulation-throughput.md`). For a long run, the documented options are the background policy for the game (1.6 W, but chunk loading falls behind after a reset) and a smaller game window. Neither blocked this smoke test.
