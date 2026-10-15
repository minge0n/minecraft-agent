# Stage 4: temporal attack commands and lives without a time limit

Question: can the actor learn to break real blocks when it chooses how long to hold attack?

Answer: **yes, in this bounded test.** With the commands `RELEASE_ATTACK` and `HOLD_ATTACK_20`, 200 actor updates in imagination give an actor that looks down and holds attack. In real Minecraft it breaks 47 blocks in 4 lives of 1,000 ticks (11.75 per 1,000 ticks). Persistent random actions break 10 (2.5 per 1,000 ticks), the untrained temporal actor 19 (4.75), the previous per-tick actor 0.

The world model, the reward head and `reward-break-v1` are unchanged. They are the frozen artifacts of `docs/stage4-block-breaking.md` (`runs/stage4/break-continued`).

## Temporal attack commands

`src/minecraft_rl/minecraft_temporal_attack.py`:

- `RELEASE_ATTACK`: one tick with attack off.
- `HOLD_ATTACK_20`: 20 consecutive ticks with attack on.

`AttackController` stores the number of forced hold ticks that remain. The actor chooses a command only when none remain (a decision tick). During a hold the actor draws no attack sample. The pitch is chosen on every tick. Every other factor stays at its no-op value. Nothing looks at the observation to choose or stop a command: no aiming, no block detection, no mining progress, no release after a break.

`TemporalBreakAgent` is the existing actor and critic with one more input, the controller phase `remaining / 20`. The actor and the critic thus know whether the attack is forced. The normal action schema `player-action-v3` is unchanged. A command becomes physical per-tick actions before it reaches Minecraft or the world model.

## Real and imagined ticks are the same

`temporal_step` is the only function that turns a command into a physical action. Real play (`TemporalActorPolicy`) and imagination (`imagine_temporal`) both call it. Each tick is one physical action, one Minecraft tick and one RSSM transition, so a hold is 20 transitions, each with attack 1 in the existing action encoder. A test feeds the same start state and seed through both paths and gets identical inputs and actions for 30 ticks.

Imagined rollouts start from real posterior states. The training data was collected without the controller, so a start state after k consecutive attack ticks gets the controller state `(-k) mod 20`: it is read as a hold that started at a decision tick. 39% of the 4,824 start states are decision ticks. An imagination window never restarts the controller. A hold that is still running at the horizon is in the phase input of the critic's bootstrap value.

## Credit for delayed reward

The actor loss is the existing one (`losses`). Two things change through the agent:

- The command log probability and entropy count only on decision ticks. On a forced tick they are zero, so no gradient reaches the command head there. The pitch counts on every tick.
- The lambda return of the decision tick contains the rewards of the whole hold, discounted per tick (gamma 0.99, lambda 0.95). A break at tick 15 of a hold therefore reaches the hold decision with weight 0.99^15 times lambda terms. A test checks that a delayed reward pushes the command head toward `HOLD_ATTACK_20`.

The critic is still trained on every tick.

## Minecraft lives without a time limit

`src/minecraft_rl/minecraft_life.py` plays lives. A life starts with `RESET` (world seed 100000, fixed spawn, normal survival rules) and ends only when the player dies. The environment never ends a life after a number of ticks, and wall-clock time never ends a life. A death is never created.

| Term | Meaning | Ends the life |
|---|---|---|
| Episode termination | The player dies (`terminated = true`). The final objective will also end a life when it is supported. | Yes |
| External truncation | The run stops a life from outside: the operational `--tick-budget`, a manual interrupt or a failure. `truncated = true`, `terminated = false`, `stop_reason` records why. | No, the life is unfinished |
| Measurement checkpoint | Metrics of the first N ticks (`--checkpoints`, here 200 and 1,000) | No |
| Data segment | The life is written in parts of `--segment-steps` (500) transitions, so a long life is never held in memory | No |
| Video segment | One video file per data segment | No |
| Replay window | 32 consecutive transitions from one life | No |
| Training budget | 200 actor updates | No |

Inside one life nothing resets: the world, the player position, the inventory, health and food, the RSSM state of the policy, the attack controller, the reward memory of `reward-v1`, the life id and the step count. A new life resets all of them.

A truncated life has `terminated` false on every step. The continuation target `1 - terminated` is therefore 1 on every step, and the continuation model never learns a stop as a death. Exact resumption of a stopped life was not built. A truncated life stays labeled as truncated.

`join_segments` joins the stored segments of one life into one `Episode`. It refuses a gap in the step index. `SequenceReplay` samples windows from one `Episode` only, so no window crosses a real life boundary. The previous fixed-length collectors (`minecraft_play`, `minecraft_collect`) are unchanged and remain the record of the earlier short diagnostics.

## Imagined reward before training

From 200 in-reach start states at a decision tick (pitch 60 degrees or more, center ray 4.5 blocks or less), horizon 32, the frozen continued model and its head:

| Physical attack | Attack ticks | Predicted reward over 32 ticks | Share reaching a step of 0.3 |
|---|---|---|---|
| No attack | 0 | -0.10 | 3% |
| Short attack (5 ticks) | 5 | -0.09 | 9% |
| One 20-tick hold | 20 | +0.46 | 42% |
| Two 20-tick holds (32 ticks in the horizon) | 32 | +1.55 | 65% |

Sustained attack gives predicted reward and short attack does not. The horizon of 32 holds one full hold and the start of a second.

## Actor in imagination

`src/minecraft_rl/minecraft_temporal_train.py`, horizon 32, the existing `ActorConfig` otherwise (discount 0.99, lambda 0.95, entropy 0.003, Adam 3e-4, 256 start states per update), 200 updates, 2 min 18 s on the efficiency cores.

| | Untrained | Last 50 of 200 updates |
|---|---|---|
| Imagined return | 0.15 (first 10 updates) | 5.72 |
| Imagined reward per tick | -0.001 | +0.030 |
| Critic value | 0.003 | 6.47 |
| `HOLD_ATTACK_20` share of decisions | 0.49 | 0.98 |
| Command entropy (decision ticks, nats) | 0.69 | 0.08 |
| Pitch entropy (nats) | 1.61 | 0.27 |
| Attack share of imagined ticks | 0.95 | 1.00 |

## Real evaluation

Each policy: 4 lives, world seed 100000, fixed spawn, start pitch 0, normal survival, video, continued world model for the actors, game on the efficiency cores behind the chunk gate (`--background-game`). The operational budget was 1,000 ticks per life. **No life died, so all 16 lives are truncated, not completed lives.** All metrics are at equal tick exposure: the same 200 and 1,000 ticks of each life.

| | Per-tick actor (previous) | Random persistent actions | Untrained temporal actor | Trained temporal actor |
|---|---|---|---|---|
| Blocks broken at 200 ticks (4 lives) | 0 | 4 | 0 | 12 |
| Blocks broken at 1,000 ticks (4 lives) | 0 | 10 | 19 | 47 |
| Per life at 1,000 ticks | 0, 0, 0, 0 | 1, 2, 3, 4 | 4, 5, 6, 4 | 12, 11, 12, 12 |
| Blocks broken per 1,000 ticks | 0 | 2.5 | 4.75 | 11.75 |
| Real `reward-break-v1` return per 1,000 ticks | 0 | 2.5 | 4.75 | 11.75 |
| First break (tick) | none | 52-982 | 253-569 | 27-29 |
| Items held at 1,000 ticks (per life) | 0 | 1-4 | 4-5 | 2 |
| Item types acquired | 0 | 1 | 1 | 1 |
| Attack share of ticks | 0.2% | 66% | 95% | 99.8% |
| Commands at 1,000 ticks, hold / release | - | - | 191 / 202 | 200 / 8 |
| Mean critic value | 0.53 | - | 0.04 | 5.44 |
| Deaths, health at the end | 0, 20 | 0, 20 | 0, 20 | 0, 20 |

Attack runs (all lives):

| | Count | Median length | Longest | Runs of 15-19 / exactly 20 / 40 / over 40 |
|---|---|---|---|---|
| Per-tick actor | 7 | 1 | 1 | 0 / 0 / 0 / 0 |
| Random | 148 | 15 | 97 | 22 / 1 / 2 / 11 |
| Untrained temporal | 105 | 20 | 160 | 2 / 54 / 30 / 17 |
| Trained temporal | 11 | 338 | 760 | 0 / 0 / 0 / 11 |

The trained actor holds attack almost all the time: 200 of its 208 decisions choose hold again. Its pitch is 80 degrees or more on 99% of ticks (3,942 of 4,000), against 21% for random.

The break pattern is the same in all 4 lives. The center ray (straight down) shows it: the actor breaks the 2 sand blocks (block 39) under the player at ticks 27 and 52, and then breaks a column of a second block type (block 115) straight down, one block every 87 ticks (9-10 blocks per life). The player falls one block after each break. Every life ends holding 2 items: sand, and nothing from block 115. A block that takes 87 held ticks with an empty hand and drops nothing is consistent with stone, but the block name was not read from the registry, so this is a hypothesis. The reward head was trained on sand breaks only, and `reward-break-v1` pays every block the same.

The real discounted return (gamma 0.99) of the trained actor's real reward is 1.1 on average over the first 800 ticks of each life. The critic predicts about 5.4 and the imagined return in training was 5.7. Imagined reward per tick was +0.030, real reward per tick +0.012. The imagined return overestimates the real one by a factor of about 5, but the real breaks came, so the actor training was not extended.

## Video

Every life was recorded: 3 video files per life (500, 500 and 0 transitions; the empty last segment came from a segment split at the budget and was fixed afterwards, see below), all `complete`. The videos were not watched: the frames were decoded with AVFoundation into statistics (64 x 36 pixels, one frame per 0.5 s, first segment of life 0). No person or agent viewed the images.

| | Change per 0.5 s | Sky share of the image |
|---|---|---|
| Per-tick actor | 0.019 | 0.02 |
| Random | 0.062 | 0.55 |
| Untrained temporal | 0.073 | 0.69 |
| Trained temporal | 0.035 | 0.02 |

The trained actor's frames show almost no sky, which is consistent with looking down, and change more than the per-tick actor's, which is consistent with breaking and falling. This is consistent with the recorded data, not a direct observation.

## Checks

- 19 focused tests in `tests/test_minecraft_temporal_attack.py`: hold length, release, consecutive holds, pitch on every tick, controller reset only at a new life, no reset at a checkpoint, real and imagined equivalence, one physical action per RSSM transition, one tick per step (a skipped tick stops the life), credit for delayed reward, masking of command log probability and entropy, per-tick discounting, play beyond the old 200- and 1,000-step limits, death termination, external truncation, segment joining, replay windows inside one life, and video segments without a reset.
- `./scripts/check`: all tests pass.
- Runtime probes: `minecraft-tick-gate-probe.py`, `minecraft-observation-probe.py`, `minecraft-recording-probe.py` pass. No mod code changed.
- A 300-tick smoke life (`runs/stage4/temporal-smoke`, 3 segments of 100) after the segment fix: 3 non-empty segments, 3 complete videos, a joined life of 300 ticks with one tick per step, truncated by the budget.

Fix found during the run: a segment was closed right after the step that filled it, so a life stopped by the budget at a segment boundary got an empty last segment, and `join_segments` failed on its missing reward array. A segment now closes before the next step, so an empty segment no longer happens, and `join_segments` skips an empty segment from the earlier runs.

## Result and blocker

- The temporal actor learns when to hold attack and breaks real blocks: 11.75 per 1,000 ticks, against 2.5 for random persistent actions and 0 for the per-tick actor at equal tick exposure. 4 lives per policy, one training seed: this is one bounded result, not a seed study.
- A part of the gain comes from the command space alone: the untrained temporal actor already breaks 4.75 per 1,000 ticks. Training more than doubles it.
- Action persistence was the blocker of `docs/stage4-block-breaking.md`. With temporal commands it is no longer the blocker.
- **Remaining blocker:** the imagined return overestimates the real one by a factor of about 5. A likely cause: the training data contains sand breaks only (19 breaks, all sand), so the world model and the reward head have never seen the second block type, which really needs 87 held ticks per break. What the frozen model predicts for that block was not measured. The actor's behavior (look down, hold forever) works for both blocks, so the error does no harm here. It will do harm in a task where the block type matters. The next data must contain the blocks that the actor reaches.
- No life died in the evaluation, so death termination was tested only in the unit tests, not against a real death in Minecraft.
