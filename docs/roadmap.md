# Learning roadmap: Dreamer first

Status: **decided direction; Stage 1, Stage 1.5, Stage 2A, Stage 2B, Stage 2C, Stage 2D, Stage 2E, Stage 2F, Stage 2G and Stage 2H are done; the toy Dreamer-style agent works with a deterministic GRU world model and with an RSSM (stochastic latent, KL), and the RSSM represents an uncertain future that the GRU cannot. Next is the first Minecraft RSSM world-model smoke test.** Supersedes the earlier "Q-learning, then DQN" path. DQN may appear later as an optional baseline but is not on the critical path.

```text
Stage 1   tabular Q-learning baseline (done, docs/stage1.md)
Stage 1.5 Minecraft lockstep + structured observation contract (done, docs/minecraft-spike.md)
Stage 2A  neural-network sanity check (done, docs/stage2a.md)
Stage 2B  recurrent model sanity check (done, docs/stage2b.md)
Stage 2C  learned dynamics in a toy environment (done, docs/stage2c.md)
Stage 2D  imagination and compounding-error measurement (done, docs/stage2d.md)
Stage 2E  actor-critic trained in imagination (done, docs/stage2e.md)
Stage 2F  integrated Dreamer-style toy agent (done, docs/stage2f.md)
Stage 2G  RSSM world model (stochastic latent, KL) in the same toy loop (done, docs/stage2g.md)
Stage 2H  stochastic signal T-maze: GRU against RSSM (done, docs/stage2h.md)
Next      Minecraft RSSM world-model smoke test on real structured transitions
Later     Dreamer on the structured Minecraft observation
```

Stage 2A starts only after the Stage 1.5 gate in `docs/minecraft-spike.md` is validated. Minecraft training starts only after Stage 2F works and the Minecraft environment track below reaches its step 8 for the scale being trained. Running neural code is not a reason to start long Minecraft training.

## Minecraft environment track

```text
1 one-tick lockstep                     proven
2 structured current-visible observation proven (scripted scene)
3 factorized actions                    proven
4 reset / episode semantics             proven (fresh world, death termination)
5 determinism characterization          measured (docs/replay-characterization.md)
6 single-instance throughput            measured (docs/decisions/simulation-throughput.md)
7 accelerated mode validated vs paced   proven (controlled replay equivalence)
8 multi-instance workers                isolation proven for 2 workers; scaling open (docs/decisions/parallel-workers.md)
9 connect to Dreamer training           after Stage 2F
```

Correctness precedes throughput. Session recording (`docs/decisions/recording.md`) is required for every meaningful session from the first training, evaluation or recorded debug run; its cost is benchmarked together with steps 6-8.

## Target architecture

```text
structured current observation -> observation encoder -> recurrent latent state (RSSM)
    learned world model: next latent, reward, continuation
    imagined trajectories -> actor + critic -> factorized Minecraft action
```

The world model is learned from real transitions only; no hard-coded Minecraft physics, recipes or strategy. The loop alternates: collect real experience, improve the world model, imagine trajectories, improve actor and critic, collect again. It must not memorize and replay successful trajectories. Knowledge lives only in learned parameters, embeddings, recurrent state and dynamics.

## Implementation policy

PyTorch, random initialization, no high-level RL or Dreamer package hiding the algorithm. The DreamerV3 paper and code are technical references and comparison targets. Code should make explicit: observation encoder, replay sequences, RSSM with deterministic recurrent state and stochastic latent state, observation, reward and continuation prediction, KL regularization, imagined rollout, actor, critic and return/value targets. Implement incrementally in small validated commits. Each new component is explained to the developer in Korean (problem, inputs/outputs and tensor shapes, parameters, loss, gradients, Minecraft relevance).

## Stages

- **2A neural sanity check:** add PyTorch; a tiny network learns a deterministic supervised mapping. Verifies the local ML environment, gradients, checkpoint save/load, CPU/GPU device selection, and inspectable tensor and optimizer behavior. Not DQN.
- **2B recurrent model:** a small recurrent model on a toy sequence task, proving that an earlier observation changes a later prediction.
- **2C learned dynamics:** in a tiny environment, learn `history + action -> next latent/observation, reward`. Tests and plots show prediction error improving.
- **2D imagination:** roll the learned model forward without the real environment, compare imagined and real rollouts from the same starting points at horizons 1, 5, 10, 20, and measure compounding error. A falling training loss is not proof of a correct model.
- **2E actor-critic in imagination:** train actor and critic on imagined trajectories, evaluate only in the real toy environment, and show improvement above random.
- **2F integrated toy agent:** one understandable Dreamer-style loop combining the pieces.

### Toy environment

A small T-maze cue-memory task (`src/minecraft_rl/tmaze.py`, specified in `docs/stage2c.md`): a cue is visible at the start, disappears, and the correct turn at a later junction requires remembering it. It exercises recurrent state, partial observability, world-model learning, actor-critic learning and imagination cheaply. An equally small alternative is acceptable only with a clear reason.

## Model-error evaluation (first-class)

For real starting points, compare real and imagined rollouts at increasing horizons and track observation, reward and continuation prediction error, plus task-relevant state error where measurable. Flag policies with high imagined return but poor real return. Respond to model exploitation by improving data coverage, uncertainty handling, model design or imagination horizon, never by hard-coding task solutions.

## Minecraft integration (later)

The encoder learns categorical block, entity and item embeddings from scratch, without pretrained or text semantics. The environment never keeps supplying objects that are no longer visible; the recurrent state must retain them if useful.

## Reward direction

Extrinsic reward comes from advancements as a scalar only; the policy never sees advancement names, descriptions or recipe-bearing metadata. One-time rewards are deduplicated per episode. Start with a simple baseline and measure it. The later `reward_i = base_reward_i * mastery_multiplier_i` experiment makes the reward non-stationary and must be tested, not assumed beneficial. No handcrafted skill hierarchy or planner; if hierarchy becomes necessary, prefer learned options or latent goals justified experimentally.
