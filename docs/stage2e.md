# Stage 2E: actor-critic trained in imagination

**Purpose.** Train a policy without touching the real environment during policy learning: an actor and a critic learn only from trajectories that a frozen, learned world model imagines, and are then evaluated in the real T-maze. This is Dreamer's behavior-learning half. It also tests the Stage 2D warning: a policy optimized against a model can find what the model gets wrong.

## Pipeline

1. Collect 512 uniform-random-policy episodes and train the Stage 2C world model on them (seed `s`, 5,000 steps, `docs/stage2c.md`). Freeze it (`requires_grad_(False)`; a test checks its parameters do not change).
2. Every valid real step of those episodes becomes a start state (8,709 for seed 0).
3. Train actor and critic for 1,000 updates, each on 256 start states imagined 15 steps ahead.
4. Evaluate in the real T-maze (`L = 4`, 40-step limit, 256 episodes alternating cues) with greedy and sampled actions, against a uniform random policy.

## Policy state and imagination

The policy acts on `s_t = [h_{t-1}, onehot(o_t)]` (37 values), where `h_{t-1}` (32) is the world model's GRU state after the history before step `t` and `o_t` is the current observation. In the real maze `h` is built from real observations; in imagination from the model's own predictions. In both cases the actor, not the environment, chooses `a_t`:

```text
a_t ~ pi(. | s_t)                                  actor: Linear(37,64) tanh Linear(64,3), softmax
h_t = GRU(h_{t-1}, [onehot(o_t), onehot(a_t)])     frozen world model
r_hat_t, c_hat_t = reward head(h_t), sigmoid(continuation head(h_t))
o_t+1 = argmax of the next-observation head(h_t)   fed back as the next input
```

Imagination is run without gradients; the losses re-evaluate actor and critic on the stored states. Critic: `v(s_t)` = Linear(37,64) tanh Linear(64,1). Learned parameters: 5,124 (actor 2,627, critic 2,497).

## Return target and losses

Symbols: discount `gamma = 0.95`, `lambda = 0.95`, horizon `T = 15`, `r_hat_t` the predicted reward of transition `t`, `c_hat_t` the predicted probability that the episode continues after it. The lambda-return mixes one-step bootstrapped targets with longer imagined returns:

```text
R_T = v(s_T)
R_t = r_hat_t + gamma c_hat_t ((1 - lambda) v(s_{t+1}) + lambda R_{t+1})
w_t = prod_{j<t} c_hat_j          imagined probability of still being in the episode (w_0 = 1)
critic loss = mean_t w_t (1/2) (v(s_t) - sg(R_t))^2
actor loss  = -mean_t w_t [ log pi(a_t | s_t) sg(R_t - v(s_t)) + eta H(pi(. | s_t)) ]
```

`sg` stops gradients, `H` is the policy entropy and `eta = 0.01`. The actor term is REINFORCE with the critic as baseline: it raises the log-probability of actions whose imagined return exceeded the critic's expectation. It is used instead of backpropagating returns through the model (Dreamer's dynamics gradient) because the observations fed back are discrete argmax choices, through which no gradient flows. One Adam optimizer (learning rate 0.001) on `actor loss + critic loss`; the two losses touch disjoint parameters. `loss.backward()` reaches the actor through `log pi` and the critic through `v`, never the frozen world model.

**Control.** The same actor-critic on `s_t = onehot(o_t)` only (1,028 parameters), trained identically in the same imagination. At the junction it cannot know the cue, so in the real maze it can at best always turn one way (50% success).

## Checks

- Real-maze success rate (correct turn), wrong-turn rate, truncation rate, return and length, per cue; greedy and sampled actions.
- **Imagined versus real return:** the discounted return the world model imagines for the policy from each cue's real start state, next to the real discounted return of the same policy from the same start (Stage 2D's first-class error, now for the policy's own behavior).
- Imagined start-state return and policy entropy during training.
- Unit tests `tests/test_actor_critic.py` (about 2 s): lambda-returns against hand computation and termination, that the control ignores the recurrent state, rollout shapes, start-state coverage, a short `L = 1` run where the recurrent agent reaches 100% in the real maze, beats random by more than 30 points, the control stays at or below 50% and the world model is unchanged, and checkpoint resume.

## Run

```sh
.venv/bin/python -m minecraft_rl.actor_critic --seed 0 --device cpu --output runs/stage2e/seed0/metrics.json
```

## Results (macOS arm64, Python 3.12.11, PyTorch 2.14.0, CPU, about 25 s per seed alone)

The frozen world model reaches 100% held-out turn-reward sign accuracy for every seed. The uniform random policy succeeds in 48-55% of real episodes (return -0.04 to +0.11).

Seed 1, recurrent agent, real maze with greedy actions:

| Update | Imagined lambda-return of the training batch | Entropy | Real success | Real mean length | Sampled success |
| --- | --- | --- | --- | --- | --- |
| 0 | -0.065 | 1.087 | 0% (always truncated) | 40.0 | 49% |
| 50 | 0.167 | 1.055 | 100% | 6.5 | 63% |
| 200 | 0.792 | 0.373 | 100% | 6.5 | 99% |
| 1,000 | 0.874 | 0.100 | 100% | 6.0 | 100% |

The shortest successful episode is 6 steps (4 corridor cells, the junction, the turn). The control over the same updates reaches 50% greedy success (always the same turn) with entropy still 0.77.

Seeds 0-9 after 1,000 updates:

- **Recurrent agent:** 9 of 10 seeds reach 100% greedy and 100% sampled real success with no wrong turns, mean length 6.0-7.0; greedy success first reaches 100% after 50-200 updates. For these seeds the imagined and real discounted returns from the start states agree within 0.03.
- **Seed 0 (model exploitation):** 100% success for cue right, 0% for cue left, overall 50%, no wrong turns. With cue left the agent waits at the junction until truncation. The world model predicts a slowly growing positive reward for waiting at the junction (0.01 to 0.09 per step after 7 steps there), a state the random-policy training data barely covers (at most 6 junction steps in Stage 2D's data). It imagines a discounted return of 1.85 for this behavior, which is impossible in the real maze (maximum 1); the real return is 0. Imagined minus real return: +0.93.
- **No-memory control:** 50% greedy success for 8 seeds (always one turn) and 0% for two (seed 5 waits at the junction, imagining a return of 2.12; seed 7 never commits greedily). Sampled success 0-50%. Imagined minus real return is up to +0.97 for the seeds that wait.

**Interpretation.** Behavior learned entirely inside the learned model transfers to the real environment: the recurrent agent goes from random (about 50%) to perfect cue-dependent turning in 9 of 10 seeds, and it needs the world model's recurrent state to do so (the control cannot exceed 50%). The failures are exactly the Stage 2D failure mode: when the policy reaches a state the model was not trained on, the model invents reward there, and the actor optimizes the invention. The imagined-versus-real return gap flags every such case (+0.93 and +0.97 against at most 0.03 elsewhere), which is why it is reported. No task-specific fix is applied. The roadmap response is data coverage: Stage 2F collects real experience with the current policy, so the waiting behavior would be tried in the real maze, return 0 there, and correct the model.

## Limits

One round only: the world model never sees the policy's own data. The actor uses REINFORCE rather than dynamics gradients, and there is no return normalization, stochastic latent or KL term yet. Evaluation is in the same maze length as training. Next is Stage 2F: an integrated loop that alternates real data collection with the current policy, world-model training and actor-critic training in imagination (`docs/roadmap.md`).
