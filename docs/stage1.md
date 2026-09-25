# Stage 1: tabular GridWorld

**Established behavior:** The fixed 4×4 grid has start `(0, 0)`, goal `(3, 3)` and blocking cells `(1, 1)`, `(2, 1)`, `(1, 2)`; `(x, y)` increases to the right and downward:

```text
S . . .
. # # .
. # . .
. . . G
```

An observation is the integer `state = 4*y + x`, from 0 through 15. Blocked cells have indices but are never visited. Four actions are `UP=0`, `RIGHT=1`, `DOWN=2`, `LEFT=3`. An attempt to move off-grid or into an obstacle leaves the position unchanged and consumes one step. The only reward is `1.0` upon entering the goal; all other steps reward `0.0`. The goal terminates an episode; otherwise the episode truncates after 20 steps. Reset restores the start and counter, and a completed episode cannot be stepped again before reset.

**Decision:** `Q[state][action]` holds a 16×4 nested Python list of floats, initialized to zero, including unvisited cells. At every training step, choose a uniformly random action with probability `epsilon = 0.2`; otherwise choose the largest Q-value. Ties during training are sampled uniformly using the seeded RNG; evaluation ties choose the first action in the defined order. The exploration RNG is `random.Random(seed)` and cannot leak into evaluation. `--seed` and `--episodes` select the run; fixed defaults are 42 and 1,000. Training and greedy evaluation are separate loops; evaluation executes 100 episodes with no Q updates. The same deterministic map is used for all episodes, so repeating evaluation measures the policy on this map, **not** generalization to unseen worlds.

The one-step update for a transition `(s, a, r, s')` is:

```text
target = r + gamma * max(Q[s'])       if the episode continues
       = r                           if terminated or truncated
Q[s][a] = Q[s][a] + alpha * (target - Q[s][a])
```

Here `s` and `s'` are state indices, `a` is an action, `r` is the immediate reward, `alpha = 0.5` is the fraction moved toward the target, and `gamma = 0.95` discounts future reward. This implementation uses zero continuation at the time limit: the truncated episode is treated as ended, not bootstrapped. There are no tensors, neural parameters, differentiable losses, `.backward()` calls, or frameworks. The 64 table entries are learned directly. Integer toy coordinates are intentionally exposed through state encoding; this is **not** an acceptable Minecraft policy observation.

**Inspection:** Run `.venv/bin/python -m minecraft_rl.train --episodes 1000 --seed 42 --output runs/stage1/metrics.json`. Standard output prints success/step counts, each state and its four Q-values, and a policy map with `S`, `G`, `#`, and action arrows. The ignored JSON stores full-precision `q_table`, action order, `greedy_policy` (unvisited/terminal states included for inspection, not valid moves), config, commit when available, seed, duration, training reward, successes and steps, last-100 training successes, evaluation steps and success fraction. Only time and timestamp vary between otherwise identical seeded runs. No checkpoints are required: rerun the small experiment instead.

**Reward-hacking check:** Wall collisions earn nothing. Reaching the goal repeatedly across episodes earns one reward per episode; this is the task's definition but would be a reset exploit for many Minecraft rewards. Truncation pays nothing. No intermediate novelty or shaping reward is available for farming.

**Experiment results (local smoke runs, seed 42, 1,000 training episodes, uncommitted checkout):** A first attempt used the first action for all greedy ties even during training. It found no goal in 1,000 episodes, leaving all Q-values at zero (0/100 evaluation successes). This was an exploration failure on the sparse-reward map, not evidence that longer training was the right fix. After sampling greedy training ties with the seeded RNG, training recorded 958 successful episodes in 8,212 steps, including 100/100 successes in the last 100 episodes. Greedy evaluation reached the goal 100/100 times in 600 total steps, six per episode (a shortest route). The committed example configuration is not a Minecraft or unseen-map generalization result. The commit field was null because no commit existed; the ignored JSON run artifact is local. No Stage 2 work is part of this experiment.
