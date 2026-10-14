# Reward contract

Status: decided for the first Minecraft actor milestone. Reward version `reward-v1`. Implemented in `src/minecraft_rl/minecraft_reward.py`; the constants live in one versioned specification there.

## Goal and scope

The final goal is to kill the Ender Dragon on held-out fresh seeds. The actor gets one scalar reward per step. Inside, the reward is a sum of named components. Every transition stores the components, so a run can show where its reward came from.

The components are training signals and diagnostics. They are never policy observations. The policy gets only the scalar, through the world model's reward prediction and the real return.

## Where human Minecraft knowledge enters

The reward is a training objective, not a plan. It says which outcomes are good. It never says how to reach them: no recipe, no route, no skill, no prerequisite reaches the policy. Human knowledge enters in these places, and only these:

- The importance weights: which items count as progress, and how much.
- The progression milestones: which advancements mark major steps toward the End.
- The terminal event: the Dragon kill.
- The relative magnitudes of all components.

What stays learned: which actions work in which states, recipes and their prerequisites, tool use, navigation, where resources are, combat, and the long-term strategy.

## Channels

| Channel | Event | Credit rule | `reward-v1` value |
|---|---|---|---|
| `block_seen_novelty` | a block type appears in the ray grid | first time per episode, per block type | 0.01 |
| `block_broken_novelty` | the player breaks a block | first time per episode, per block type | 0.05 |
| `item_acquired_novelty` | an item type appears in the inventory, armor or offhand | first time per episode, per item type | 0.1 |
| `importance` | an important item type is acquired | first time per episode, per item type, on top of the novelty | table below |
| `advancement` | the player completes an advancement that has a display | once per advancement, per episode | 1.0 |
| `progression` | a major milestone advancement | once per milestone, per episode, on top of the advancement | table below |
| `survival` | the player dies | once, at the terminal step | -1.0 |
| `terminal` | the Ender Dragon dies (`minecraft:end/kill_dragon`) | once per episode | 1,000 |

Every block type and every item type can earn novelty. No list restricts it.

Seeing is worth least, breaking more, and acquiring most. So breaking an ore with the wrong tool earns the break novelty, and only the useful drop earns the acquisition and its importance. The policy can learn that one way of breaking a block pays more than another without any tool rule in its observation.

An item counts as acquired when its type is in the inventory, armor or offhand after a step and was never there before in the episode. The items at the episode start count as held and earn nothing. A crafted item counts as acquired.

Advancements without a display are recipe unlocks. They earn nothing, because they mostly reward picking up ingredients and would duplicate the item channel.

## Novelty and importance are separate

Novelty says that the agent met something new. Importance says that the event helps long-term progression. Every type earns novelty; only the items in the importance table earn an extra bonus. The table is human knowledge and the only place where the reward names useful items.

Importance in `reward-v1`, paid on the first acquisition of a matching item type. Patterns use shell-style wildcards on registry names; an item takes the highest matching weight:

| Items | Bonus | Reason |
|---|---|---|
| `minecraft:*_log`, `minecraft:*_stem` | 0.2 | wood starts every tool chain |
| `minecraft:*_planks`, `minecraft:stick` | 0.1 | first crafted materials |
| `minecraft:crafting_table` | 0.3 | unlocks 3 x 3 crafting |
| `minecraft:wooden_pickaxe` | 0.5 | first mining tool |
| `minecraft:cobblestone`, `minecraft:coal` | 0.3 | stone tools and fuel |
| `minecraft:stone_pickaxe`, `minecraft:furnace` | 0.7 | iron mining and smelting |
| `minecraft:raw_iron`, `minecraft:bucket` | 1.0 | iron age |
| `minecraft:iron_ingot`, `minecraft:flint_and_steel` | 1.5 | smelted iron, portal lighting |
| `minecraft:iron_pickaxe`, `minecraft:obsidian`, `minecraft:blaze_powder` | 2.0 | diamond mining, portal, eyes |
| `minecraft:diamond`, `minecraft:diamond_pickaxe` | 3.0 | rare progression resources |
| `minecraft:blaze_rod`, `minecraft:ender_pearl` | 5.0 | End access ingredients |
| `minecraft:ender_eye` | 8.0 | locates and opens the End portal |

The values grow roughly with the number of steps that a human needs to reach the item. They are initial values, chosen by hand, not tuned.

## Progression milestones

Progression rewards are major steps toward the End, defined by vanilla advancement IDs. They add to the advancement reward.

| Advancement | Bonus |
|---|---|
| `minecraft:story/enter_the_nether` | 10 |
| `minecraft:nether/obtain_blaze_rod` | 10 |
| `minecraft:story/follow_ender_eye` | 15 |
| `minecraft:story/enter_the_end` | 30 |

Later milestones (for example a new lowest Dragon health) can join this table without changing the policy interface: the policy still sees one scalar.

## The Dragon kill dominates

The terminal reward of 1,000 is larger than the sum of every other positive reward that one episode can earn. `shaping_upper_bound` computes that sum from the real registries: every block type seen and broken, every item type acquired, every importance bonus, every displayed advancement and every milestone. Each run records the bound in its metadata, and a test checks that it stays below the terminal reward.

## Anti-farming rules

All novelty, importance, advancement and milestone credit is type-first and once per episode. So these loops earn nothing after the first time:

- place a block, break it, place it again, break it again: the block type earns break novelty once;
- acquire an item, drop it, pick it up again: the item type earns acquisition reward once;
- turn the camera around and around: each block type earns seen novelty once.

No reward exists for damage, healing, standing still or staying alive, so damage and heal loops and idle survival earn nothing. A future Dragon-health signal must reward a new lowest health, never repeated damage.

Remaining risks, measured but not prevented in `reward-v1`:

- Novelty resets each episode, so an agent can earn the same cheap novelty in every episode. That is intended: it rewards exploration per life.
- Seen novelty is cheap and can dominate a short episode. Its value (0.01) is a tenth of acquisition.
- The death penalty can make the agent avoid risk. It is small (-1) next to progression.
- An invalid abstract action earns nothing, so spam is free apart from its time cost. The actor experiment reports the invalid rate.

## Determinism and data

The reward is a pure function of the episode: the observations (seen blocks, held items), the broken block types and the completed advancements that the environment reports for each step. The same trajectory gives the same reward. A dataset stores the scalar, the components per step, the events and the reward version, so a run can recompute or relabel its reward.

The broken blocks and advancements come from the environment as privileged events in the step reply, in a field separate from the observation. They never reach a policy observation.

Old Stage 3 data has no events. Seen and acquired novelty can be recomputed from its observations, but breaks and advancements cannot, so it is not relabeled.

## Metrics

Every step, episode and evaluation reports the total reward, the reward per component, the number of novelty and importance events, the advancement reward and the terminal reward.
