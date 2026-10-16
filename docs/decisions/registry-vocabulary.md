# Decision: full Minecraft registry vocabulary for the world model

Status: implemented and smoke-tested. The Stage 3 and Stage 4 checkpoints keep the compact vocabulary and load unchanged. New world models use the registry vocabulary with the "factored" heads. No new model is trained on the full data yet, and none is canonical.

## The old limitation

`CompactVocabulary` maps the raw registry ids that occur in the training split to dense indices, plus one shared index for every other id (`unknown`). The canonical model has 60 block, 2 fluid, 8 entity, 8 item and 1 effect index. Every id outside the training split shares one embedding and one output class, so the model cannot tell an unseen item from another unseen item. Measured on recorded data:

- The sand item (`minecraft:sand`, item id 88) is unknown to the model in every temporal-actor life, because the random block-breaking data never kept sand long enough in the inventory.
- `runs/stage3/data/eval_seed` shows 10 block types that the training split does not have (for example `minecraft:packed_ice`, `minecraft:snow_block`, `minecraft:pointed_dripstone`).

A model that must learn to collect sand, iron and diamonds cannot work like this.

## The registry

`scripts/minecraft-registry.py` launches the project-owned Minecraft 26.3 client once, reads `DEBUG_REGISTRY` and `SCHEMA`, makes sure that the sizes agree, and writes `runs/registry/minecraft-26.3.json` (`minecraft-registry-v1`) with a SHA-256 fingerprint of the version and every name in index order.

| Family | Registered types | Model categories | Empty or no-hit value |
|---|---|---|---|
| Block | 1,286 | 1,286 | ray kind "no hit" |
| Fluid | 5 | 5 | ray kind "no hit" |
| Entity | 161 | 161 | ray kind "no hit" |
| Item | 1,658 | 1,658 | item 0 `minecraft:air` is the empty slot |
| Status effect | 40 | 41 | effect value 0 is the empty slot, effect id + 1 otherwise |

Fingerprint: `706b7869b5855014691c67e700d981dfcdfdbd54c26dd37db72964e1d0b2ffa6`. The file is generated under the ignored `runs/`. `minecraft_registry.PINNED_FINGERPRINT` and `PINNED_SIZES` pin its content in the repository, and `Registry.pinned` refuses a file that differs.

## The mapping

`RegistryVocabulary` (`src/minecraft_rl/minecraft_replay.py`) uses the raw registry index as the model index. The environment already reports raw indices (`BuiltInRegistries.*.getId`), and the stored episodes keep them, so no historical data is rewritten. The mapping is deterministic: it depends only on the registry, never on which episodes a dataset holds or in which order they were collected.

- Every registered type is a distinct category. No id is unknown.
- The joint ray class keeps its form: 0 is no hit, then every block, fluid and entity type. There are 1,453 ray classes.
- An id outside the registry raises an error (`check_raw_ids`).
- A model records `registry-vocabulary-v1`, the Minecraft version, the fingerprint and the sizes in `extra["registry"]`. `check_registry` refuses data from another registry instead of reading ids with the wrong identities.
- Names are only for diagnostics and metadata. The model never sees a name or a feature from a name.

## Encoder

The encoder is unchanged: trainable embedding tables for ray classes (1,453 x 8), items (1,658 x 16) and effects (41 x 8). Every registry id has its own row from random initialization. An id that never appears in training keeps its random row: the model supports it but knows nothing about it until it sees it. When future data shows it, the gradient reaches only that row, without a new model or a new vocabulary.

## Decoder: "factored" heads

A full registry with the old "joint" heads would have 19.6 million parameters. Most are the per-slot item heads (36 inventory slots x 1,658 items from a 256-value trunk is 15.3 million) and a softmax over 1,453 classes for each of 825 rays. `ModelConfig(heads="factored")` (`FactoredDecoder` in `src/minecraft_rl/minecraft_world_model.py`) replaces only the categorical output layers:

- Rays: the patch ray-grid layer gives a 16-value feature per ray, as before. Shared by all rays, `kind_head` (16 to 4) gives log p(kind) over no hit, block, fluid and entity, and `type_heads[kind]` gives log p(type | kind) over every registered type of that kind (16 to 1,286, 5 and 161). log p(class) = log p(kind) + log p(type | kind), so every registered type is still one class of the same joint ray-class distribution. A separate layer gives the distance.
- Items: the self trunk gives one 32-value feature per item slot (36 inventory, 4 armor, 1 offhand), plus a learned slot embedding. One item classifier (32 to 1,658) is shared by all 41 slots. Effects work the same way with their own shared classifier. Counts and durability keep their own heads.
- Loss: the exact cross-entropy of the target. The type softmax of a kind runs only on rays whose target has that kind, so a no-hit ray has no type loss. `TargetLogSoftmax` computes it in chunks of 4,096 rays and recomputes each chunk in the backward pass, so it never holds the logits of the whole batch (16 x 33 x 825 rays x 1,286 blocks would be 560 million values). It is exact, with a `gradcheck` test, not a sampled or approximate loss.
- The objective (`group_mean`, component scale 825, pitch 128), the encoder, the RSSM core and the reductions are unchanged.

Input embeddings and output classifiers are not tied: the item embedding has 16 values, the slot feature 32.

## Cost

| | Compact, joint (canonical) | Registry, joint | Registry, factored |
|---|---|---|---|
| Parameters | 2,101,421 | 19,618,289 | 2,524,309 |
| Encoder | 543,048 | 580,776 | 580,776 |
| Decoder | 783,844 | 18,262,984 | 1,169,004 |
| RSSM core and continuation | 774,529 | 774,529 | 774,529 |

One training update, batch 16 windows of 32 transitions of the Stage 3 training data, efficiency cores, one intra-op thread (`runs/registry/smoke/cost.json`):

| | Compact, joint | Registry, factored |
|---|---|---|
| Update time | 0.52 s | 3.8 s |
| Peak resident memory of the process | - | 2.1 GiB (1.1 GiB before the first update) |

The factored model has 20% more parameters than the compact model. An update takes about 7 times longer. Almost all of the extra time is the exact block-type softmax: 1,286 classes for each of the about 360,000 block rays of a batch. Where this becomes too slow, the measured next step is fewer rays per loss evaluation or a faster device, not an approximate loss.

## Checkpoints

- `minecraft-rssm-checkpoint-v3` records `heads` and `item_slot_features`. A factored checkpoint without `extra["registry"]` is refused when it is saved.
- v1 and v2 checkpoints have no heads field and load as "joint" with their compact vocabulary. A v1 or v2 file that names heads is refused. Every earlier Stage 3 and Stage 4 checkpoint loads without change, and `load_world_model` still refuses any non-canonical config.
- `minecraft_world_model_eval` handles both decoders: `categorical_outputs` turns factored outputs into the joint form for small evaluation batches, and `unknown_classes` is empty for a registry vocabulary.

## Weight migration

`src/minecraft_rl/minecraft_registry_migration.py` initializes a factored registry model from a joint compact model. It copies:

- Every layer whose name and shape do not depend on the vocabulary: the ray-grid layers, the self encoder, the fusion layer, the RSSM core, the self trunk, the scalar, slot, count and durability heads, and the continuation head. For the continued block-breaking model, 2,000,449 values.
- Embedding rows by registry id: the row of compact index i goes to registry id `ids[i]` (80 rows). The compact unknown row is not copied, because it stood for every unseen id.
- Ray type rows of the old ray-class head to the type heads by registry id (70 rows), and the distance row.

The kind head, the shared item and effect classifiers and every unseen id start fresh. A test checks that every migrated row lands on its own registry id and that unseen rows keep their initialization. Migration copies representations of ids the old model saw. It gives no meaning to an unseen id.

## Tiny-overfit smoke test

`src/minecraft_rl/minecraft_registry_smoke.py`, the migrated factored model, 8 fixed real windows of 32 transitions: 4 from temporal-actor lives 0 and 1 that hold the sand item, 4 from the starts of 4 different Stage 3 training episodes. The windows show 26 block types, empty and filled slots, and the sand item, which was unknown to the old model. 300 updates of the canonical objective, Adam 1e-3, clip 1000, nothing tuned, 11 minutes on the efficiency cores. Metrics are posterior reconstruction on the same windows (`runs/registry/smoke/tiny.json`).

| | Before (migrated) | After 300 updates |
|---|---|---|
| Ray class accuracy | 0.013 | 0.929 |
| Ray kind accuracy | 0.013 | 0.976 |
| Empty slot predicted empty | 0.00 | 1.00 |
| Sand item in its 132 slots | 0.00 | 1.00 |
| Block types above 0.5 accuracy, of 26 | 0 | 14 |
| Sandstone / sand / short grass / grass block rays | 0.0 / 0.0 / 0.0 / 0.0 | 0.997 / 0.994 / 0.919 / 0.724 |

Before training the migrated model predicts nothing right, because the kind head and the item classifier are fresh. After training it tells the sand item apart from empty slots, and many block types apart from each other, each with its own registry id. Block types with few rays in the windows stay low (for example `fern`, 144 rays, 0.0; `oak_log`, 195 rays, 0.26). A first run with only the 4 sand windows (2 block types) fitted 0.997 ray accuracy (`runs/registry/smoke-sand-only`).

Held out, the same overfit model (`runs/registry/smoke/held_out.json`):

| | Ray class accuracy | Ray kind accuracy | Block types in the overfit windows: rays, accuracy | Block types not in them: rays, accuracy |
|---|---|---|---|---|
| Temporal-actor lives 2 and 3, 4 windows | 0.917 | 0.986 | 6 types, 106,024 rays, 0.92 | 1 type (`cactus_flower`), 4 rays, 0.0 |
| Stage 3 `eval_seed`, 4 windows | 0.362 | 0.728 | 13 types, 49,936 rays, 0.24 | 7 types, 26,146 rays, 0.0 |

The sand item in the held-out lives: 66 slots, accuracy 1.00. Block types that the 8 windows never showed, such as `packed_ice` and `snow_block`, have their own ids and are never predicted, as expected from untrained rows. The `eval_seed` result shows that 8 windows do not make a general model. That needs ordinary training on the full data.

## Limitations

- Supporting an id is not knowing it. An unseen id has a random embedding and a random classifier row until data shows it.
- The registry is fixed for Minecraft 26.3. Another version needs a new registry file, and its checkpoints are refused for the old registry.
- An update is about 7 times slower than with the compact vocabulary. The Stage 3 training scripts still build compact models. The first full training run of a registry model is a separate task.
- The reward heads and actors of Stage 4 read the compact models. A registry model needs its own reward head before any actor uses it.
