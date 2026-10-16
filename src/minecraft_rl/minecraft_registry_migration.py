"""Initialize a registry-vocabulary model from a compact-vocabulary model
(docs/decisions/registry-vocabulary.md).

Only weights whose meaning is the same in both models are copied:

- Every layer whose shape and meaning do not depend on the vocabulary: the
  ray-grid encoder layer, the self encoder, the fusion layer, the RSSM core
  (action layer, GRU, prior, posterior), the decoder's ray-grid layer and
  self trunk, the scalar, slot, count and durability heads, and the
  continuation head. Layers match by name and shape.
- Embedding rows, by registry id: the row of compact index i goes to the
  row of registry id `ids[i]`. A compact index is never copied to any other
  registry id. The compact unknown row is not copied: it stood for every
  unseen id at once.
- The ray type heads of the factored decoder, by registry id, from the rows
  of the old joint ray-class head, and the ray distance row. The kind head
  has no counterpart and starts fresh, so the kind probabilities need
  training.

Registry ids that the compact model never saw keep their fresh random
initialization. The new shared item and effect classifiers have no
counterpart in the old per-slot heads and also start fresh. Migration
copies learned representations of known ids. It gives an unseen id no
meaning.
"""

import torch

from minecraft_rl.minecraft_interface import (
    RAY_KIND_BLOCK,
    RAY_KIND_ENTITY,
    RAY_KIND_FLUID,
)
from minecraft_rl.minecraft_replay import CompactVocabulary
from minecraft_rl.minecraft_world_model import FactoredDecoder, MinecraftRSSM

KINDS = (
    ("block", RAY_KIND_BLOCK),
    ("fluid", RAY_KIND_FLUID),
    ("entity", RAY_KIND_ENTITY),
)
EMBEDDINGS = {
    "encoder.ray_embedding.weight",
    "encoder.item_embedding.weight",
    "encoder.effect_embedding.weight",
}


def ray_class_rows(
    compact: CompactVocabulary, old: MinecraftRSSM
) -> dict[int, tuple[str, int]]:
    """Old joint ray class -> (family, registry id) for every known type."""
    out = {}
    for family, kind in KINDS:
        first = old.vocabulary.kind_offsets()[kind]
        for index, registry_id in enumerate(compact.ids[family]):
            out[first + index] = (family, registry_id)
    return out


@torch.no_grad()
def migrate(
    old: MinecraftRSSM, compact: CompactVocabulary, new: MinecraftRSSM
) -> dict[str, int]:
    """Copy the weights of `old` into `new` as the module docstring says.
    Returns the number of copied values or rows per group."""
    if old.config.heads != "joint" or not isinstance(new.decoder, FactoredDecoder):
        raise ValueError("migration goes from joint heads to factored heads")
    for field in ("ray_layer", "patch", "fusion", "hidden", "embed_dim"):
        if getattr(old.config, field) != getattr(new.config, field):
            raise ValueError(f"migration needs the same {field}")
    counts = {"shared_values": 0, "embedding_rows": 0, "ray_type_rows": 0}
    old_state = old.state_dict()
    for name, value in new.state_dict().items():
        if name in EMBEDDINGS or name not in old_state:
            continue
        if old_state[name].shape == value.shape:
            value.copy_(old_state[name])
            counts["shared_values"] += value.numel()

    rows = ray_class_rows(compact, old)
    offsets = new.vocabulary.kind_offsets()
    old_rays = old.encoder.ray_embedding.weight
    new_rays = new.encoder.ray_embedding.weight
    new_rays[0] = old_rays[0]
    counts["embedding_rows"] += 1
    for old_class, (family, registry_id) in rows.items():
        new_rays[offsets[dict(KINDS)[family]] + registry_id] = old_rays[old_class]
        counts["embedding_rows"] += 1
    for name, family in (("item_embedding", "item"), ("effect_embedding", "effect")):
        old_table = getattr(old.encoder, name).weight
        new_table = getattr(new.encoder, name).weight
        for index, registry_id in enumerate(compact.ids[family]):
            new_table[registry_id] = old_table[index]
            counts["embedding_rows"] += 1

    old_head = old.decoder.ray_head
    decoder = new.decoder
    decoder.ray_distance.weight.copy_(old_head.weight[-1:])
    decoder.ray_distance.bias.copy_(old_head.bias[-1:])
    for old_class, (family, registry_id) in rows.items():
        head = decoder.type_heads[family]
        head.weight[registry_id] = old_head.weight[old_class]
        head.bias[registry_id] = old_head.bias[old_class]
        counts["ray_type_rows"] += 1
    return counts
