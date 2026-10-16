"""Measure and smoke-test the full-registry world model
(docs/decisions/registry-vocabulary.md).

Stages, written to `--output`:

1. `cost.json`: parameters by part for the compact model, the full-registry
   model with the old joint heads (built, not trained) and the factored
   model; one real training update of the factored model on a Stage 3
   batch: time, peak memory, and whether every part gets gradients.
2. `tiny.json`: a deliberate overfit of 8 fixed real windows: 4 from
   temporal-actor lives with the sand item (an unknown id of the compact
   model) and 4 from different Stage 3 training episodes with many block
   types, empty and filled slots. The model is the factored registry
   model, migrated from the continued block-breaking model or fresh.
   Reports one-step reconstruction per category before and after.
3. `held_out.json`: posterior reconstruction on held-out real lives, by
   category: seen in the overfit windows or not.

No hyperparameter is tuned: the canonical objective, Adam 1e-3, clip 1000.
"""

import argparse
import json
import resource
import statistics
import time
from pathlib import Path
from typing import Any

import numpy
import torch

from minecraft_rl import macos_qos, runtime
from minecraft_rl.minecraft_actor import CANONICAL_OBJECTIVE
from minecraft_rl.minecraft_dataset import join_segments, load_episodes
from minecraft_rl.minecraft_interface import RAY_KIND_BLOCK
from minecraft_rl.minecraft_registry import Registry
from minecraft_rl.minecraft_registry_migration import migrate
from minecraft_rl.minecraft_replay import (
    CompactVocabulary,
    RegistryVocabulary,
    SequenceReplay,
    load_schema,
)
from minecraft_rl.minecraft_world_model import (
    MinecraftRSSM,
    ModelConfig,
    load_checkpoint,
    parameter_count,
    ray_class,
    save_checkpoint,
)
from minecraft_rl.minecraft_world_model_eval import categorical_outputs
from minecraft_rl.provenance import git_commit

REGISTRY = Path("runs/registry/minecraft-26.3.json")
STAGE3 = Path("runs/stage3/data")
OLD = Path("runs/stage4/break-continued/world_model.pt")
LIVES = Path("runs/stage4/temporal-eval")
SAND_ITEM = "minecraft:sand"


def peak_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def parts(model: MinecraftRSSM) -> dict[str, int]:
    total = parameter_count(model)
    encoder = parameter_count(model.encoder)
    decoder = parameter_count(model.decoder)
    return {
        "total": total,
        "encoder": encoder,
        "decoder": decoder,
        "core": total - encoder - decoder,
    }


def train_step(model, optimizer, batch, generator) -> dict[str, float]:
    model.train()
    terms = model.losses(
        batch.observations,
        batch.actions,
        batch.continues,
        generator,
        CANONICAL_OBJECTIVE,
    )
    optimizer.zero_grad()
    terms["total"].backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1000.0)
    optimizer.step()
    return {k: float(v.detach()) for k, v in terms.items()} | {"grad_norm": float(norm)}


def cost(args, registry: Registry) -> dict[str, Any]:
    schema = load_schema(STAGE3)
    vocabulary = RegistryVocabulary(registry)
    v = vocabulary.model_vocabulary(
        schema["rows"], schema["columns"], schema["max_distance"]
    )
    old, _ = load_checkpoint(OLD)
    torch.manual_seed(0)
    joint = MinecraftRSSM(v, ModelConfig(heads="joint"))
    joint_parts = parts(joint)
    del joint
    torch.manual_seed(0)
    model = MinecraftRSSM(v, ModelConfig(heads="factored"))
    replay = SequenceReplay(load_episodes(STAGE3 / "train"), vocabulary, 32)
    generator = torch.Generator().manual_seed(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-5)
    before = peak_mib()
    times = []
    for update in range(args.cost_updates):
        batch = replay.sample(16, generator)
        started = time.perf_counter()
        terms = train_step(model, optimizer, batch, generator)
        times.append(time.perf_counter() - started)
        if update == 0:
            no_grad = [
                n
                for n, p in model.named_parameters()
                if p.grad is None or p.grad.abs().sum() == 0
            ]
            first = terms
    return {
        "registry_sizes": registry.sizes(),
        "parameters": {
            "compact_joint (break-continued)": parts(old),
            "registry_joint (not trained)": joint_parts,
            "registry_factored": parts(model),
        },
        "batch": "16 windows of 32 transitions, Stage 3 train",
        "update_seconds_median": statistics.median(times[1:]),
        "update_seconds": times,
        "peak_rss_mib_before": before,
        "peak_rss_mib_after": peak_mib(),
        "first_update": first,
        "parameters_without_gradient_first_update": no_grad,
        "finite": all(numpy.isfinite(v) for v in first.values()),
    }


def overfit_windows(lives, stage3, vocabulary, registry, count: int):
    """`count` fixed windows of 32: half from temporal-actor lives that hold
    the sand item, half from the starts of different Stage 3 training
    episodes, which show many block types."""
    sand = registry.names["item"].index(SAND_ITEM)
    replay = SequenceReplay(lives + stage3, vocabulary, 32)
    picked = []
    for e, episode in enumerate(replay.episodes[: len(lives)]):
        items = torch.as_tensor(episode.arrays["obs_inventory_item"])
        has_sand = (items == sand).any(-1)
        for start in range(0, episode.steps - 32, 96):
            if has_sand[start : start + 33].any() and len(picked) < count // 2:
                picked.append((e, start))
    for e in range(len(lives), len(lives) + count - len(picked)):
        picked.append((e, 0))
    return replay, picked


@torch.no_grad()
def reconstruction(model, batch, generator, registry, seen: dict | None = None):
    """Posterior reconstruction of every state of `batch` by category."""
    model.eval()
    model.mode_latents = True
    f = model.filter(batch.observations, batch.actions, generator)
    s = torch.cat([f["h"], f["z"]], -1)
    out = categorical_outputs(model, model.decoder(s))
    model.mode_latents = False
    o = batch.observations
    v = model.vocabulary
    target_rays = ray_class(o["ray_kind"], o["ray_type"], v)
    predicted_rays = out["ray_class"].argmax(-1)
    items = o["inventory_item"]
    predicted_items = out["inventory_item"].argmax(-1)
    result: dict[str, Any] = {
        "ray_accuracy": float((predicted_rays == target_rays).float().mean()),
        "ray_kind_accuracy": float(
            (v.class_kinds()[predicted_rays] == o["ray_kind"]).float().mean()
        ),
        "empty_slot_accuracy": float((predicted_items[items == 0] == 0).float().mean()),
        "filled_slot_accuracy": float(
            (predicted_items[items > 0] == items[items > 0]).float().mean()
        )
        if (items > 0).any()
        else None,
    }
    blocks = {}
    block_rays = o["ray_kind"] == RAY_KIND_BLOCK
    for block in o["ray_type"][block_rays].unique().tolist():
        mask = block_rays & (o["ray_type"] == block)
        name = registry.names["block"][block]
        blocks[name] = {
            "rays": int(mask.sum()),
            "accuracy": float(
                (predicted_rays[mask] == target_rays[mask]).float().mean()
            ),
        }
        if seen is not None:
            blocks[name]["in_overfit_windows"] = block in seen["blocks"]
    item_rows = {}
    for item in items[items > 0].unique().tolist():
        mask = items == item
        name = registry.names["item"][item]
        item_rows[name] = {
            "slots": int(mask.sum()),
            "accuracy": float((predicted_items[mask] == item).float().mean()),
        }
        if seen is not None:
            item_rows[name]["in_overfit_windows"] = item in seen["items"]
    result["blocks"] = dict(sorted(blocks.items(), key=lambda kv: -kv[1]["rays"]))
    result["items"] = item_rows
    return result


def categories(batch) -> dict[str, set[int]]:
    o = batch.observations
    return {
        "blocks": set(o["ray_type"][o["ray_kind"] == RAY_KIND_BLOCK].unique().tolist()),
        "items": set(o["inventory_item"].unique().tolist()) - {0},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/registry/smoke"))
    parser.add_argument("--cost-updates", type=int, default=6)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    macos_qos.run_in_background()
    runtime.configure(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    registry = Registry.pinned(REGISTRY)
    vocabulary = RegistryVocabulary(registry)
    path = args.output / "cost.json"
    if not path.exists():
        path.write_text(json.dumps(cost(args, registry), indent=1))
    print(json.loads(path.read_text())["update_seconds_median"], flush=True)

    train_lives = join_segments(load_episodes(LIVES / "trained" / "eval"))
    train_lives = [e for e in train_lives if e.meta.episode in (0, 1)]
    stage3 = sorted(load_episodes(STAGE3 / "train"), key=lambda e: e.meta.episode)
    replay, picked = overfit_windows(
        train_lives,
        stage3[:: max(1, len(stage3) // 4)][:4],
        vocabulary,
        registry,
        args.windows,
    )
    tiny = replay.batch(picked)
    seen = categories(tiny)
    schema = load_schema(STAGE3)
    v = vocabulary.model_vocabulary(
        schema["rows"], schema["columns"], schema["max_distance"]
    )
    old, extra = load_checkpoint(OLD)
    compact = CompactVocabulary.from_json(extra["vocabulary"])
    torch.manual_seed(args.seed)
    model = MinecraftRSSM(v, ModelConfig(heads="factored"))
    migration = migrate(old, compact, model)
    generator = torch.Generator().manual_seed(args.seed + 1)
    before = reconstruction(
        model, tiny, torch.Generator().manual_seed(5), registry, seen
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-5)
    history = []
    started = time.perf_counter()
    for update in range(1, args.updates + 1):
        terms = train_step(model, optimizer, tiny, generator)
        if update % 50 == 0:
            history.append({"update": update, "total": terms["total"]})
            print(json.dumps(history[-1]), flush=True)
    seconds = time.perf_counter() - started
    after = reconstruction(
        model, tiny, torch.Generator().manual_seed(5), registry, seen
    )
    sand = registry.names["item"].index(SAND_ITEM)
    (args.output / "tiny.json").write_text(
        json.dumps(
            {
                "git_commit": git_commit(),
                "windows": picked,
                "updates": args.updates,
                "seconds": seconds,
                "migration": migration,
                "sand_item_id": sand,
                "sand_in_compact_vocabulary": sand in compact.ids["item"],
                "seen_blocks": len(seen["blocks"]),
                "seen_items": sorted(registry.names["item"][i] for i in seen["items"]),
                "before": before,
                "after": after,
                "history": history,
            },
            indent=1,
        )
    )
    save_checkpoint(
        args.output / "world_model.pt",
        model,
        {
            "registry": vocabulary.to_json(),
            "objective": CANONICAL_OBJECTIVE.to_json(),
            "experimental": "tiny overfit of the registry smoke test",
        },
    )
    held_lives = join_segments(load_episodes(LIVES / "trained" / "eval"))
    held_lives = [e for e in held_lives if e.meta.episode in (2, 3)]
    held_replay = SequenceReplay(held_lives, vocabulary, 32)
    held = held_replay.batch([(0, 0), (0, 400), (1, 0), (1, 400)])
    seed_eps = load_episodes(STAGE3 / "eval_seed")
    seed_replay = SequenceReplay(seed_eps, vocabulary, 32)
    seed_batch = seed_replay.batch([(i, 0) for i in range(min(4, len(seed_eps)))])
    (args.output / "held_out.json").write_text(
        json.dumps(
            {
                "temporal_actor_lives_2_3": reconstruction(
                    model, held, torch.Generator().manual_seed(6), registry, seen
                ),
                "stage3_eval_seed": reconstruction(
                    model, seed_batch, torch.Generator().manual_seed(6), registry, seen
                ),
            },
            indent=1,
        )
    )
    print("written", args.output, flush=True)


if __name__ == "__main__":
    main()
