"""Benchmark a few Minecraft RSSM sizes on one fixed replay batch.

Each candidate trains for a few warm-up updates, then for `--updates` timed
updates on the same batch. One update is the full training step: encoder,
RSSM filter, decoder, losses, backward pass and Adam step. A second pass
times the forward and backward work of the encoder, the RSSM core and the
decoder on their own, with the other parts detached.

The run uses the canonical runtime (one intra-op thread). It reports the
median of the timed updates, the CPU time per update and the peak resident
memory of the process.

Example:

    .venv/bin/python -m minecraft_rl.minecraft_model_benchmark \\
        --dataset runs/stage3/data --output runs/stage3/benchmark.json
"""

import argparse
import json
import resource
import statistics
import time
from pathlib import Path

import torch

from minecraft_rl import runtime
from minecraft_rl.minecraft_dataset import load_episodes
from minecraft_rl.minecraft_replay import (
    CompactVocabulary,
    SequenceReplay,
    load_schema,
    raw_sizes,
)
from minecraft_rl.minecraft_world_model import (
    MinecraftRSSM,
    ModelConfig,
    parameter_count,
    reconstruction_losses,
)
from minecraft_rl.provenance import git_commit

CANDIDATES = {
    "A_conv_32_32": ModelConfig(
        ray_layer="conv", encoder_channels=32, decoder_channels=32
    ),
    "B_conv_16_16": ModelConfig(
        ray_layer="conv", encoder_channels=16, decoder_channels=16
    ),
    "C_conv_16_8": ModelConfig(
        ray_layer="conv", encoder_channels=16, decoder_channels=8
    ),
    "D_patch_16_16": ModelConfig(
        ray_layer="patch", encoder_channels=16, decoder_channels=16
    ),
    "E_patch_32_32": ModelConfig(
        ray_layer="patch", encoder_channels=32, decoder_channels=32
    ),
}


def cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def peak_memory_mb() -> float:
    # macOS reports ru_maxrss in bytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def timed(function, repeats: int) -> tuple[float, float]:
    """Median wall seconds and median CPU seconds of `function()`."""
    wall, cpu = [], []
    for _ in range(repeats):
        started, cpu_started = time.perf_counter(), cpu_seconds()
        function()
        wall.append(time.perf_counter() - started)
        cpu.append(cpu_seconds() - cpu_started)
    return statistics.median(wall), statistics.median(cpu)


def benchmark(
    name: str, config: ModelConfig, vocabulary, batch, warmup: int, updates: int
) -> dict:
    torch.manual_seed(0)
    model = MinecraftRSSM(vocabulary, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    generator = torch.Generator().manual_seed(0)
    o, actions, continues = batch.observations, batch.actions, batch.continues

    def update():
        terms = model.losses(o, actions, continues, generator)
        optimizer.zero_grad()
        terms["total"].backward()
        optimizer.step()

    timed(update, warmup)
    wall, cpu = timed(update, updates)

    def encoder_step():
        model.zero_grad()
        model.encoder(o).square().mean().backward()

    embedded = model.encoder(o).detach()

    def core_step():
        model.zero_grad()
        f = model.filter_embedded(embedded, actions, generator, straight_through=True)
        (f["h"].square().mean() + f["posterior"].square().mean()).backward()

    with torch.no_grad():
        f = model.filter_embedded(embedded, actions, generator)
    s = torch.cat([f["h"], f["z"]], -1)

    def decoder_step():
        model.zero_grad()
        terms = reconstruction_losses(model.decoder(s), o, model.vocabulary)
        sum(t.mean() for t in terms.values()).backward()

    parts = {
        part: timed(step, updates)[0]
        for part, step in (
            ("encoder", encoder_step),
            ("rssm_core", core_step),
            ("decoder", decoder_step),
        )
    }
    result = {
        "name": name,
        "config": config.to_json(),
        "parameters": parameter_count(model),
        "parameters_encoder": parameter_count(model.encoder),
        "parameters_decoder": parameter_count(model.decoder),
        "update_wall_seconds_median": wall,
        "update_cpu_seconds_median": cpu,
        "part_wall_seconds_median": parts,
        "peak_memory_mb_so_far": peak_memory_mb(),
    }
    print(
        f"{name}: {result['parameters']:,} parameters, update {wall:.3f} s "
        f"(cpu {cpu:.3f} s), encoder {parts['encoder']:.3f}, core "
        f"{parts['rssm_core']:.3f}, decoder {parts['decoder']:.3f}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("runs/stage3/data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--updates", type=int, default=7)
    parser.add_argument("--candidates", default=",".join(CANDIDATES))
    args = parser.parse_args()
    runtime.configure(0)
    schema = load_schema(args.dataset)
    train = load_episodes(args.dataset / "train")
    compact = CompactVocabulary.from_episodes(train, raw_sizes(schema))
    vocabulary = compact.model_vocabulary(
        schema["rows"], schema["columns"], schema["max_distance"]
    )
    replay = SequenceReplay(train, compact, args.length)
    batch = replay.sample(args.batch, torch.Generator().manual_seed(0))
    results = [
        benchmark(name, CANDIDATES[name], vocabulary, batch, args.warmup, args.updates)
        for name in args.candidates.split(",")
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "git_commit": git_commit(),
                "runtime": runtime.metadata(torch.device("cpu")),
                "batch": args.batch,
                "length": args.length,
                "warmup_updates": args.warmup,
                "timed_updates": args.updates,
                "vocabulary_id": compact.identifier(),
                "results": results,
                "peak_memory_mb": peak_memory_mb(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
