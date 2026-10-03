import json
import subprocess
import sys

import torch

from minecraft_rl import runtime

PROBE = """
import hashlib, json, sys
import torch
from minecraft_rl import actor_critic, runtime, world_model
runtime.configure(0)
model_config = world_model.Config(seed=0, corridor_length=1, max_steps=12, steps=60)
config = actor_critic.Config(model=model_config, horizon=6, steps=20, start_states=32)
device = torch.device("cpu")
model, episodes, _ = actor_critic.train_world_model(config, device)
starts = actor_critic.start_states(model, episodes)
agent, optimizer = actor_critic.build(config, device)
actor_critic.train(
    model, agent, optimizer, starts, config, torch.Generator().manual_seed(0), 20
)
weights = b"".join(
    p.detach().numpy().tobytes()
    for p in [*model.parameters(), *agent.parameters()]
)
print(json.dumps({
    "digest": hashlib.sha256(weights).hexdigest(),
    "runtime": runtime.metadata(device),
}))
"""


def run_probe() -> dict:
    completed = subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def test_configure_records_the_canonical_runtime():
    runtime.configure(3)
    recorded = runtime.metadata(torch.device("cpu"))
    assert recorded["canonical"]
    assert recorded["global_seed_python_numpy_torch"] == 3
    assert recorded["intra_op_threads"] == 1
    assert recorded["inter_op_threads"] == 1
    assert recorded["deterministic_algorithms"]
    assert recorded["default_dtype"] == "torch.float32"
    assert set(recorded["versions"]) == {"python", "torch", "numpy"}


def test_configure_seeds_every_generator():
    import random

    import numpy

    runtime.configure(5)
    first = (random.random(), numpy.random.rand(), torch.rand(1).item())
    runtime.configure(5)
    assert (random.random(), numpy.random.rand(), torch.rand(1).item()) == first


def test_two_processes_with_the_same_seed_train_identical_weights():
    first, second = run_probe(), run_probe()
    assert first["runtime"]["canonical"]
    assert first["runtime"]["intra_op_threads"] == 1
    assert first["digest"] == second["digest"]
