"""Gradient share of each loss component on the shared parts of the model.

A loss component is one group of `LOSS_GROUPS` (ray class, ray distance,
self state, inventory), the continuation loss, or the KL loss. For one fixed
batch, `component_gradients` takes the gradient of each weighted component
alone and reports its norm on three shared places:

- `ray_encoder`: the parameters of the ray embedding and the ray-grid layer
  of the encoder.
- `rssm_core`: the parameters of the GRU cell, the action layer, the prior
  and the posterior.
- `latent_state`: the decoder input s = [h, z] itself, as one tensor.

Each component passes through the objective with its training weight, so
the norms show how much each component moves the shared parameters in one
update. The pitch part of the self state is reported on its own as well. It
is already part of "self". The diagnostic never changes parameters.
"""

import torch

from minecraft_rl.minecraft_world_model import (
    LOSS_GROUPS,
    MinecraftRSSM,
    Objective,
)

PLACES = ("ray_encoder", "rssm_core", "latent_state")
RAY_ENCODER_PREFIXES = ("encoder.ray_embedding.", "encoder.grid.")
CORE_PREFIXES = ("cell.", "action_net.", "prior_net.", "posterior_net.")


def _parameters(model: MinecraftRSSM, prefixes: tuple[str, ...]) -> list:
    return [p for n, p in model.named_parameters() if n.startswith(prefixes)]


def _norm(gradients) -> float:
    squares = [(g.double() ** 2).sum() for g in gradients if g is not None]
    return float(torch.sqrt(sum(squares))) if squares else 0.0


def component_gradients(
    model: MinecraftRSSM,
    batch,
    objective: Objective,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """For each component: its weighted loss value and the gradient norm of
    that loss on each place in `PLACES`."""
    was_training = model.training
    model.train()
    generator = torch.Generator().manual_seed(seed)
    terms, s = model.loss_graph(
        batch.observations, batch.actions, batch.continues, generator, objective
    )
    scale = objective.component_scale
    components = {
        group: scale * sum(objective.weight(group) * terms[name] for name in names)
        for group, names in LOSS_GROUPS.items()
    }
    components["pitch_in_self"] = (
        scale * objective.weight("self") * terms["pitch_component"]
    )
    components["continuation"] = (
        scale * objective.weight("continuation") * terms["continuation"]
    )
    components["kl"] = objective.weight("kl") * terms["kl_loss"]
    places = {
        "ray_encoder": _parameters(model, RAY_ENCODER_PREFIXES),
        "rssm_core": _parameters(model, CORE_PREFIXES),
    }
    out = {}
    for name, loss in components.items():
        flat = [*places["ray_encoder"], *places["rssm_core"], s]
        gradients = torch.autograd.grad(
            loss, flat, retain_graph=True, allow_unused=True
        )
        first = len(places["ray_encoder"])
        second = first + len(places["rssm_core"])
        out[name] = {
            "loss": float(loss.detach()),
            "ray_encoder": _norm(gradients[:first]),
            "rssm_core": _norm(gradients[first:second]),
            "latent_state": _norm(gradients[second:]),
        }
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return out
