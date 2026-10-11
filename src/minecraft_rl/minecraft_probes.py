"""Linear probes for the information that an encoder stage keeps.

A probe fits a ridge regression from the features of one stage to a target
value, such as the pitch, on fixed training windows, and reports the mean
absolute error on other fixed windows. If a linear map can recover the
target from the features, the next layer can use it too. If it cannot, the
information is lost or hidden in a form that a single layer cannot read.

`stage_features` collects the features of the observation path of the
Minecraft RSSM at these stages, in order:

- `self_input`: the normalized self-state input of the self encoder.
- `self_output`: the output of the self encoder.
- `fusion_input`: the concatenated ray and self features, before the last
  encoder layer.
- `fusion_preactivation`: the last encoder layer before its activation.
- `fusion_output`: the output of the shared layer. Without a shared layer
  (fusion "late"), the three fusion stages repeat the joint features.
- `encoder_output`: e, what the posterior gets from the observation. With
  fusion "self_skip", e is the shared output and the skip projection.
- `posterior_input`: [h, e], the input of the posterior network.
- `posterior_hidden`: the hidden layer of the posterior network.
- `z`: the most likely posterior state.

The probe standardizes each feature on the fit set, drops features without
variance, and selects the ridge strength by a fixed 4-fold split of the fit
set. Model parameters never change.
"""

import torch

from minecraft_rl.minecraft_world_model import MinecraftRSSM

STAGES = (
    "self_input",
    "self_output",
    "fusion_input",
    "fusion_preactivation",
    "fusion_output",
    "encoder_output",
    "posterior_input",
    "posterior_hidden",
    "z",
)
RIDGE_STRENGTHS = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
FOLDS = 4


@torch.no_grad()
def stage_features(model: MinecraftRSSM, batch) -> dict[str, torch.Tensor]:
    """Features (windows, L + 1, size) of every stage in `STAGES`."""
    model.eval()
    o = batch.observations
    lead = o["health"].shape
    encoder = model.encoder
    taps = encoder.taps(o)
    mode = model.mode_latents
    model.mode_latents = True
    filtered = model.filter(o, batch.actions, torch.Generator().manual_seed(0))
    model.mode_latents = mode
    embedded = filtered["embedded"]
    posterior_input = torch.cat([filtered["h"], embedded], -1)
    hidden = model.posterior_net[1](model.posterior_net[0](posterior_input))
    out = {name: value.reshape(*lead, -1) for name, value in taps.items()}
    out["posterior_input"] = posterior_input
    out["posterior_hidden"] = hidden
    out["z"] = filtered["z"]
    return out


def _standardize(fit: torch.Tensor, test: torch.Tensor):
    keep = fit.std(0) > 1e-6
    fit, test = fit[:, keep], test[:, keep]
    mean, std = fit.mean(0), fit.std(0)
    return (fit - mean) / std, (test - mean) / std, int(keep.sum())


def _ridge(x: torch.Tensor, y: torch.Tensor, strength: float) -> torch.Tensor:
    x = torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], 1)
    eye = torch.eye(x.shape[1], dtype=x.dtype)
    eye[-1, -1] = 0.0
    return torch.linalg.solve(x.T @ x + strength * eye, x.T @ y)


def _predict(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], 1) @ w


def probe(
    fit_x: torch.Tensor, fit_y: torch.Tensor, test_x: torch.Tensor, test_y: torch.Tensor
) -> dict[str, float]:
    """Ridge probe from features (n, d) to a target (n,). Returns the test
    mean absolute error, the chosen strength and the features used."""
    fit_x, test_x, used = _standardize(fit_x.double(), test_x.double())
    fit_y, test_y = fit_y.double(), test_y.double()
    folds = torch.arange(len(fit_x)) % FOLDS
    best, best_error = RIDGE_STRENGTHS[0], float("inf")
    for strength in RIDGE_STRENGTHS:
        errors = []
        for k in range(FOLDS):
            train, held = folds != k, folds == k
            w = _ridge(fit_x[train], fit_y[train], strength)
            errors.append((_predict(fit_x[held], w) - fit_y[held]).abs().mean())
        error = float(torch.stack(errors).mean())
        if error < best_error:
            best, best_error = strength, error
    w = _ridge(fit_x, fit_y, best)
    return {
        "mae": float((_predict(test_x, w) - test_y).abs().mean()),
        "strength": best,
        "features": used,
    }


def stage_probes(
    model: MinecraftRSSM, fit_batch, test_batch, target: str = "pitch"
) -> dict[str, dict[str, float]]:
    """Probe `target` of the same state at every stage. Windows of the fit
    and the test batch must not overlap."""
    fit, test = stage_features(model, fit_batch), stage_features(model, test_batch)
    fit_y = fit_batch.observations[target].flatten()
    test_y = test_batch.observations[target].flatten()
    out = {
        stage: probe(fit[stage].flatten(0, 1), fit_y, test[stage].flatten(0, 1), test_y)
        for stage in STAGES
    }
    out["mean_predictor"] = {
        "mae": float((test_y - fit_y.mean()).abs().mean()),
        "strength": 0.0,
        "features": 0,
    }
    return out


@torch.no_grad()
def saturation(model: MinecraftRSSM, batch) -> dict[str, float]:
    """The share of encoder outputs in the flat negative part of the ELU
    (below -0.95), and the share of pre-activations below -3."""
    taps = model.encoder.taps(batch.observations)
    if model.encoder.out is None:
        return {
            "fusion_output_below_minus_0_95": None,
            "fusion_preactivation_below_minus_3": None,
        }
    return {
        "fusion_output_below_minus_0_95": float(
            (taps["fusion_output"] < -0.95).float().mean()
        ),
        "fusion_preactivation_below_minus_3": float(
            (taps["fusion_preactivation"] < -3.0).float().mean()
        ),
    }
