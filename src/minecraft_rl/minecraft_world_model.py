"""Stage 3 RSSM world model of structured Minecraft observations.

See docs/stage3-minecraft-world-model.md. The recurrent core is the Stage 2G
RSSM (`rssm.py`): deterministic state h, 8 x 4... categorical state z, prior
p(z | h), posterior q(z | h, e) with e the encoded observation, the same KL
loss with free nats 1.0, and imagination that advances h and samples z from
the prior without decoding observations. Only the input and output layers
are new:

- `ObservationEncoder` turns one observation into a vector e. The 25 x 33 ray
  grid becomes per-ray features (a ray-kind embedding, a learned embedding of
  the hit type within its kind, and the normalized distance), followed by a
  small convolutional network. The self state uses normalized scalars and
  learned item and effect embeddings.
- `ActionEncoder` turns the factorized action (9 buttons, 2 camera deltas,
  hotbar choice) into a vector.
- `ObservationDecoder` predicts every observation field from s = [h, z] with
  a loss that fits its type: cross-entropy for categorical fields, mean
  squared error for normalized continuous fields, and masks where a field
  has no meaning (no hit type on an empty ray).

All embeddings start from random initialization. Ids are categorical
indices into embedding tables, never numbers.
"""

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from minecraft_rl.minecraft_dataset import ACTION_BUTTONS
from minecraft_rl.minecraft_interface import (
    ARMOR_SLOTS,
    EFFECT_SLOTS,
    HOTBAR_SLOTS,
    INVENTORY_SLOTS,
    MAX_CAMERA_DELTA_DEGREES,
    RAY_KIND_BLOCK,
    RAY_KIND_ENTITY,
    RAY_KIND_FLUID,
    RAY_KINDS,
)
from minecraft_rl.rssm import (
    categorical_entropy,
    categorical_kl_per_variable,
    kl_loss,
    sample_one_hot,
)

# Normalization constants (docs/decisions/observation.md).
HEALTH_SCALE = 20.0
FOOD_SCALE = 20.0
AIR_SCALE = 10.0
ARMOR_SCALE = 20.0
PITCH_SCALE = 90.0
XP_LEVEL_LOG_SCALE = math.log1p(30.0)
COUNT_LOG_SCALE = math.log1p(64.0)
EFFECT_SECONDS_LOG_SCALE = math.log1p(600.0)

SCALARS = (
    "health",
    "max_health",
    "absorption",
    "food",
    "air",
    "armor",
    "xp_level",
    "xp_progress",
    "pitch",
)


@dataclass(frozen=True)
class Vocabulary:
    rows: int
    columns: int
    max_distance: float
    block_types: int
    fluid_types: int
    entity_types: int
    item_types: int
    effect_types: int

    @property
    def rays(self) -> int:
        return self.rows * self.columns


@dataclass(frozen=True)
class ModelConfig:
    """Stage 2G core sizes, scaled moderately for the larger observation."""

    hidden: int = 256
    latent_variables: int = 16
    latent_classes: int = 16
    embed_dim: int = 256
    type_embedding: int = 8
    item_embedding: int = 16
    effect_embedding: int = 8
    conv_channels: int = 32
    kl_prior_scale: float = 1.0
    kl_posterior_scale: float = 1.0
    free_nats: float = 1.0

    def to_json(self) -> dict:
        return asdict(self)


def observation_tensors(arrays: dict, prefix: str = "obs_") -> dict[str, torch.Tensor]:
    """Model inputs from dataset arrays with any leading shape (..., field)."""
    t = {
        name[len(prefix) :]: torch.as_tensor(value)
        for name, value in arrays.items()
        if name.startswith(prefix)
    }
    return {
        "ray_kind": t["ray_kind"].long(),
        "ray_type": t["ray_type"].long(),
        "ray_distance": t["ray_distance"].float(),
        "health": t["health"].float(),
        "max_health": t["max_health"].float(),
        "absorption": t["absorption"].float(),
        "food": t["food"].float(),
        "air": t["air_bubbles"].float(),
        "armor": t["armor"].float(),
        "xp_level": t["xp_level"].float(),
        "xp_progress": t["xp_progress"].float(),
        "pitch": t["pitch"].float(),
        "selected_slot": t["selected_slot"].long(),
        "inventory_item": t["inventory_item"].long(),
        "inventory_count": t["inventory_count"].float(),
        "inventory_durability": t["inventory_durability"].float(),
        "armor_item": t["armor_item"].long(),
        "armor_durability": t["armor_durability"].float(),
        "offhand_item": t["offhand_item"].long(),
        "offhand_count": t["offhand_count"].float(),
        "offhand_durability": t["offhand_durability"].float(),
        "effect_type": t["effect_type"].long(),
        "effect_amplifier": t["effect_amplifier"].float(),
        "effect_seconds": t["effect_seconds"].float(),
    }


def normalized_scalars(o: dict[str, torch.Tensor]) -> torch.Tensor:
    """The scalar self state (..., 9), each about in [0, 1] or [-1, 1]."""
    return torch.stack(
        [
            o["health"] / HEALTH_SCALE,
            o["max_health"] / HEALTH_SCALE,
            o["absorption"] / HEALTH_SCALE,
            o["food"] / FOOD_SCALE,
            o["air"] / AIR_SCALE,
            o["armor"] / ARMOR_SCALE,
            torch.log1p(o["xp_level"]) / XP_LEVEL_LOG_SCALE,
            o["xp_progress"],
            o["pitch"] / PITCH_SCALE,
        ],
        -1,
    )


def normalized_counts(count: torch.Tensor) -> torch.Tensor:
    return torch.log1p(count) / COUNT_LOG_SCALE


def normalized_effect_seconds(
    seconds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    infinite = (seconds < 0).float()
    finite = torch.log1p(seconds.clamp(min=0)) / EFFECT_SECONDS_LOG_SCALE
    return finite, infinite


class RayTypeEmbedding(nn.Module):
    """One learned embedding table per ray kind. A ray of kind k looks up its
    type in table k; an empty ray uses a zero vector."""

    def __init__(self, vocabulary: Vocabulary, size: int) -> None:
        super().__init__()
        self.block = nn.Embedding(vocabulary.block_types, size)
        self.fluid = nn.Embedding(vocabulary.fluid_types, size)
        self.entity = nn.Embedding(vocabulary.entity_types, size)

    def forward(self, kind: torch.Tensor, type_id: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(*kind.shape, self.block.embedding_dim, device=kind.device)
        for table, value in (
            (self.block, RAY_KIND_BLOCK),
            (self.fluid, RAY_KIND_FLUID),
            (self.entity, RAY_KIND_ENTITY),
        ):
            mask = kind == value
            if mask.any():
                out[mask] = table(type_id[mask].clamp(max=table.num_embeddings - 1))
        return out


class ObservationEncoder(nn.Module):
    def __init__(self, vocabulary: Vocabulary, config: ModelConfig) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        self.kind_embedding = nn.Embedding(RAY_KINDS, 4)
        self.type_embedding = RayTypeEmbedding(vocabulary, config.type_embedding)
        channels = 4 + config.type_embedding + 1
        c = config.conv_channels
        self.conv = nn.Sequential(
            nn.Conv2d(channels, c, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(c, 2 * c, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(2 * c, 2 * c, 3, stride=2, padding=1),
            nn.ELU(),
        )
        with torch.no_grad():
            conv_out = self.conv(
                torch.zeros(1, channels, vocabulary.rows, vocabulary.columns)
            ).numel()
        self.item_embedding = nn.Embedding(vocabulary.item_types, config.item_embedding)
        self.effect_embedding = nn.Embedding(
            vocabulary.effect_types + 1, config.effect_embedding
        )
        self.slot_embedding = nn.Embedding(HOTBAR_SLOTS, 8)
        stack = config.item_embedding + 2
        self_size = (
            len(SCALARS)
            + 8
            + (INVENTORY_SLOTS + ARMOR_SLOTS + 1) * stack
            + EFFECT_SLOTS * (config.effect_embedding + 3)
        )
        self.self_net = nn.Sequential(nn.Linear(self_size, config.embed_dim), nn.ELU())
        self.out = nn.Sequential(
            nn.Linear(conv_out + config.embed_dim, config.embed_dim), nn.ELU()
        )

    def forward(self, o: dict[str, torch.Tensor]) -> torch.Tensor:
        lead = o["health"].shape
        v = self.vocabulary
        kind = o["ray_kind"].reshape(-1, v.rays)
        rays = torch.cat(
            [
                self.kind_embedding(kind),
                self.type_embedding(kind, o["ray_type"].reshape(-1, v.rays)),
                (o["ray_distance"].reshape(-1, v.rays) / v.max_distance).unsqueeze(-1),
            ],
            -1,
        )
        grid = rays.reshape(-1, v.rows, v.columns, rays.shape[-1]).permute(0, 3, 1, 2)
        visual = self.conv(grid).flatten(1)

        def stacks(item, count, durability):
            return torch.cat(
                [
                    self.item_embedding(item),
                    normalized_counts(count).unsqueeze(-1),
                    durability.unsqueeze(-1),
                ],
                -1,
            ).flatten(-2)

        seconds, infinite = normalized_effect_seconds(o["effect_seconds"])
        effects = torch.cat(
            [
                self.effect_embedding(o["effect_type"]),
                (o["effect_amplifier"] / 4.0).unsqueeze(-1),
                seconds.unsqueeze(-1),
                infinite.unsqueeze(-1),
            ],
            -1,
        ).flatten(-2)
        armor_count = (o["armor_item"] > 0).float()
        self_state = torch.cat(
            [
                normalized_scalars(o),
                self.slot_embedding(o["selected_slot"]),
                stacks(
                    o["inventory_item"], o["inventory_count"], o["inventory_durability"]
                ),
                stacks(o["armor_item"], armor_count, o["armor_durability"]),
                stacks(
                    o["offhand_item"].unsqueeze(-1),
                    o["offhand_count"].unsqueeze(-1),
                    o["offhand_durability"].unsqueeze(-1),
                ),
                effects,
            ],
            -1,
        ).reshape(-1, self.self_net[0].in_features)
        joint = torch.cat([visual, self.self_net(self_state)], -1)
        return self.out(joint).reshape(*lead, -1)


ACTION_SIZE = len(ACTION_BUTTONS) + 2 + HOTBAR_SLOTS + 1


def action_tensor(
    buttons: torch.Tensor, camera: torch.Tensor, hotbar: torch.Tensor
) -> torch.Tensor:
    """The factorized action as a vector (..., 21): buttons, camera deltas
    divided by 45 degrees, and a one-hot of the hotbar choice with slot 9
    meaning keep."""
    choice = torch.where(
        hotbar < 0, torch.full_like(hotbar, HOTBAR_SLOTS), hotbar
    ).long()
    return torch.cat(
        [
            buttons.float(),
            camera.float() / MAX_CAMERA_DELTA_DEGREES,
            nn.functional.one_hot(choice, HOTBAR_SLOTS + 1).float(),
        ],
        -1,
    )


class ObservationDecoder(nn.Module):
    """Predicts every observation field from s = [h, z]."""

    def __init__(
        self, vocabulary: Vocabulary, config: ModelConfig, features: int
    ) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        c = config.conv_channels
        self.grid_rows = math.ceil(vocabulary.rows / 8)
        self.grid_columns = math.ceil(vocabulary.columns / 8)
        self.grid = nn.Linear(features, 2 * c * self.grid_rows * self.grid_columns)
        self.deconv = nn.Sequential(
            nn.ELU(),
            nn.ConvTranspose2d(2 * c, 2 * c, 4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(2 * c, c, 4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(c, c, 4, stride=2, padding=1),
            nn.ELU(),
        )
        self.ray_kind = nn.Conv2d(c, RAY_KINDS, 1)
        self.ray_block = nn.Conv2d(c, vocabulary.block_types, 1)
        self.ray_fluid = nn.Conv2d(c, vocabulary.fluid_types, 1)
        self.ray_entity = nn.Conv2d(c, vocabulary.entity_types, 1)
        self.ray_distance = nn.Conv2d(c, 1, 1)
        hidden = config.embed_dim
        self.self_trunk = nn.Sequential(nn.Linear(features, hidden), nn.ELU())
        self.scalars = nn.Linear(hidden, len(SCALARS))
        self.selected_slot = nn.Linear(hidden, HOTBAR_SLOTS)
        self.inventory_item = nn.Linear(hidden, INVENTORY_SLOTS * vocabulary.item_types)
        self.inventory_count = nn.Linear(hidden, INVENTORY_SLOTS)
        self.armor_item = nn.Linear(hidden, ARMOR_SLOTS * vocabulary.item_types)
        self.offhand_item = nn.Linear(hidden, vocabulary.item_types)
        self.effect_type = nn.Linear(
            hidden, EFFECT_SLOTS * (vocabulary.effect_types + 1)
        )

    def forward(self, s: torch.Tensor) -> dict[str, torch.Tensor]:
        lead = s.shape[:-1]
        flat = s.reshape(-1, s.shape[-1])
        v = self.vocabulary
        grid = self.grid(flat).reshape(
            flat.shape[0], -1, self.grid_rows, self.grid_columns
        )
        grid = self.deconv(grid)[:, :, : v.rows, : v.columns]

        def per_ray(head):
            out = head(grid).permute(0, 2, 3, 1).reshape(flat.shape[0], v.rays, -1)
            return out.reshape(*lead, v.rays, -1)

        trunk = self.self_trunk(flat)
        return {
            "ray_kind": per_ray(self.ray_kind),
            "ray_block": per_ray(self.ray_block),
            "ray_fluid": per_ray(self.ray_fluid),
            "ray_entity": per_ray(self.ray_entity),
            "ray_distance": per_ray(self.ray_distance).squeeze(-1),
            "scalars": self.scalars(trunk).reshape(*lead, -1),
            "selected_slot": self.selected_slot(trunk).reshape(*lead, -1),
            "inventory_item": self.inventory_item(trunk).reshape(
                *lead, INVENTORY_SLOTS, v.item_types
            ),
            "inventory_count": self.inventory_count(trunk).reshape(
                *lead, INVENTORY_SLOTS
            ),
            "armor_item": self.armor_item(trunk).reshape(
                *lead, ARMOR_SLOTS, v.item_types
            ),
            "offhand_item": self.offhand_item(trunk).reshape(*lead, v.item_types),
            "effect_type": self.effect_type(trunk).reshape(
                *lead, EFFECT_SLOTS, v.effect_types + 1
            ),
        }


def _cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-element cross-entropy with the class axis last."""
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none"
    ).reshape(target.shape)


def reconstruction_losses(
    prediction: dict[str, torch.Tensor], o: dict[str, torch.Tensor], max_distance: float
) -> dict[str, torch.Tensor]:
    """Per-state loss of each field group, shape (...). Ray terms average over
    rays. A hit-type loss counts only rays of that kind."""
    kind = o["ray_kind"]
    out = {"ray_kind": _cross_entropy(prediction["ray_kind"], kind).mean(-1)}
    for name, value in (
        ("ray_block", RAY_KIND_BLOCK),
        ("ray_fluid", RAY_KIND_FLUID),
        ("ray_entity", RAY_KIND_ENTITY),
    ):
        mask = (kind == value).float()
        target = torch.where(
            kind == value, o["ray_type"], torch.zeros_like(o["ray_type"])
        )
        target = target.clamp(max=prediction[name].shape[-1] - 1)
        out[name] = (_cross_entropy(prediction[name], target) * mask).sum(
            -1
        ) / mask.shape[-1]
    distance = o["ray_distance"] / max_distance
    out["ray_distance"] = ((prediction["ray_distance"] - distance) ** 2).mean(-1)
    out["scalars"] = ((prediction["scalars"] - normalized_scalars(o)) ** 2).mean(-1)
    out["selected_slot"] = _cross_entropy(
        prediction["selected_slot"], o["selected_slot"]
    )
    out["inventory_item"] = _cross_entropy(
        prediction["inventory_item"], o["inventory_item"]
    ).mean(-1)
    out["inventory_count"] = (
        (prediction["inventory_count"] - normalized_counts(o["inventory_count"])) ** 2
    ).mean(-1)
    out["armor_item"] = _cross_entropy(prediction["armor_item"], o["armor_item"]).mean(
        -1
    )
    out["offhand_item"] = _cross_entropy(prediction["offhand_item"], o["offhand_item"])
    out["effect_type"] = _cross_entropy(
        prediction["effect_type"], o["effect_type"]
    ).mean(-1)
    return out


class MinecraftRSSM(nn.Module):
    """The Stage 2G RSSM core with Minecraft encoders and decoder.

    h_t = GRUCell(h_{t-1}, [z_{t-1}, action_{t-1}]), h_0 = 0
    prior p(z_t | h_t), posterior q(z_t | h_t, e_t), e_t = encoder(o_t)
    decoder(s_t) predicts o_t; continue_head(s_t) predicts that the
    transition into s_t did not end the episode.
    """

    def __init__(self, vocabulary: Vocabulary, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.vocabulary = vocabulary
        self.hidden = config.hidden
        self.variables = config.latent_variables
        self.classes = config.latent_classes
        self.latent_size = self.variables * self.classes
        features = self.hidden + self.latent_size
        self.encoder = ObservationEncoder(vocabulary, config)
        self.action_net = nn.Sequential(nn.Linear(ACTION_SIZE, 64), nn.ELU())
        self.cell = nn.GRUCell(self.latent_size + 64, self.hidden)
        self.prior_net = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.ELU(),
            nn.Linear(self.hidden, self.latent_size),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(self.hidden + config.embed_dim, self.hidden),
            nn.ELU(),
            nn.Linear(self.hidden, self.latent_size),
        )
        self.decoder = ObservationDecoder(vocabulary, config, features)
        self.continue_head = nn.Linear(features, 1)
        self.mode_latents = False

    def prior_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.prior_net(h).unflatten(-1, (self.variables, self.classes))

    def posterior_logits(self, h: torch.Tensor, embedded: torch.Tensor) -> torch.Tensor:
        return self.posterior_net(torch.cat([h, embedded], -1)).unflatten(
            -1, (self.variables, self.classes)
        )

    def transition(
        self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return self.cell(torch.cat([z, self.action_net(action)], -1), h)

    def sample(
        self,
        logits: torch.Tensor,
        generator: torch.Generator | None,
        straight_through: bool,
    ) -> torch.Tensor:
        mode = self.mode_latents and not straight_through
        return sample_one_hot(logits, generator, straight_through, mode).flatten(-2)

    def filter(
        self,
        o: dict[str, torch.Tensor],
        actions: torch.Tensor,
        generator: torch.Generator | None,
        straight_through: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Posterior states for observations (batch, L + 1, ...) and actions
        (batch, L, 21): h, z, prior and posterior logits over L + 1 states."""
        embedded = self.encoder(o)
        batch, length = embedded.shape[:2]
        h = torch.zeros(batch, self.hidden, device=embedded.device)
        hs, zs, priors, posteriors = [], [], [], []
        for k in range(length):
            if k > 0:
                h = self.transition(h, zs[-1], actions[:, k - 1])
            prior = self.prior_logits(h)
            posterior = self.posterior_logits(h, embedded[:, k])
            z = self.sample(posterior, generator, straight_through)
            hs.append(h)
            zs.append(z)
            priors.append(prior)
            posteriors.append(posterior)
        return {
            "h": torch.stack(hs, 1),
            "z": torch.stack(zs, 1),
            "prior": torch.stack(priors, 1),
            "posterior": torch.stack(posteriors, 1),
        }

    def losses(
        self,
        o: dict[str, torch.Tensor],
        actions: torch.Tensor,
        continues: torch.Tensor,
        generator: torch.Generator | None,
    ) -> dict[str, torch.Tensor]:
        """Negative evidence lower bound per state, averaged over the batch.
        `continues` (batch, L) is 0 for the transition that ended an episode."""
        filtered = self.filter(o, actions, generator, straight_through=True)
        s = torch.cat([filtered["h"], filtered["z"]], -1)
        terms = reconstruction_losses(self.decoder(s), o, self.vocabulary.max_distance)
        continuation = nn.functional.binary_cross_entropy_with_logits(
            self.continue_head(s[:, 1:]).squeeze(-1), continues, reduction="none"
        )
        posterior, prior = filtered["posterior"], filtered["prior"]
        regularizer = kl_loss(
            posterior,
            prior,
            self.config.kl_prior_scale,
            self.config.kl_posterior_scale,
            self.config.free_nats,
        )
        out = {name: value.mean() for name, value in terms.items()}
        out["continuation"] = continuation.mean()
        raw_kl = categorical_kl_per_variable(posterior, prior).sum(-1)
        out["kl"] = raw_kl.mean()
        out["kl_loss"] = regularizer.mean()
        out["kl_below_free_nats"] = (
            (raw_kl < self.config.free_nats).float().mean().detach()
        )
        out["reconstruction"] = sum(out[name] for name in terms)
        out["total"] = out["reconstruction"] + out["continuation"] + out["kl_loss"]
        return out

    def imagine(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        actions: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Latent imagination from the state (h, z) with the given actions
        (batch, K, 21). It advances h and samples z from the prior, and never
        decodes or re-encodes an observation. Returns s (batch, K, H + V K)."""
        states = []
        for k in range(actions.shape[1]):
            h = self.transition(h, z, actions[:, k])
            z = self.sample(self.prior_logits(h), generator, straight_through=False)
            states.append(torch.cat([h, z], -1))
        return torch.stack(states, 1)


def latent_statistics(
    filtered: dict[str, torch.Tensor], free_nats: float
) -> dict[str, float]:
    """Stage 2G latent diagnostics over all states of a batch."""
    posterior, prior = filtered["posterior"], filtered["prior"]
    per_variable = categorical_kl_per_variable(posterior, prior)
    kl = per_variable.sum(-1)
    classes = posterior.shape[-1]
    winners = (
        nn.functional.one_hot(posterior.argmax(-1), classes).flatten(0, -3).float()
    )
    frequency = winners.mean(0)
    perplexity = torch.exp(-(frequency * frequency.clamp(min=1e-12).log()).sum(-1))
    agreement = (prior.argmax(-1) == posterior.argmax(-1)).float()
    regularizer = kl.clamp(min=free_nats)
    return {
        "kl_mean": kl.mean().item(),
        "kl_p50": kl.median().item(),
        "kl_p90": kl.quantile(0.9).item() if kl.numel() < 16_000_000 else float("nan"),
        "kl_posterior_part_effective": regularizer.mean().item(),
        "below_free_nats_fraction": (kl < free_nats).float().mean().item(),
        "kl_per_variable": per_variable.flatten(0, -2).mean(0).tolist(),
        "active_variables": int((per_variable.flatten(0, -2).mean(0) > 0.01).sum()),
        "prior_entropy": categorical_entropy(prior).mean().item(),
        "posterior_entropy": categorical_entropy(posterior).mean().item(),
        "maximum_entropy": posterior.shape[-2] * math.log(classes),
        "classes_used": int(winners.amax(0).sum()),
        "classes_total": int(posterior.shape[-2] * classes),
        "class_perplexity_mean": perplexity.mean().item(),
        "prior_posterior_agreement": agreement.mean().item(),
    }
