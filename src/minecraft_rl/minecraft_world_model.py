"""Stage 3 RSSM world model of structured Minecraft observations.

See docs/stage3-minecraft-world-model.md. The recurrent core is the Stage 2G
RSSM (`rssm.py`): a deterministic state h, a categorical state z, a prior
p(z | h), a posterior q(z | h, e) with e the encoded observation, the same KL
loss with free nats 1.0, and imagination that advances h and samples z from
the prior without decoding observations. Only the input and output layers
are new:

- `ObservationEncoder` turns one observation into a vector e. Each ray of
  the 25 x 33 grid gets a learned embedding of its ray class and its
  normalized distance. A ray class is one value for "no hit" plus one value
  per hit type of each kind (block, fluid, entity), so kind and type form
  one categorical variable. A ray-grid layer turns the grid into a vector.
  The self state uses normalized scalars and learned item, effect and slot
  embeddings.
- The action is a vector of 9 buttons, 2 camera deltas and a one-hot hotbar
  choice (`action_tensor`), followed by one learned layer.
- `ObservationDecoder` predicts every observation field from s = [h, z].
  The loss of one observation is its negative log-likelihood, summed over
  all of its elements: cross-entropy for categorical fields and squared
  error (a Gaussian with unit variance) for normalized continuous fields. A
  field without meaning has a mask: no distance for a ray without a hit, and
  no count, durability or effect timer for an empty slot.

Two ray-grid layers exist. `conv` uses strided convolutions and transposed
convolutions. `patch` cuts the grid into non-overlapping 4 x 4 patches and
uses one linear layer per patch, the same weights for every patch. A patch
layer is a convolution with kernel size equal to stride, and needs much less
computation (docs/stage3-minecraft-world-model.md has the measurement).

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
    RAY_KIND_NONE,
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
AMPLIFIER_SCALE = 4.0
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

# Field groups of the reconstruction loss, in report order.
LOSS_TERMS = (
    "ray_class",
    "ray_distance",
    "scalars",
    "selected_slot",
    "inventory_item",
    "inventory_count",
    "inventory_durability",
    "armor_item",
    "armor_durability",
    "offhand_item",
    "offhand_count",
    "offhand_durability",
    "effect_type",
    "effect_amplifier",
    "effect_seconds",
    "effect_infinite",
)


@dataclass(frozen=True)
class Vocabulary:
    """Grid layout and category counts. Each count includes every value the
    model can see, so a compact vocabulary passes its unknown index too.
    Item index 0 and effect index 0 mean an empty slot."""

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

    @property
    def ray_classes(self) -> int:
        return 1 + self.block_types + self.fluid_types + self.entity_types

    def kind_offsets(self) -> dict[int, int]:
        """First ray class of each hit kind. Class 0 means no hit."""
        return {
            RAY_KIND_BLOCK: 1,
            RAY_KIND_FLUID: 1 + self.block_types,
            RAY_KIND_ENTITY: 1 + self.block_types + self.fluid_types,
        }

    def class_kinds(self) -> torch.Tensor:
        """The ray kind of every ray class, shape (ray_classes,)."""
        kinds = torch.full((self.ray_classes,), RAY_KIND_NONE, dtype=torch.long)
        offsets = self.kind_offsets()
        sizes = {
            RAY_KIND_BLOCK: self.block_types,
            RAY_KIND_FLUID: self.fluid_types,
            RAY_KIND_ENTITY: self.entity_types,
        }
        for kind, first in offsets.items():
            kinds[first : first + sizes[kind]] = kind
        return kinds


def ray_class(
    kind: torch.Tensor, type_id: torch.Tensor, vocabulary: Vocabulary
) -> torch.Tensor:
    """The joint ray class of each ray: 0 for no hit, otherwise the offset of
    the kind plus the type index within the kind."""
    out = torch.zeros_like(kind, dtype=torch.long)
    for value, first in vocabulary.kind_offsets().items():
        out = torch.where(kind == value, first + type_id.long(), out)
    return out


@dataclass(frozen=True)
class ModelConfig:
    """Stage 2G core sizes, scaled moderately for the larger observation."""

    hidden: int = 256
    latent_variables: int = 16
    latent_classes: int = 16
    embed_dim: int = 256
    action_dim: int = 64
    ray_embedding: int = 8
    item_embedding: int = 16
    effect_embedding: int = 8
    ray_layer: str = "patch"
    encoder_channels: int = 16
    decoder_channels: int = 16
    patch: int = 4
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
    """A finite timer in [0, 1] and a flag for an infinite effect (-1)."""
    infinite = (seconds < 0).float()
    finite = torch.log1p(seconds.clamp(min=0)) / EFFECT_SECONDS_LOG_SCALE
    return finite, infinite


def _patch_grid(rows: int, columns: int, patch: int) -> tuple[int, int]:
    return math.ceil(rows / patch), math.ceil(columns / patch)


class PatchEncoder(nn.Module):
    """(N, C, R, W) -> (N, patches * channels): one shared linear layer per
    non-overlapping patch, after zero padding to whole patches."""

    def __init__(self, inputs: int, channels: int, rows: int, columns: int, patch: int):
        super().__init__()
        self.rows, self.columns, self.patch = rows, columns, patch
        self.grid_rows, self.grid_columns = _patch_grid(rows, columns, patch)
        self.linear = nn.Linear(inputs * patch * patch, channels)
        self.outputs = self.grid_rows * self.grid_columns * channels

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        p = self.patch
        grid = nn.functional.pad(
            grid,
            (
                0,
                self.grid_columns * p - self.columns,
                0,
                self.grid_rows * p - self.rows,
            ),
        )
        n, c = grid.shape[:2]
        patches = (
            grid.reshape(n, c, self.grid_rows, p, self.grid_columns, p)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(n, self.grid_rows * self.grid_columns, c * p * p)
        )
        return nn.functional.elu(self.linear(patches)).flatten(1)


class ConvEncoder(nn.Module):
    def __init__(self, inputs: int, channels: int, rows: int, columns: int):
        super().__init__()
        c = channels
        self.net = nn.Sequential(
            nn.Conv2d(inputs, c, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(c, 2 * c, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(2 * c, 2 * c, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            self.outputs = self.net(torch.zeros(1, inputs, rows, columns)).shape[1]

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        return self.net(grid)


class ObservationEncoder(nn.Module):
    def __init__(self, vocabulary: Vocabulary, config: ModelConfig) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        v = vocabulary
        self.ray_embedding = nn.Embedding(v.ray_classes, config.ray_embedding)
        inputs = config.ray_embedding + 1
        if config.ray_layer == "patch":
            self.grid = PatchEncoder(
                inputs, config.encoder_channels, v.rows, v.columns, config.patch
            )
        elif config.ray_layer == "conv":
            self.grid = ConvEncoder(inputs, config.encoder_channels, v.rows, v.columns)
        else:
            raise ValueError(f"unknown ray layer {config.ray_layer!r}")
        self.item_embedding = nn.Embedding(v.item_types, config.item_embedding)
        self.effect_embedding = nn.Embedding(v.effect_types, config.effect_embedding)
        self.slot_embedding = nn.Embedding(HOTBAR_SLOTS, 8)
        stack = config.item_embedding + 2
        self.self_size = (
            len(SCALARS)
            + 8
            + (INVENTORY_SLOTS + ARMOR_SLOTS + 1) * stack
            + EFFECT_SLOTS * (config.effect_embedding + 3)
        )
        self.self_net = nn.Sequential(
            nn.Linear(self.self_size, config.embed_dim), nn.ELU()
        )
        self.out = nn.Sequential(
            nn.Linear(self.grid.outputs + config.embed_dim, config.embed_dim), nn.ELU()
        )

    def rays(self, o: dict[str, torch.Tensor]) -> torch.Tensor:
        v = self.vocabulary
        classes = ray_class(o["ray_kind"], o["ray_type"], v).reshape(-1, v.rays)
        distance = o["ray_distance"].reshape(-1, v.rays, 1) / v.max_distance
        features = torch.cat([self.ray_embedding(classes), distance], -1)
        grid = features.reshape(-1, v.rows, v.columns, features.shape[-1])
        return self.grid(grid.permute(0, 3, 1, 2))

    def self_state(self, o: dict[str, torch.Tensor]) -> torch.Tensor:
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
                (o["effect_amplifier"] / AMPLIFIER_SCALE).unsqueeze(-1),
                seconds.unsqueeze(-1),
                infinite.unsqueeze(-1),
            ],
            -1,
        ).flatten(-2)
        armor_count = (o["armor_item"] > 0).float()
        features = torch.cat(
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
        )
        return self.self_net(features.reshape(-1, self.self_size))

    def forward(self, o: dict[str, torch.Tensor]) -> torch.Tensor:
        lead = o["health"].shape
        joint = torch.cat([self.rays(o), self.self_state(o)], -1)
        return self.out(joint).reshape(*lead, -1)


ACTION_SIZE = len(ACTION_BUTTONS) + 2 + HOTBAR_SLOTS + 1


def action_tensor(
    buttons: torch.Tensor, camera: torch.Tensor, hotbar: torch.Tensor
) -> torch.Tensor:
    """The factorized action as a vector (..., 21): buttons, camera deltas
    divided by the largest delta, and a one-hot of the hotbar choice with
    index 9 meaning keep the current slot."""
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


class PatchDecoder(nn.Module):
    """features -> (N, R, W, channels): one linear layer to a patch grid,
    then one shared linear layer that expands each patch to its rays."""

    def __init__(
        self, features: int, channels: int, rows: int, columns: int, patch: int
    ):
        super().__init__()
        self.rows, self.columns, self.patch, self.channels = (
            rows,
            columns,
            patch,
            channels,
        )
        self.grid_rows, self.grid_columns = _patch_grid(rows, columns, patch)
        self.grid = nn.Linear(features, self.grid_rows * self.grid_columns * channels)
        self.expand = nn.Linear(channels, patch * patch * channels)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        n, p, c = s.shape[0], self.patch, self.channels
        cells = nn.functional.elu(self.grid(s)).reshape(
            n, self.grid_rows * self.grid_columns, c
        )
        rays = nn.functional.elu(self.expand(cells))
        rays = (
            rays.reshape(n, self.grid_rows, self.grid_columns, p, p, c)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(n, self.grid_rows * p, self.grid_columns * p, c)
        )
        return rays[:, : self.rows, : self.columns]


class ConvDecoder(nn.Module):
    def __init__(self, features: int, channels: int, rows: int, columns: int):
        super().__init__()
        c = channels
        self.rows, self.columns = rows, columns
        self.grid_rows, self.grid_columns = _patch_grid(rows, columns, 8)
        self.grid = nn.Linear(features, 2 * c * self.grid_rows * self.grid_columns)
        self.net = nn.Sequential(
            nn.ELU(),
            nn.ConvTranspose2d(2 * c, 2 * c, 4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(2 * c, c, 4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(c, c, 4, stride=2, padding=1),
            nn.ELU(),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        grid = self.grid(s).reshape(s.shape[0], -1, self.grid_rows, self.grid_columns)
        out = self.net(grid)[:, :, : self.rows, : self.columns]
        return out.permute(0, 2, 3, 1)


class ObservationDecoder(nn.Module):
    """Predicts every observation field from s = [h, z]."""

    def __init__(
        self, vocabulary: Vocabulary, config: ModelConfig, features: int
    ) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        v, c = vocabulary, config.decoder_channels
        if config.ray_layer == "patch":
            self.grid = PatchDecoder(features, c, v.rows, v.columns, config.patch)
        elif config.ray_layer == "conv":
            self.grid = ConvDecoder(features, c, v.rows, v.columns)
        else:
            raise ValueError(f"unknown ray layer {config.ray_layer!r}")
        self.ray_head = nn.Linear(c, v.ray_classes + 1)
        hidden = config.embed_dim
        self.self_trunk = nn.Sequential(nn.Linear(features, hidden), nn.ELU())
        self.heads = nn.ModuleDict(
            {
                "scalars": nn.Linear(hidden, len(SCALARS)),
                "selected_slot": nn.Linear(hidden, HOTBAR_SLOTS),
                "inventory_item": nn.Linear(hidden, INVENTORY_SLOTS * v.item_types),
                "inventory_count": nn.Linear(hidden, INVENTORY_SLOTS),
                "inventory_durability": nn.Linear(hidden, INVENTORY_SLOTS),
                "armor_item": nn.Linear(hidden, ARMOR_SLOTS * v.item_types),
                "armor_durability": nn.Linear(hidden, ARMOR_SLOTS),
                "offhand_item": nn.Linear(hidden, v.item_types),
                "offhand_count": nn.Linear(hidden, 1),
                "offhand_durability": nn.Linear(hidden, 1),
                "effect_type": nn.Linear(hidden, EFFECT_SLOTS * v.effect_types),
                "effect_amplifier": nn.Linear(hidden, EFFECT_SLOTS),
                "effect_seconds": nn.Linear(hidden, EFFECT_SLOTS),
                "effect_infinite": nn.Linear(hidden, EFFECT_SLOTS),
            }
        )

    def forward(self, s: torch.Tensor) -> dict[str, torch.Tensor]:
        lead = s.shape[:-1]
        flat = s.reshape(-1, s.shape[-1])
        v = self.vocabulary
        rays = self.ray_head(self.grid(flat)).reshape(*lead, v.rays, -1)
        trunk = self.self_trunk(flat)
        out = {
            "ray_class": rays[..., :-1],
            "ray_distance": rays[..., -1],
        }
        shapes = {
            "inventory_item": (INVENTORY_SLOTS, v.item_types),
            "armor_item": (ARMOR_SLOTS, v.item_types),
            "effect_type": (EFFECT_SLOTS, v.effect_types),
        }
        for name, head in self.heads.items():
            value = head(trunk)
            if name in shapes:
                value = value.reshape(-1, *shapes[name])
            elif name in ("offhand_count", "offhand_durability"):
                value = value.squeeze(-1)
            out[name] = value.reshape(*lead, *value.shape[1:])
        return out


def _cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-element cross-entropy with the class axis last."""
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none"
    ).reshape(target.shape)


def targets(
    o: dict[str, torch.Tensor], vocabulary: Vocabulary
) -> dict[str, torch.Tensor]:
    """Loss targets and masks in model units. A mask is 1 where the field has
    a meaning: a hit for the ray distance, a filled slot for counts and
    durability bars, an active effect for amplifier and timer."""
    v = vocabulary
    seconds, infinite = normalized_effect_seconds(o["effect_seconds"])
    filled = (o["inventory_item"] > 0).float()
    armor = (o["armor_item"] > 0).float()
    offhand = (o["offhand_item"] > 0).float()
    effect = (o["effect_type"] > 0).float()
    return {
        "ray_class": ray_class(o["ray_kind"], o["ray_type"], v),
        "ray_distance": o["ray_distance"] / v.max_distance,
        "ray_distance_mask": (o["ray_kind"] != RAY_KIND_NONE).float(),
        "scalars": normalized_scalars(o),
        "selected_slot": o["selected_slot"],
        "inventory_item": o["inventory_item"],
        "inventory_count": normalized_counts(o["inventory_count"]),
        "inventory_count_mask": filled,
        "inventory_durability": o["inventory_durability"],
        "inventory_durability_mask": filled,
        "armor_item": o["armor_item"],
        "armor_durability": o["armor_durability"],
        "armor_durability_mask": armor,
        "offhand_item": o["offhand_item"],
        "offhand_count": normalized_counts(o["offhand_count"]),
        "offhand_count_mask": offhand,
        "offhand_durability": o["offhand_durability"],
        "offhand_durability_mask": offhand,
        "effect_type": o["effect_type"],
        "effect_amplifier": o["effect_amplifier"] / AMPLIFIER_SCALE,
        "effect_amplifier_mask": effect,
        "effect_seconds": seconds,
        "effect_seconds_mask": effect * (1.0 - infinite),
        "effect_infinite": infinite,
        "effect_infinite_mask": effect,
    }


CATEGORICAL_TERMS = (
    "ray_class",
    "selected_slot",
    "inventory_item",
    "armor_item",
    "offhand_item",
    "effect_type",
)
BINARY_TERMS = ("effect_infinite",)

# Loss groups for component weights and the gradient diagnostics.
LOSS_GROUPS = {
    "ray_class": ("ray_class",),
    "ray_distance": ("ray_distance",),
    "self": (
        "scalars",
        "selected_slot",
        "effect_type",
        "effect_amplifier",
        "effect_seconds",
        "effect_infinite",
    ),
    "inventory": (
        "inventory_item",
        "inventory_count",
        "inventory_durability",
        "armor_item",
        "armor_durability",
        "offhand_item",
        "offhand_count",
        "offhand_durability",
    ),
}
GROUP_OF_TERM = {term: group for group, terms in LOSS_GROUPS.items() for term in terms}
PITCH_INDEX = SCALARS.index("pitch")

# How each loss term turns its element losses into one number.
# - "sum": sum over the valid elements of one state, then the mean over the
#   states. This is the log likelihood of the whole observation. A field
#   with more elements gets a larger weight: 825 ray-class terms against one
#   pitch value.
# - "ray_mean": as "sum", but the ray-class term is the mean over the rays.
# - "semantic_mean": every term is the mean over its own valid elements in
#   the batch. The size of a field then no longer sets its weight. The 9
#   scalars are 9 different quantities (health, food, pitch, ...), so each
#   scalar is its own component: the "scalars" term is the sum of 9 means.
# - "group_mean": as "semantic_mean", then each loss group of `LOSS_GROUPS`
#   becomes the mean of its components instead of their sum. A component
#   counts only if it has a valid element in the batch, so an empty armor
#   slot adds no durability component. Each group is then one normalized
#   loss: the number of fields in a group no longer sets its weight either.
REDUCTIONS = ("sum", "ray_mean", "semantic_mean", "group_mean")
SEPARATE_COMPONENT_TERMS = ("scalars",)


@dataclass(frozen=True)
class Objective:
    """The training objective: a reduction and fixed weights per loss group
    (`LOSS_GROUPS`, "continuation" and "kl"). A group without a weight gets
    1. The KL term keeps its Stage 2G form (prior and posterior parts, free
    nats); its weight only scales the whole term.

    The weight of "kl" sets the trade-off between reconstruction and the
    information in z. Under "sum", reconstruction counts in nats per
    observation, as the KL does. A mean over elements counts in nats per
    element, so the same KL weighs about as much as a whole grid of rays.

    `component_scale` multiplies every reconstruction component and the
    continuation loss, not the KL. With "semantic_mean" and the ray count as
    the scale, the ray-class component equals the summed ray-class term of
    "sum", the KL keeps its weight relative to it, and every other
    component weighs as much as the ray grid. The overall loss scale then
    also stays that of "sum". That matters for Adam: Adam ignores the scale
    of a gradient only where the gradient is much larger than its eps.
    """

    reduction: str = "sum"
    weights: tuple[tuple[str, float], ...] = ()
    component_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.reduction not in REDUCTIONS:
            raise ValueError(f"unknown reduction {self.reduction!r}")
        for group, _ in self.weights:
            if group not in (*LOSS_GROUPS, "continuation", "kl"):
                raise ValueError(f"unknown loss group {group!r}")

    def weight(self, group: str) -> float:
        return dict(self.weights).get(group, 1.0)

    def to_json(self) -> dict:
        return {
            "reduction": self.reduction,
            "weights": dict(self.weights),
            "component_scale": self.component_scale,
        }


# The objective of the first Stage 3 runs: the plain log likelihood.
SUM_OBJECTIVE = Objective()


def element_losses(
    prediction: dict[str, torch.Tensor],
    o: dict[str, torch.Tensor],
    vocabulary: Vocabulary,
) -> dict[str, tuple[torch.Tensor, torch.Tensor | None]]:
    """Per term: the loss of every element, shape (..., elements of the
    term), and its mask of valid elements, or None if all are valid.
    Categorical terms use cross-entropy, binary terms binary cross-entropy,
    and continuous terms the squared error of values that `targets` divides
    by their fixed game range."""
    t = targets(o, vocabulary)
    lead = o["health"].dim()
    out = {}
    for name in LOSS_TERMS:
        if name in CATEGORICAL_TERMS:
            element = _cross_entropy(prediction[name], t[name])
        elif name in BINARY_TERMS:
            element = nn.functional.binary_cross_entropy_with_logits(
                prediction[name], t[name], reduction="none"
            )
        else:
            element = (prediction[name] - t[name]) ** 2
        mask = t.get(f"{name}_mask")
        shape = (*element.shape[:lead], -1)
        out[name] = (
            element.reshape(shape),
            None if mask is None else mask.reshape(shape),
        )
    return out


def reconstruction_losses(
    prediction: dict[str, torch.Tensor],
    o: dict[str, torch.Tensor],
    vocabulary: Vocabulary,
) -> dict[str, torch.Tensor]:
    """Negative log-likelihood of each field group per state, shape (...),
    summed over the elements of the group and over valid elements only."""
    out = {}
    for name, (element, mask) in element_losses(prediction, o, vocabulary).items():
        if mask is not None:
            element = element * mask
        out[name] = element.sum(-1)
    return out


def reduce_term(
    name: str, element: torch.Tensor, mask: torch.Tensor | None, reduction: str
) -> torch.Tensor:
    """The loss of one term as one number. `element` has the shape
    (..., elements of the term)."""
    if mask is not None:
        element = element * mask
    if (
        reduction in ("semantic_mean", "group_mean")
        and name not in SEPARATE_COMPONENT_TERMS
    ):
        count = (
            element.new_tensor(float(element.numel())) if mask is None else mask.sum()
        )
        return element.sum() / count.clamp(min=1.0)
    per_state = element.sum(-1)
    if reduction == "ray_mean" and name == "ray_class":
        per_state = per_state / element.shape[-1]
    return per_state.mean()


def objective_terms(
    prediction: dict[str, torch.Tensor],
    o: dict[str, torch.Tensor],
    vocabulary: Vocabulary,
    reduction: str,
) -> dict[str, torch.Tensor]:
    """Every reconstruction term reduced as `reduction` says, plus
    "pitch_component": the part of the "scalars" term that comes from the
    pitch. It is already part of "scalars" and is only for diagnostics."""
    elements = element_losses(prediction, o, vocabulary)
    out = {}
    for name, (element, mask) in elements.items():
        out[name] = reduce_term(name, element, mask, reduction)
        if name == "scalars":
            out["pitch_component"] = element[..., PITCH_INDEX].mean()
    if reduction == "group_mean":
        for terms in LOSS_GROUPS.values():
            count = max(1, sum(_components(name, *elements[name]) for name in terms))
            for name in terms:
                out[name] = out[name] / count
            if "scalars" in terms:
                out["pitch_component"] = out["pitch_component"] / count
    return out


def _components(name: str, element: torch.Tensor, mask: torch.Tensor | None) -> int:
    """The number of loss components in a term: one per scalar for the
    scalars, else one if the term has a valid element in the batch."""
    if name in SEPARATE_COMPONENT_TERMS:
        return element.shape[-1]
    return int(mask is None or bool(mask.sum() > 0))


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
        self.features = self.hidden + self.latent_size
        self.encoder = ObservationEncoder(vocabulary, config)
        self.action_net = nn.Sequential(
            nn.Linear(ACTION_SIZE, config.action_dim), nn.ELU()
        )
        self.cell = nn.GRUCell(self.latent_size + config.action_dim, self.hidden)
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
        self.decoder = ObservationDecoder(vocabulary, config, self.features)
        self.continue_head = nn.Linear(self.features, 1)
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

    def filter_embedded(
        self,
        embedded: torch.Tensor,
        actions: torch.Tensor,
        generator: torch.Generator | None,
        straight_through: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Posterior states for encoded observations (batch, L + 1, E) and
        actions (batch, L, 21): h, z, prior and posterior logits."""
        batch, length = embedded.shape[:2]
        h = torch.zeros(batch, self.hidden)
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

    def filter(
        self,
        o: dict[str, torch.Tensor],
        actions: torch.Tensor,
        generator: torch.Generator | None,
        straight_through: bool = False,
    ) -> dict[str, torch.Tensor]:
        return self.filter_embedded(
            self.encoder(o), actions, generator, straight_through
        )

    def losses(
        self,
        o: dict[str, torch.Tensor],
        actions: torch.Tensor,
        continues: torch.Tensor,
        generator: torch.Generator | None,
        objective: Objective = SUM_OBJECTIVE,
    ) -> dict[str, torch.Tensor]:
        """The training loss of a batch of windows: reconstruction terms
        reduced and weighted by `objective`, the continuation loss and the KL
        loss. `continues` (batch, L) is 0 for the transition that ended an
        episode."""
        return self.loss_graph(o, actions, continues, generator, objective)[0]

    def loss_graph(
        self,
        o: dict[str, torch.Tensor],
        actions: torch.Tensor,
        continues: torch.Tensor,
        generator: torch.Generator | None,
        objective: Objective = SUM_OBJECTIVE,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """As `losses`, and also the decoder input s = [h, z], for gradient
        diagnostics."""
        filtered = self.filter(o, actions, generator, straight_through=True)
        s = torch.cat([filtered["h"], filtered["z"]], -1)
        out = objective_terms(self.decoder(s), o, self.vocabulary, objective.reduction)
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
        out["continuation"] = continuation.mean()
        raw_kl = categorical_kl_per_variable(posterior, prior).sum(-1)
        out["kl"] = raw_kl.mean().detach()
        out["kl_loss"] = regularizer.mean()
        out["kl_below_free_nats"] = (
            (raw_kl < self.config.free_nats).float().mean().detach()
        )
        scale = objective.component_scale
        out["reconstruction"] = scale * sum(
            objective.weight(GROUP_OF_TERM[name]) * out[name] for name in LOSS_TERMS
        )
        out["total"] = (
            out["reconstruction"]
            + scale * objective.weight("continuation") * out["continuation"]
            + objective.weight("kl") * out["kl_loss"]
        )
        return out, s

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


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def latent_statistics(
    filtered: dict[str, torch.Tensor], free_nats: float
) -> dict[str, float]:
    """Stage 2G latent diagnostics over all states of a batch."""
    posterior, prior = filtered["posterior"], filtered["prior"]
    per_variable = categorical_kl_per_variable(posterior, prior)
    kl = per_variable.sum(-1).flatten()
    classes = posterior.shape[-1]
    winners = (
        nn.functional.one_hot(posterior.argmax(-1), classes).flatten(0, -3).float()
    )
    frequency = winners.mean(0)
    perplexity = torch.exp(-(frequency * frequency.clamp(min=1e-12).log()).sum(-1))
    agreement = (prior.argmax(-1) == posterior.argmax(-1)).float()
    variable_kl = per_variable.flatten(0, -2).mean(0)
    return {
        "kl_mean": kl.mean().item(),
        "kl_p50": kl.median().item(),
        "kl_p90": kl.quantile(0.9).item(),
        "kl_effective_posterior_part": kl.clamp(min=free_nats).mean().item(),
        "below_free_nats_fraction": (kl < free_nats).float().mean().item(),
        "kl_per_variable": variable_kl.tolist(),
        "active_variables": int((variable_kl > 0.01).sum()),
        "prior_entropy": categorical_entropy(prior).mean().item(),
        "posterior_entropy": categorical_entropy(posterior).mean().item(),
        "maximum_entropy": posterior.shape[-2] * math.log(classes),
        "classes_used": int(winners.amax(0).sum()),
        "classes_total": int(posterior.shape[-2] * classes),
        "class_perplexity_mean": perplexity.mean().item(),
        "prior_posterior_agreement": agreement.mean().item(),
    }


def save_checkpoint(path, model: MinecraftRSSM, extra: dict | None = None) -> None:
    """Weights, config and vocabulary. Loads with `weights_only=True`."""
    torch.save(
        {
            "format": "minecraft-rssm-checkpoint-v1",
            "config": model.config.to_json(),
            "vocabulary": asdict(model.vocabulary),
            "state_dict": model.state_dict(),
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(path) -> tuple[MinecraftRSSM, dict]:
    data = torch.load(path, weights_only=True)
    if data.get("format") != "minecraft-rssm-checkpoint-v1":
        raise ValueError(f"{path} is not a Minecraft RSSM checkpoint")
    model = MinecraftRSSM(
        Vocabulary(**data["vocabulary"]), ModelConfig(**data["config"])
    )
    model.load_state_dict(data["state_dict"])
    return model, data["extra"]
