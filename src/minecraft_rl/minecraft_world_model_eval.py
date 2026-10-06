"""Held-out evaluation of the Minecraft RSSM against trivial baselines.

Two evaluations exist (docs/stage3-minecraft-world-model.md):

- `one_step` filters real windows with the posterior and scores the prior
  prediction of each next observation: the decoder reads s_k = [h_k, z_k]
  with h_k from the posterior state at k - 1 and the real action a_{k-1},
  and z_k the most likely class of each prior variable. It also scores the
  same prediction with the action of another window (action shuffle), the
  posterior reconstruction, and event subsets.
- `imagination` filters C real steps, then rolls the prior forward for H
  steps with the real future actions, without decoding and re-encoding,
  and scores the decoded state at every horizon.

Baselines: persistence predicts that the last real observation stays the
same. The frequency baseline predicts, for each ray position, the most
frequent known ray class of the training split, and the most frequent
selected slot.

Metrics are sums and counts, so every number is a mean over the elements
it names. "Changed" metrics use only elements whose target differs from the
last real observation. Persistence is wrong on all of them by definition,
so these metrics show whether the model predicts change. Ray targets of an
unknown class (an id that the training split never showed) are left out of
every ray accuracy and reported on their own: a prediction of the unknown
class does not identify the original block.
"""

from collections import defaultdict

import torch
from torch import nn

from minecraft_rl.minecraft_dataset import ACTION_BUTTONS
from minecraft_rl.minecraft_interface import RAY_KIND_ENTITY, RAY_KIND_NONE
from minecraft_rl.minecraft_replay import CompactVocabulary, SequenceReplay
from minecraft_rl.minecraft_world_model import (
    COUNT_LOG_SCALE,
    FOOD_SCALE,
    HEALTH_SCALE,
    PITCH_SCALE,
    MinecraftRSSM,
    latent_statistics,
    ray_class,
)

HORIZONS = (1, 5, 10, 20)
MIN_EVENT_EXAMPLES = 20
BUTTON = {name: index for index, name in enumerate(ACTION_BUTTONS)}
MOVEMENT_BUTTONS = ("forward", "back", "left", "right", "jump")


def unknown_classes(compact: CompactVocabulary, model: MinecraftRSSM) -> torch.Tensor:
    """Bool (ray_classes,): the joint ray classes that stand for unknown."""
    v = model.vocabulary
    out = torch.zeros(v.ray_classes, dtype=torch.bool)
    for kind, first in v.kind_offsets().items():
        family = {1: "block", 2: "fluid", 3: "entity"}[kind]
        out[first + compact.unknown(family)] = True
    return out


def ray_frequency_baseline(
    replay: SequenceReplay, model: MinecraftRSSM, unknown: torch.Tensor
) -> torch.Tensor:
    """The most frequent known ray class per ray position, shape (rays,)."""
    v = model.vocabulary
    counts = torch.zeros(v.rays, v.ray_classes)
    for o in replay.observations:
        classes = ray_class(o["ray_kind"], o["ray_type"], v)
        counts.scatter_add_(
            1, classes.T, torch.ones(classes.T.shape, dtype=counts.dtype)
        )
    counts[:, unknown] = -1.0
    return counts.argmax(1)


def slot_frequency_baseline(replay: SequenceReplay) -> int:
    slots = torch.cat([o["selected_slot"] for o in replay.observations])
    return int(torch.bincount(slots).argmax())


def point_prediction(prediction: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Decoder outputs as values in observation units."""
    scalars = prediction["scalars"]
    return {
        "ray_class": prediction["ray_class"].argmax(-1),
        "ray_distance": prediction["ray_distance"],
        "health": scalars[..., 0] * HEALTH_SCALE,
        "food": scalars[..., 3] * FOOD_SCALE,
        "pitch": scalars[..., 8] * PITCH_SCALE,
        "selected_slot": prediction["selected_slot"].argmax(-1),
        "inventory_item": prediction["inventory_item"].argmax(-1),
        "inventory_count": torch.expm1(
            prediction["inventory_count"].clamp(min=0) * COUNT_LOG_SCALE
        ),
    }


def observed_point(o: dict[str, torch.Tensor], model: MinecraftRSSM) -> dict:
    """An observation in the same form, for persistence and as a target.
    Ray distance is in units of the maximum distance, as the decoder."""
    return {
        "ray_class": ray_class(o["ray_kind"], o["ray_type"], model.vocabulary),
        "ray_distance": o["ray_distance"] / model.vocabulary.max_distance,
        "health": o["health"],
        "food": o["food"],
        "pitch": o["pitch"],
        "selected_slot": o["selected_slot"],
        "inventory_item": o["inventory_item"],
        "inventory_count": o["inventory_count"],
        "ray_kind": o["ray_kind"],
    }


def point_metrics(
    point: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    unknown: torch.Tensor,
    max_distance: float,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """(sum, count) per state, shape (...) of the state axes, per metric."""

    def ratio(values: torch.Tensor, mask: torch.Tensor, axes: int):
        values = values.float() * mask
        mask = mask.float()
        if axes:
            values = values.flatten(-axes).sum(-1)
            mask = mask.flatten(-axes).sum(-1)
        return values, mask

    target_class, reference_class = target["ray_class"], reference["ray_class"]
    known = ~unknown[target_class]
    changed = target_class != reference_class
    correct = point["ray_class"] == target_class
    hit = target["ray_kind"] != RAY_KIND_NONE
    distance_error = (
        point["ray_distance"] - target["ray_distance"]
    ).abs() * max_distance
    distance_changed = (
        target["ray_distance"] - reference["ray_distance"]
    ).abs() * max_distance > 0.25
    out = {
        "ray_accuracy": ratio(correct, known, 1),
        "ray_accuracy_changed": ratio(correct, known & changed, 1),
        "ray_distance_mae_blocks": ratio(distance_error, hit, 1),
        "ray_distance_mae_changed_blocks": ratio(
            distance_error, hit & distance_changed, 1
        ),
    }
    for name, threshold in (("pitch", 0.1), ("health", 0.01), ("food", 0.5)):
        error = (point[name] - target[name]).abs()
        changed_value = (target[name] - reference[name]).abs() > threshold
        out[f"{name}_mae"] = ratio(error, torch.ones_like(error, dtype=torch.bool), 0)
        out[f"{name}_mae_changed"] = ratio(error, changed_value, 0)
    slot_correct = point["selected_slot"] == target["selected_slot"]
    out["selected_slot_accuracy_changed"] = ratio(
        slot_correct, target["selected_slot"] != reference["selected_slot"], 0
    )
    item_changed = target["inventory_item"] != reference["inventory_item"]
    out["inventory_item_accuracy_changed"] = ratio(
        point["inventory_item"] == target["inventory_item"], item_changed, 1
    )
    count_error = (point["inventory_count"] - target["inventory_count"]).abs()
    count_changed = target["inventory_count"] != reference["inventory_count"]
    out["inventory_count_mae_changed"] = ratio(count_error, count_changed, 1)
    return out


def model_ray_metrics(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    unknown: torch.Tensor,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Ray-class negative log-likelihood, and how the model treats unknown
    targets, per state."""
    logits = prediction["ray_class"]
    target_class = target["ray_class"]
    nll = nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), target_class.reshape(-1), reduction="none"
    ).reshape(target_class.shape)
    known = ~unknown[target_class]
    changed = target_class != reference["ray_class"]
    unknown_target = unknown[target_class]
    predicted_unknown = unknown[logits.argmax(-1)]
    return {
        "ray_nll": ((nll * known).sum(-1), known.float().sum(-1)),
        "ray_nll_changed": (
            (nll * (known & changed)).sum(-1),
            (known & changed).float().sum(-1),
        ),
        "ray_unknown_target_fraction": (
            unknown_target.float().sum(-1),
            torch.ones_like(nll).sum(-1),
        ),
        "ray_predicted_unknown_on_unknown_targets": (
            (predicted_unknown & unknown_target).float().sum(-1),
            unknown_target.float().sum(-1),
        ),
    }


def event_masks(
    actions_buttons: torch.Tensor,
    actions_camera: torch.Tensor,
    actions_hotbar: torch.Tensor,
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
    terminated: torch.Tensor,
    model: MinecraftRSSM,
) -> dict[str, torch.Tensor]:
    """Bool masks per transition for events that the data can show."""
    v = model.vocabulary
    camera = actions_camera.abs().sum(-1) > 1.0
    moving = actions_buttons[..., [BUTTON[b] for b in MOVEMENT_BUTTONS]].any(-1)
    before_class = ray_class(before["ray_kind"], before["ray_type"], v)
    after_class = ray_class(after["ray_kind"], after["ray_type"], v)
    entities = lambda o: (o["ray_kind"] == RAY_KIND_ENTITY).sum(-1)  # noqa: E731
    events = {
        "camera_turn": camera,
        "forward": actions_buttons[..., BUTTON["forward"]],
        "jump": actions_buttons[..., BUTTON["jump"]],
        "sneak": actions_buttons[..., BUTTON["sneak"]],
        "attack": actions_buttons[..., BUTTON["attack"]],
        "use": actions_buttons[..., BUTTON["use"]],
        "hotbar_select": (actions_hotbar >= 0)
        & (actions_hotbar != before["selected_slot"]),
        "no_movement_no_camera": ~moving & (actions_camera.abs().sum(-1) < 0.01),
        "many_rays_change": (before_class != after_class).float().mean(-1) > 0.1,
        "entity_rays_change": entities(before) != entities(after),
        "health_change": (after["health"] - before["health"]).abs() > 0.01,
        "food_change": after["food"] != before["food"],
        "inventory_change": (
            (after["inventory_item"] != before["inventory_item"])
            | (after["inventory_count"] != before["inventory_count"])
        ).any(-1),
        "termination": terminated,
    }
    return {name: mask.bool() for name, mask in events.items()}


def _slice(o: dict[str, torch.Tensor], index) -> dict[str, torch.Tensor]:
    return {name: value[index] for name, value in o.items()}


class Accumulator:
    """Sums and counts per predictor, metric and optional subset."""

    def __init__(self) -> None:
        self.sums: dict[tuple, float] = defaultdict(float)
        self.counts: dict[tuple, float] = defaultdict(float)

    def add(self, key: tuple, metrics: dict, mask: torch.Tensor | None = None) -> None:
        for name, (total, count) in metrics.items():
            if mask is not None:
                total, count = total * mask, count * mask
            self.sums[(*key, name)] += float(total.sum())
            self.counts[(*key, name)] += float(count.sum())

    def result(self) -> dict:
        out: dict = {}
        for key, total in self.sums.items():
            count = self.counts[key]
            node = out
            for part in key[:-1]:
                node = node.setdefault(str(part), {})
            node[key[-1]] = {"mean": total / count if count else None, "count": count}
        return out


def _windows(replay: SequenceReplay, index: list[tuple[int, int]], size: int):
    for first in range(0, len(index), size):
        yield replay.batch(index[first : first + size])


def _raw_actions(replay: SequenceReplay, batch_index: torch.Tensor, starts, length):
    """Buttons, camera and hotbar arrays of each window, for event masks."""
    buttons, camera, hotbar, terminated = [], [], [], []
    for e, s in zip(batch_index.tolist(), starts.tolist(), strict=True):
        a = replay.episodes[e].arrays
        buttons.append(torch.as_tensor(a["action_buttons"][s : s + length]))
        camera.append(torch.as_tensor(a["action_camera"][s : s + length]).float())
        hotbar.append(torch.as_tensor(a["action_hotbar"][s : s + length]).long())
        terminated.append(torch.as_tensor(a["terminated"][s : s + length]))
    return (
        torch.stack(buttons).bool(),
        torch.stack(camera),
        torch.stack(hotbar),
        torch.stack(terminated).bool(),
    )


@torch.no_grad()
def one_step(
    model: MinecraftRSSM,
    replay: SequenceReplay,
    index: list[tuple[int, int]],
    score_from: int,
    unknown: torch.Tensor,
    frequency_rays: torch.Tensor,
    frequency_slot: int,
    generator: torch.Generator,
    batch_size: int = 32,
    events: bool = True,
) -> dict:
    """One-step prior prediction on windows of `replay.length` transitions.
    Scores the transitions into states score_from..L of each window; earlier
    states only build context."""
    model.eval()
    model.mode_latents = True
    max_distance = model.vocabulary.max_distance
    acc = Accumulator()
    continuation = defaultdict(float)
    filtered_all = defaultdict(list)
    for batch in _windows(replay, index, batch_size):
        o, actions = batch.observations, batch.actions
        f = model.filter(o, actions, generator)
        for name in ("prior", "posterior"):
            filtered_all[name].append(f[name][:, score_from:])
        # States k = score_from..L; their previous states k - 1 index the
        # L + 1 observations, and the actions a_{k-1} index the L actions.
        k = slice(score_from, None)
        previous = slice(score_from - 1, -1)
        acted = slice(score_from - 1, None)
        h_prev, z_prev = f["h"][:, previous], f["z"][:, previous]
        a = actions[:, acted]
        rolled = torch.roll(actions, 1, 0)[:, acted]
        target = observed_point(_slice(o, (slice(None), k)), model)
        reference = observed_point(_slice(o, (slice(None), previous)), model)
        subsets: dict[str, torch.Tensor | None] = {"all": None}
        if events:
            buttons, camera, hotbar, terminated = _raw_actions(
                replay, batch.episodes, batch.starts, replay.length
            )
            subsets |= event_masks(
                buttons[:, acted],
                camera[:, acted],
                hotbar[:, acted],
                _slice(o, (slice(None), previous)),
                _slice(o, (slice(None), k)),
                terminated[:, acted],
                model,
            )
        for name, action in (("model", a), ("model_shuffled_actions", rolled)):
            h = model.transition(
                h_prev.flatten(0, 1), z_prev.flatten(0, 1), action.flatten(0, 1)
            )
            z = model.sample(model.prior_logits(h), generator, straight_through=False)
            s = torch.cat([h, z], -1).unflatten(0, h_prev.shape[:2])
            prediction = model.decoder(s)
            metrics = point_metrics(
                point_prediction(prediction), target, reference, unknown, max_distance
            )
            metrics |= model_ray_metrics(prediction, target, reference, unknown)
            for subset, mask in subsets.items():
                acc.add((subset, name), metrics, mask)
            if name == "model":
                p = torch.sigmoid(model.continue_head(s).squeeze(-1))
                continues = batch.continues[:, acted]
                continuation["model_bce_sum"] += float(
                    nn.functional.binary_cross_entropy(p, continues, reduction="sum")
                )
                continuation["always_continue_bce_sum"] += float(
                    nn.functional.binary_cross_entropy(
                        torch.full_like(continues, 1.0 - 1e-6),
                        continues,
                        reduction="sum",
                    )
                )
                continuation["transitions"] += continues.numel()
                continuation["terminal_transitions"] += float((continues == 0).sum())
                continuation["p_continue_terminal_sum"] += float(
                    p[continues == 0].sum()
                )
                continuation["p_continue_nonterminal_sum"] += float(
                    p[continues == 1].sum()
                )
        persistence = point_metrics(reference, target, reference, unknown, max_distance)
        frequency = dict(reference)
        frequency["ray_class"] = frequency_rays.expand_as(reference["ray_class"])
        frequency["selected_slot"] = torch.full_like(
            reference["selected_slot"], frequency_slot
        )
        frequency = {
            m: value
            for m, value in point_metrics(
                frequency, target, reference, unknown, max_distance
            ).items()
            if m.startswith(("ray_accuracy", "selected_slot"))
        }
        for subset, mask in subsets.items():
            acc.add((subset, "persistence"), persistence, mask)
            acc.add((subset, "frequency"), frequency, mask)
        posterior_s = torch.cat([f["h"][:, k], f["z"][:, k]], -1)
        reconstruction = point_metrics(
            point_prediction(model.decoder(posterior_s)),
            target,
            target,
            unknown,
            max_distance,
        )
        acc.add(("all", "posterior_reconstruction"), reconstruction)
    model.mode_latents = False
    result = acc.result()
    for subset in list(result):
        if subset == "all":
            continue
        count = result[subset]["persistence"]["pitch_mae"]["count"]
        result[subset]["transitions"] = count
        if count < MIN_EVENT_EXAMPLES:
            result[subset] = {"transitions": count, "too_few_examples": True}
    n = continuation["transitions"]
    terminal = continuation["terminal_transitions"]
    result["continuation"] = {
        "transitions": n,
        "terminal_transitions": terminal,
        "model_bce": continuation["model_bce_sum"] / n if n else None,
        "always_continue_bce": continuation["always_continue_bce_sum"] / n
        if n
        else None,
        "p_continue_on_terminal": (
            continuation["p_continue_terminal_sum"] / terminal if terminal else None
        ),
        "p_continue_on_nonterminal": (
            continuation["p_continue_nonterminal_sum"] / (n - terminal)
            if n > terminal
            else None
        ),
    }
    stats = latent_statistics(
        {name: torch.cat(values) for name, values in filtered_all.items()},
        model.config.free_nats,
    )
    stats.pop("kl_per_variable")
    result["latent"] = stats
    return result


@torch.no_grad()
def imagination(
    model: MinecraftRSSM,
    replay: SequenceReplay,
    index: list[tuple[int, int]],
    context: int,
    unknown: torch.Tensor,
    generator: torch.Generator,
    batch_size: int = 32,
) -> dict:
    """Latent rollouts: windows of `context + H` transitions, H = replay
    length - context. Scores horizon k = 1..H against o_{C + k}."""
    model.eval()
    model.mode_latents = True
    max_distance = model.vocabulary.max_distance
    horizon = replay.length - context
    acc = Accumulator()
    for batch in _windows(replay, index, batch_size):
        o, actions = batch.observations, batch.actions
        known = _slice(o, (slice(None), slice(0, context + 1)))
        f = model.filter(known, actions[:, :context], generator)
        h, z = f["h"][:, -1], f["z"][:, -1]
        future = actions[:, context:]
        target = observed_point(
            _slice(o, (slice(None), slice(context + 1, None))), model
        )
        last = observed_point(
            _slice(o, (slice(None), slice(context, context + 1))), model
        )
        reference = {
            name: value.expand_as(target[name]) for name, value in last.items()
        }
        for name, action in (
            ("model", future),
            ("model_shuffled_actions", torch.roll(future, 1, 0)),
        ):
            s = model.imagine(h, z, action, generator)
            prediction = model.decoder(s)
            metrics = point_metrics(
                point_prediction(prediction), target, reference, unknown, max_distance
            )
            metrics |= model_ray_metrics(prediction, target, reference, unknown)
            for k in HORIZONS:
                if k <= horizon:
                    acc.add(
                        (f"horizon_{k}", name),
                        {
                            m: (t[:, k - 1], c[:, k - 1])
                            for m, (t, c) in metrics.items()
                        },
                    )
        persistence = point_metrics(reference, target, reference, unknown, max_distance)
        for k in HORIZONS:
            if k <= horizon:
                acc.add(
                    (f"horizon_{k}", "persistence"),
                    {
                        m: (t[:, k - 1], c[:, k - 1])
                        for m, (t, c) in persistence.items()
                    },
                )
    model.mode_latents = False
    return acc.result()
