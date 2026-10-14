"""Reward prediction on the Minecraft RSSM with full episode context
(docs/stage4-reward-prediction.md).

The reward of transition t (`reward[t]`, from action a_t into o_{t+1}) is
predicted from s_{t+1} = [h_{t+1}, z_{t+1}]. `reward-v1` novelty depends on
the whole episode so far, so s_{t+1} must come from a posterior filter that
starts at the first observation of the episode (h_0 = 0) and carries h over
every step, as the acting `Filter` does. A window that restarts h at 0 in the
middle of an episode cannot know what the episode saw before the window.

- `EpisodeData`: one episode as frozen-encoder embeddings, world-model
  actions, rewards and event labels. The labels are for evaluation only and
  never reach a model.
- `filter_episode`: the chronological posterior states of one episode.
- `reward_report`: error on positive and zero steps, precision, recall and
  false-positive rate at fixed thresholds, average precision, and errors by
  event type, next to the zero baseline.
- `RecurrentTrainer`: the reward-aware partial unfreeze. The GRU, the action
  layer and the prior and posterior networks learn from the canonical world
  model objective plus the reward loss; the encoder, the decoder and the
  continuation head stay frozen.
"""

import copy
from dataclasses import dataclass
from typing import Any

import numpy
import torch

from minecraft_rl.minecraft_actor import RewardHead
from minecraft_rl.minecraft_dataset import ACTION_BUTTONS, Episode
from minecraft_rl.minecraft_interface import RAY_KIND_BLOCK
from minecraft_rl.minecraft_replay import CompactVocabulary, compact_arrays
from minecraft_rl.minecraft_reward import COMPONENTS
from minecraft_rl.minecraft_world_model import (
    MinecraftRSSM,
    Objective,
    action_tensor,
    observation_tensors,
)

# Reward-v1 rewards are 0.01 (one block type seen) or more. A step counts as
# predicted positive at or above a threshold.
THRESHOLDS = (0.005, 0.02)
# The parameters that learn in the reward-aware run; everything else stays
# frozen (docs/stage4-reward-prediction.md).
TRAINABLE_MODULES = ("action_net", "cell", "prior_net", "posterior_net")
EVENT_TYPES = (
    "seen_first_step",
    "seen_later",
    "broken",
    "acquired",
    "zero_repeat_view",
    "zero_other",
)
BEHAVIORS = ("camera_spin", "jump_or_use", "neither")


@dataclass
class EpisodeData:
    """One episode for reward learning. `embedded` (T + 1, E) comes from the
    frozen encoder; `observations` are the compact observation tensors, for the
    observation loss; `labels` are bool masks (T,) per event type."""

    episode: int
    observations: dict[str, torch.Tensor]
    embedded: torch.Tensor
    actions: torch.Tensor
    continues: torch.Tensor
    rewards: torch.Tensor
    labels: dict[str, torch.Tensor]

    @property
    def steps(self) -> int:
        return int(self.rewards.shape[0])


def event_labels(episode: Episode) -> dict[str, numpy.ndarray]:
    """Bool masks per transition: which reward component paid, and for zero
    reward steps whether a block type that already paid seen novelty is in
    view again ("zero_repeat_view"). Evaluation only."""
    components = episode.arrays["reward_components"]
    reward = episode.arrays["reward"]
    index = {name: k for k, name in enumerate(COMPONENTS)}
    steps = len(reward)
    seen = components[:, index["block_seen_novelty"]] > 0
    first = numpy.zeros(steps, dtype=bool)
    first[0] = True
    repeat = numpy.zeros(steps, dtype=bool)
    kinds, types = episode.arrays["obs_ray_kind"], episode.arrays["obs_ray_type"]
    paid: set[int] = set(types[0][kinds[0] == RAY_KIND_BLOCK].tolist())
    for t in range(steps):
        visible = set(types[t + 1][kinds[t + 1] == RAY_KIND_BLOCK].tolist())
        if reward[t] == 0 and visible & paid:
            repeat[t] = True
        paid |= visible
    zero = reward == 0
    return {
        "positive": reward > 0,
        "zero": zero,
        "seen_first_step": seen & first,
        "seen_later": seen & ~first,
        "broken": components[:, index["block_broken_novelty"]] > 0,
        "acquired": components[:, index["item_acquired_novelty"]] > 0,
        "zero_repeat_view": zero & repeat,
        "zero_other": zero & ~repeat,
    }


@torch.no_grad()
def episode_data(
    model: MinecraftRSSM, episodes: list[Episode], vocabulary: CompactVocabulary
) -> list[EpisodeData]:
    out = []
    for episode in episodes:
        if "reward" not in episode.arrays:
            raise ValueError(f"episode {episode.meta.episode} has no reward")
        a = compact_arrays(episode.arrays, vocabulary)
        o = observation_tensors(a)
        actions = action_tensor(
            torch.as_tensor(a["action_buttons"]),
            torch.as_tensor(a["action_camera"]),
            torch.as_tensor(a["action_hotbar"]),
        )
        labels = {
            k: torch.as_tensor(v)
            for k, v in (event_labels(episode) | behavior_labels(episode)).items()
        }
        out.append(
            EpisodeData(
                episode=episode.meta.episode,
                observations=o,
                embedded=model.encoder(o),
                actions=actions,
                continues=1.0 - torch.as_tensor(a["terminated"]).float(),
                rewards=torch.as_tensor(episode.arrays["reward"]).float(),
                labels=labels,
            )
        )
    return out


def filter_episode(
    model: MinecraftRSSM, data: EpisodeData, generator: torch.Generator
) -> torch.Tensor:
    """Posterior states s_0..s_T (T + 1, F) of one episode, h_0 = 0 at the
    first observation and carried over every step."""
    filtered = model.filter_embedded(
        data.embedded.unsqueeze(0), data.actions.unsqueeze(0), generator
    )
    return torch.cat([filtered["h"], filtered["z"]], -1)[0]


@torch.no_grad()
def chronological_states(
    model: MinecraftRSSM, data: list[EpisodeData], seed: int
) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [filter_episode(model, d, generator) for d in data]


@torch.no_grad()
def window_states(
    model: MinecraftRSSM, data: EpisodeData, length: int, seed: int
) -> torch.Tensor:
    """Posterior states s_1..s_T when h restarts at 0 every `length` steps,
    as in the first actor experiment. Row t is the state into which
    transition t leads; the episode tail that fills no window is dropped."""
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for start in range(0, data.steps - length + 1, length):
        filtered = model.filter_embedded(
            data.embedded[start : start + length + 1].unsqueeze(0),
            data.actions[start : start + length].unsqueeze(0),
            generator,
        )
        rows.append(torch.cat([filtered["h"], filtered["z"]], -1)[0, 1:])
    return torch.cat(rows)


def average_precision(score: torch.Tensor, positive: torch.Tensor) -> float | None:
    """Area under the precision-recall curve as the mean precision at the rank
    of every positive step (ties broken by order)."""
    count = int(positive.sum())
    if count == 0:
        return None
    order = torch.argsort(score, descending=True, stable=True)
    hits = positive[order].float()
    precision = hits.cumsum(0) / torch.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / count)


def reward_report(
    predicted: torch.Tensor, target: torch.Tensor, labels: dict[str, torch.Tensor]
) -> dict[str, Any]:
    """Every number comes with the count of steps it describes."""
    positive, zero = labels["positive"], labels["zero"]
    out: dict[str, Any] = {
        "transitions": int(target.numel()),
        "positive_count": int(positive.sum()),
        "predicted_reward_per_step": float(predicted.mean()),
        "actual_reward_per_step": float(target.mean()),
    }
    for name, mask in (("positive", positive), ("zero", zero)):
        if mask.any():
            out[f"{name}_mae"] = float((predicted - target)[mask].abs().mean())
            out[f"{name}_zero_baseline_mae"] = float(target[mask].abs().mean())
            out[f"{name}_predicted_mean"] = float(predicted[mask].mean())
        else:
            out[f"{name}_mae"] = None
    out["average_precision"] = average_precision(predicted, positive)
    out["average_precision_chance"] = (
        float(positive.float().mean()) if positive.any() else None
    )
    for threshold in THRESHOLDS:
        called = predicted >= threshold
        true_positive = int((called & positive).sum())
        out[f"at_{threshold}"] = {
            "called_positive": int(called.sum()),
            "precision": true_positive / int(called.sum()) if called.any() else None,
            "recall": true_positive / int(positive.sum()) if positive.any() else None,
            "false_positive_rate": float(called[zero].float().mean())
            if zero.any()
            else None,
        }
    out["by_event"] = {}
    for name in EVENT_TYPES:
        mask = labels[name]
        count = int(mask.sum())
        out["by_event"][name] = {
            "count": count,
            "predicted_mean": float(predicted[mask].mean()) if count else None,
            "actual_mean": float(target[mask].mean()) if count else None,
            "called_at_0.005": float((predicted[mask] >= THRESHOLDS[0]).float().mean())
            if count
            else None,
        }
    out["by_behavior"] = {}
    for name in BEHAVIORS:
        if name not in labels:
            continue
        mask = labels[name] & zero
        count = int(mask.sum())
        out["by_behavior"][name] = {
            "zero_reward_steps": count,
            "predicted_mean": float(predicted[mask].mean()) if count else None,
            "false_positive_rate_at_0.005": float(
                (predicted[mask] >= THRESHOLDS[0]).float().mean()
            )
            if count
            else None,
        }
    return out


def report_on(
    head: RewardHead, states: list[torch.Tensor], data: list[EpisodeData]
) -> dict[str, Any]:
    """`states[i]` (T + 1, F) of episode i: s_{t+1} predicts reward[t]."""
    with torch.no_grad():
        predicted = torch.cat([head.reward(s[1:]) for s in states])
    target = torch.cat([d.rewards for d in data])
    labels = {
        name: torch.cat([d.labels[name] for d in data]) for name in data[0].labels
    }
    return reward_report(predicted, target, labels)


@dataclass(frozen=True)
class RewardTraining:
    """The reward loss of the first actor experiment (scale 100, positive
    weight 20), with chunks around rewarded steps sampled more often."""

    scale: float = 100.0
    positive_weight: float = 20.0
    chunk: int = 32
    batch: int = 16
    positive_chunk_share: float = 0.5
    learning_rate: float = 1e-3

    def to_json(self) -> dict[str, Any]:
        return dict(self.__dict__)


class ChunkSampler:
    """Chunks of `chunk` consecutive transitions. A share of the chunks is
    centered on a rewarded transition; the rest are uniform."""

    def __init__(self, data: list[EpisodeData], config: RewardTraining) -> None:
        self.data = data
        self.config = config
        self.positives = [
            (e, int(t))
            for e, d in enumerate(data)
            for t in torch.nonzero(d.rewards > 0).flatten()
        ]
        self.starts = torch.tensor([d.steps - config.chunk + 1 for d in data])
        if (self.starts < 1).any():
            raise ValueError(f"every episode needs {config.chunk} steps")

    def sample(self, generator: torch.Generator) -> list[tuple[int, int]]:
        out = []
        chunk = self.config.chunk
        for _ in range(self.config.batch):
            draw = torch.rand((), generator=generator)
            if self.positives and draw < self.config.positive_chunk_share:
                k = int(torch.randint(len(self.positives), (), generator=generator))
                e, t = self.positives[k]
                offset = int(torch.randint(chunk, (), generator=generator))
                start = min(max(0, t - offset), int(self.starts[e]) - 1)
            else:
                weights = self.starts.float() / self.starts.sum()
                e = int(torch.multinomial(weights, 1, generator=generator))
                start = int(torch.randint(int(self.starts[e]), (), generator=generator))
            out.append((e, start))
        return out


def reward_loss(
    head: RewardHead, s: torch.Tensor, target: torch.Tensor, config: RewardTraining
) -> torch.Tensor:
    weight = 1.0 + (config.positive_weight - 1.0) * (target != 0).float()
    return (weight * (head(s) - target * config.scale) ** 2).mean()


def train_frozen_head(
    states: list[torch.Tensor],
    data: list[EpisodeData],
    features: int,
    config: RewardTraining,
    updates: int,
    seed: int,
) -> tuple[RewardHead, list[dict[str, float]]]:
    """A reward head on fixed chronological states of the frozen RSSM."""
    torch.manual_seed(seed)
    head = RewardHead(features)
    head.scale = config.scale
    optimizer = torch.optim.Adam(head.parameters(), lr=config.learning_rate)
    sampler = ChunkSampler(data, config)
    generator = torch.Generator().manual_seed(seed + 1)
    history = []
    for update in range(updates):
        index = sampler.sample(generator)
        s = torch.stack([states[e][t + 1 : t + 1 + config.chunk] for e, t in index])
        target = torch.stack([data[e].rewards[t : t + config.chunk] for e, t in index])
        loss = reward_loss(head, s, target, config)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if update % 100 == 0:
            history.append({"update": update, "loss": float(loss.detach())})
    head.eval()
    return head, history


def set_trainable(model: MinecraftRSSM) -> list[torch.nn.Parameter]:
    """Freeze everything, then unfreeze the recurrent dynamics. Returns the
    trainable parameters."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for name in TRAINABLE_MODULES:
        for parameter in getattr(model, name).parameters():
            parameter.requires_grad_(True)
            trainable.append(parameter)
    return trainable


class RecurrentTrainer:
    """Reward-aware training of the recurrent dynamics. Each update samples
    chunks; a chunk starts from the stored chronological h of the current
    model (refreshed every `refresh_every` updates), so the reward loss sees
    the episode history before the chunk. The loss is the canonical world
    model loss plus `component_scale` times the reward loss: the reward is
    one more loss group under the group_mean rule."""

    def __init__(
        self,
        model: MinecraftRSSM,
        objective: Objective,
        data: list[EpisodeData],
        config: RewardTraining,
        core_learning_rate: float,
        refresh_every: int,
        seed: int,
    ) -> None:
        self.model = model
        self.objective = objective
        self.data = data
        self.config = config
        self.refresh_every = refresh_every
        self.trainable = set_trainable(model)
        torch.manual_seed(seed)
        self.head = RewardHead(model.features)
        self.head.scale = config.scale
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.trainable, "lr": core_learning_rate},
                {"params": self.head.parameters(), "lr": config.learning_rate},
            ],
            eps=1e-5,
        )
        self.sampler = ChunkSampler(data, config)
        self.generator = torch.Generator().manual_seed(seed + 1)
        self.update = 0
        self.history: list[dict[str, float]] = []
        self.stored_h: list[torch.Tensor] = []

    def refresh(self) -> None:
        self.model.eval()
        states = chronological_states(self.model, self.data, seed=self.update)
        self.stored_h = [s[:, : self.model.hidden] for s in states]

    def step(self) -> dict[str, float]:
        if self.update % self.refresh_every == 0 or not self.stored_h:
            self.refresh()
        self.model.train()
        chunk = self.config.chunk
        index = self.sampler.sample(self.generator)
        d = self.data
        observations = {
            name: torch.stack(
                [d[e].observations[name][t : t + chunk + 1] for e, t in index]
            )
            for name in d[0].observations
        }
        terms, s = self.model.loss_graph(
            observations,
            torch.stack([d[e].actions[t : t + chunk] for e, t in index]),
            torch.stack([d[e].continues[t : t + chunk] for e, t in index]),
            self.generator,
            self.objective,
            embedded=torch.stack([d[e].embedded[t : t + chunk + 1] for e, t in index]),
            initial_h=torch.stack([self.stored_h[e][t] for e, t in index]),
        )
        target = torch.stack([d[e].rewards[t : t + chunk] for e, t in index])
        reward = reward_loss(self.head, s[:, 1:], target, self.config)
        total = terms["total"] + self.objective.component_scale * reward
        self.optimizer.zero_grad()
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.trainable, 1000.0)
        self.optimizer.step()
        self.update += 1
        record = {
            "update": self.update,
            "total": float(total.detach()),
            "world_model": float(terms["total"].detach()),
            "reward": float(reward.detach()),
            "kl": float(terms["kl"]),
            "ray_class": float(terms["ray_class"].detach()),
            "grad_norm": float(norm),
        }
        if self.update % 50 == 0 or self.update == 1:
            self.history.append(record)
        return record

    def state(self) -> dict[str, Any]:
        """A detached copy: later updates do not change it."""
        return copy.deepcopy(
            {
                "model": self.model.state_dict(),
                "head": self.head.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "generator": self.generator.get_state(),
                "update": self.update,
                "history": self.history,
                "stored_h": self.stored_h,
            }
        )

    def load(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        self.head.load_state_dict(state["head"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.generator.set_state(state["generator"])
        self.update = state["update"]
        self.history = list(state["history"])
        self.stored_h = list(state.get("stored_h", []))


def history_labels(data_episode: Episode, block_types: list[int]) -> torch.Tensor:
    """(T + 1, K) bool: block type k was in view at an earlier step of the
    episode and is not in view now. A state can only know this from memory.
    Evaluation only."""
    kinds, types = (
        data_episode.arrays["obs_ray_kind"],
        data_episode.arrays["obs_ray_type"],
    )
    seen = numpy.zeros(len(block_types), dtype=bool)
    rows = []
    for t in range(len(kinds)):
        visible = set(types[t][kinds[t] == RAY_KIND_BLOCK].tolist())
        now = numpy.array([k in visible for k in block_types])
        rows.append(seen & ~now)
        seen |= now
    return torch.as_tensor(numpy.stack(rows))


def probe_auc(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    updates: int = 500,
) -> float | None:
    """Held-out ROC AUC of a logistic regression probe on standardized
    features; None if a split has one class only."""
    if train_y.all() or not train_y.any() or test_y.all() or not test_y.any():
        return None
    mean, std = train_x.mean(0), train_x.std(0).clamp(min=1e-6)
    a, b = (train_x - mean) / std, (test_x - mean) / std
    probe = torch.nn.Linear(a.shape[1], 1)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-3)
    weight = (1 - train_y.float().mean()) / train_y.float().mean()
    for _ in range(updates):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            probe(a).squeeze(-1), train_y.float(), pos_weight=weight
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        score = probe(b).squeeze(-1)
    positive, negative = score[test_y], score[~test_y]
    return float((positive.unsqueeze(1) > negative.unsqueeze(0)).float().mean())


def behavior_labels(episode: Episode) -> dict[str, numpy.ndarray]:
    """Bool masks per transition for the exploit behaviors of the first actor:
    a fast camera turn (at least 10 degrees of yaw), jump or use held, and
    neither. Evaluation only."""
    buttons = episode.arrays["action_buttons"]
    yaw = numpy.abs(episode.arrays["action_camera"][:, 0])
    spin = yaw >= 10.0
    spam = (
        buttons[:, ACTION_BUTTONS.index("jump")]
        | buttons[:, ACTION_BUTTONS.index("use")]
    )
    return {"camera_spin": spin, "jump_or_use": spam, "neither": ~spin & ~spam}


def latent_ambiguity(
    states: list[torch.Tensor], data: list[EpisodeData]
) -> dict[str, Any]:
    """For each rewarded step after step 0, the distance from its state
    s_{t+1} to the nearest zero-reward state of another episode, against the
    typical distance between two consecutive states. A ratio near or below 1
    means a zero-reward history gives almost the same state."""
    rows, rewards, episode_of = [], [], []
    for e, (s, d) in enumerate(zip(states, data, strict=True)):
        rows.append(s[1:])
        rewards.append(d.rewards)
        episode_of.append(torch.full((d.steps,), e))
    s, r, owner = torch.cat(rows), torch.cat(rewards), torch.cat(episode_of)
    step = torch.cat([(x[2:] - x[1:-1]).norm(dim=-1) for x in states]).median()
    first = torch.cat([torch.arange(d.steps) == 0 for d in data])
    targets = torch.nonzero((r > 0) & ~first).flatten()
    ratios = []
    for i in targets.tolist():
        candidates = (r == 0) & (owner != owner[i])
        distance = (s[candidates] - s[i]).norm(dim=-1).min()
        ratios.append(float(distance / step))
    ratios_t = torch.tensor(ratios)
    return {
        "positive_after_step_0": len(ratios),
        "median_consecutive_state_distance": float(step),
        "nearest_zero_state_ratio_median": float(ratios_t.median()) if ratios else None,
        "share_with_ratio_at_most_1": float((ratios_t <= 1.0).float().mean())
        if ratios
        else None,
        "share_with_ratio_at_most_2": float((ratios_t <= 2.0).float().mean())
        if ratios
        else None,
    }
