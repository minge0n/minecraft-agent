"""The first Minecraft actor (docs/stage4-minecraft-actor.md).

A Dreamer-style actor-critic on the frozen Stage 3 RSSM:

- `ActionSpace`: the physical factors of `player-action-v3` as small
  categorical factors (movement, five buttons, yaw, pitch, hotbar). No
  abstract action: the frozen world model has no input for it.
- `RewardHead`: a small network r(s_t) on the RSSM features s = [h, z]. It
  predicts the reward of the transition into s_t, so it works on imagined
  states. The RSSM stays frozen; only this head learns from rewards.
- `Agent`: one actor head per factor and a critic, both on s = [h, z].
- `imagine` and `losses`: latent rollouts from real posterior states, lambda
  returns, REINFORCE with a critic baseline and an entropy bonus, as in
  Stage 2E (`actor_critic.py`).
- `Filter`: one RSSM posterior step per real Minecraft step, for acting.

Everything here reads the frozen world model and never changes it.
"""

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from minecraft_rl.actor_critic import lambda_returns
from minecraft_rl.exploration import MOVEMENTS
from minecraft_rl.minecraft_dataset import ACTION_BUTTONS
from minecraft_rl.minecraft_interface import (
    HOTBAR_SLOTS,
    KEEP_HOTBAR_SLOT,
    PlayerAction,
)
from minecraft_rl.minecraft_world_model import (
    MinecraftRSSM,
    Objective,
    action_tensor,
    load_checkpoint,
    observation_tensors,
)

YAW_DEGREES = (-30.0, -10.0, -3.0, 0.0, 3.0, 10.0, 30.0)
PITCH_DEGREES = (-10.0, -3.0, 0.0, 3.0, 10.0)
BUTTONS = ("jump", "sneak", "sprint", "attack", "use")
# The canonical Stage 3 objective of the frozen world model (AGENTS.md).
CANONICAL_OBJECTIVE = Objective(
    reduction="group_mean", component_scale=825.0, pitch_loss_scale=128.0
)
CANONICAL_CONFIG = {
    "ray_layer": "patch",
    "patch": 4,
    "fusion": "shared",
    "free_nats": 1.0,
    "hidden": 256,
    "latent_variables": 16,
    "latent_classes": 16,
}


class ActionSpace:
    """Factorized discrete physical actions. Factor sizes:
    movement 9, jump/sneak/sprint/attack/use 2 each, yaw 7, pitch 5,
    hotbar 10 (keep or slot 0-8)."""

    FACTORS = (
        ("movement", len(MOVEMENTS)),
        *((name, 2) for name in BUTTONS),
        ("yaw", len(YAW_DEGREES)),
        ("pitch", len(PITCH_DEGREES)),
        ("hotbar", HOTBAR_SLOTS + 1),
    )
    NAMES = tuple(name for name, _ in FACTORS)
    SIZES = tuple(size for _, size in FACTORS)
    # The no-op: still, no button, no turn, keep the slot.
    NOOP = (0, 0, 0, 0, 0, 0, YAW_DEGREES.index(0.0), PITCH_DEGREES.index(0.0), 0)

    @classmethod
    def to_player_action(cls, choice: tuple[int, ...] | list[int]) -> PlayerAction:
        values = dict(zip(cls.NAMES, (int(c) for c in choice), strict=True))
        forward, back, left, right = MOVEMENTS[values["movement"]]
        hotbar = values["hotbar"] - 1
        return PlayerAction(
            forward=forward,
            back=back,
            left=left,
            right=right,
            jump=bool(values["jump"]),
            sneak=bool(values["sneak"]),
            sprint=bool(values["sprint"]) and forward,
            attack=bool(values["attack"]),
            use=bool(values["use"]),
            yaw_delta=YAW_DEGREES[values["yaw"]],
            pitch_delta=PITCH_DEGREES[values["pitch"]],
            hotbar=KEEP_HOTBAR_SLOT if hotbar < 0 else hotbar,
        )

    @classmethod
    def world_model_action(cls, choices: torch.Tensor) -> torch.Tensor:
        """The 21-value action vector of the frozen RSSM for choices (..., 9)."""
        movement = torch.as_tensor(MOVEMENTS, dtype=torch.float32)[choices[..., 0]]
        sprint = choices[..., 3].float() * movement[..., 0]
        named = {
            "forward": movement[..., 0],
            "back": movement[..., 1],
            "left": movement[..., 2],
            "right": movement[..., 3],
            "jump": choices[..., 1].float(),
            "sneak": choices[..., 2].float(),
            "sprint": sprint,
            "attack": choices[..., 4].float(),
            "use": choices[..., 5].float(),
        }
        buttons = torch.stack([named[name] for name in ACTION_BUTTONS], -1)
        camera = torch.stack(
            [
                torch.as_tensor(YAW_DEGREES)[choices[..., 6]],
                torch.as_tensor(PITCH_DEGREES)[choices[..., 7]],
            ],
            -1,
        )
        hotbar = choices[..., 8] - 1
        return action_tensor(buttons, camera, hotbar)


def load_world_model(path: Path) -> tuple[MinecraftRSSM, dict[str, Any]]:
    """The frozen Stage 3 RSSM. Refuses a checkpoint whose recorded config or
    objective is not the canonical one."""
    model, extra = load_checkpoint(path)
    config = model.config.to_json()
    differing = {k: config[k] for k, v in CANONICAL_CONFIG.items() if config[k] != v}
    if differing:
        raise ValueError(f"{path}: not the canonical world model config: {differing}")
    objective = Objective.from_json(extra.get("objective", {"reduction": "sum"}))
    if objective != CANONICAL_OBJECTIVE:
        raise ValueError(f"{path}: not the canonical objective: {objective.to_json()}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, extra


class RewardHead(nn.Module):
    """r(s_t): the reward of the transition into s_t, from s = [h, z]. The
    network learns rewards times `scale`; `forward` returns that scaled value,
    `reward` the reward itself."""

    def __init__(self, features: int, hidden: int = 256) -> None:
        super().__init__()
        self.scale = 1.0
        self.net = nn.Sequential(
            nn.Linear(features, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).squeeze(-1)

    def reward(self, s: torch.Tensor) -> torch.Tensor:
        return self(s) / self.scale


class Agent(nn.Module):
    def __init__(self, features: int, hidden: int = 256) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(features, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU()
        )
        self.heads = nn.ModuleList(
            nn.Linear(hidden, size) for size in ActionSpace.SIZES
        )
        self.critic = nn.Sequential(
            nn.Linear(features, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1),
        )

    def logits(self, s: torch.Tensor) -> list[torch.Tensor]:
        x = self.trunk(s)
        return [head(x) for head in self.heads]

    def act(
        self, s: torch.Tensor, generator: torch.Generator | None, greedy: bool = False
    ) -> torch.Tensor:
        """Choices (..., 9) for states (..., F)."""
        choices = []
        for logits in self.logits(s):
            if greedy:
                choices.append(logits.argmax(-1))
            else:
                p = torch.softmax(logits, -1).reshape(-1, logits.shape[-1])
                sample = torch.multinomial(p, 1, generator=generator)
                choices.append(sample.reshape(logits.shape[:-1]))
        return torch.stack(choices, -1)

    def log_prob_entropy(
        self, s: torch.Tensor, choices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Joint log probability and entropy (the sum over factors), and the
        entropy of each factor."""
        log_prob = torch.zeros(s.shape[:-1])
        entropy = torch.zeros(s.shape[:-1])
        per_factor = []
        for k, logits in enumerate(self.logits(s)):
            log_p = torch.log_softmax(logits, -1)
            log_prob = log_prob + log_p.gather(-1, choices[..., k : k + 1]).squeeze(-1)
            h = -(log_p.exp() * log_p).sum(-1)
            entropy = entropy + h
            per_factor.append(h)
        return log_prob, entropy, per_factor

    def value(self, s: torch.Tensor) -> torch.Tensor:
        return self.critic(s).squeeze(-1)


@dataclass(frozen=True)
class ActorConfig:
    horizon: int = 15
    discount: float = 0.99
    return_lambda: float = 0.95
    entropy: float = 0.003
    learning_rate: float = 3e-4
    start_states: int = 256
    seed: int = 0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Imagined:
    features: torch.Tensor  # (n, T + 1, F)
    choices: torch.Tensor  # (n, T, 9)
    rewards: torch.Tensor  # (n, T)
    continues: torch.Tensor  # (n, T)


@torch.no_grad()
def imagine(
    model: MinecraftRSSM,
    reward_head: RewardHead,
    agent: Agent,
    h: torch.Tensor,
    z: torch.Tensor,
    horizon: int,
    generator: torch.Generator,
) -> Imagined:
    """Roll the frozen RSSM forward from posterior states (h, z) with actions
    from the actor. The reward of a step is r(s_{t+1}); the continuation is
    the RSSM continuation head on s_{t+1}."""
    s = torch.cat([h, z], -1)
    features, choices, rewards, continues = [s], [], [], []
    for _ in range(horizon):
        choice = agent.act(s, generator)
        h = model.transition(h, z, ActionSpace.world_model_action(choice))
        z = model.sample(model.prior_logits(h), generator, straight_through=False)
        s = torch.cat([h, z], -1)
        features.append(s)
        choices.append(choice)
        rewards.append(reward_head.reward(s))
        continues.append(torch.sigmoid(model.continue_head(s).squeeze(-1)))
    return Imagined(
        torch.stack(features, 1),
        torch.stack(choices, 1),
        torch.stack(rewards, 1),
        torch.stack(continues, 1),
    )


def losses(agent: Agent, imagined: Imagined, config: ActorConfig) -> dict[str, Any]:
    values = agent.value(imagined.features)
    with torch.no_grad():
        returns = lambda_returns(
            imagined.rewards,
            imagined.continues,
            values,
            config.discount,
            config.return_lambda,
        )
        alive = torch.cat(
            [
                torch.ones_like(imagined.continues[:, :1]),
                imagined.continues.cumprod(1)[:, :-1],
            ],
            1,
        )
        advantages = returns - values[:, :-1]
        # Scale-free advantages: reward magnitudes differ by orders of magnitude.
        scale = advantages.abs().mean().clamp(min=1e-3)
    critic = (alive * 0.5 * (values[:, :-1] - returns) ** 2).mean()
    log_prob, entropy, per_factor = agent.log_prob_entropy(
        imagined.features[:, :-1], imagined.choices
    )
    actor = -(alive * (log_prob * advantages / scale + config.entropy * entropy)).mean()
    return {
        "actor": actor,
        "critic": critic,
        "entropy": float(entropy.detach().mean()),
        "entropy_per_factor": {
            name: float(h.detach().mean())
            for name, h in zip(ActionSpace.NAMES, per_factor, strict=True)
        },
        "imagined_return": float(returns[:, 0].mean()),
        "imagined_reward_per_step": float(imagined.rewards.mean()),
    }


def train_actor(
    model: MinecraftRSSM,
    reward_head: RewardHead,
    agent: Agent,
    optimizer: torch.optim.Optimizer,
    starts: tuple[torch.Tensor, torch.Tensor],
    config: ActorConfig,
    generator: torch.Generator,
    updates: int,
) -> list[dict[str, Any]]:
    h_all, z_all = starts
    history = []
    for _ in range(updates):
        index = torch.randint(
            0, h_all.shape[0], (config.start_states,), generator=generator
        )
        imagined = imagine(
            model,
            reward_head,
            agent,
            h_all[index],
            z_all[index],
            config.horizon,
            generator,
        )
        terms = losses(agent, imagined, config)
        optimizer.zero_grad()
        (terms["actor"] + terms["critic"]).backward()
        norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), 100.0)
        optimizer.step()
        history.append(
            {
                "actor": float(terms["actor"].detach()),
                "critic": float(terms["critic"].detach()),
                "entropy": terms["entropy"],
                "entropy_per_factor": terms["entropy_per_factor"],
                "imagined_return": terms["imagined_return"],
                "imagined_reward_per_step": terms["imagined_reward_per_step"],
                "grad_norm": float(norm),
            }
        )
    return history


class Filter:
    """One RSSM posterior step per real step: h_t = GRU(h_{t-1}, z_{t-1},
    a_{t-1}), z_t ~ q(z | h_t, e(o_t)). Call `reset(o_0)` and then
    `step(action, o_t)`. Observations are compact dataset-style arrays."""

    def __init__(self, model: MinecraftRSSM, generator: torch.Generator) -> None:
        self.model = model
        self.generator = generator
        self.h: torch.Tensor | None = None
        self.z: torch.Tensor | None = None

    @torch.no_grad()
    def _posterior(self, arrays: dict) -> None:
        o = {k: v.unsqueeze(0) for k, v in observation_tensors(arrays).items()}
        e = self.model.encoder(o)
        logits = self.model.posterior_logits(self.h, e)
        self.z = self.model.sample(logits, self.generator, straight_through=False)

    def reset(self, arrays: dict) -> torch.Tensor:
        self.h = torch.zeros(1, self.model.hidden)
        self._posterior(arrays)
        return self.state

    @torch.no_grad()
    def step(self, choice: torch.Tensor, arrays: dict) -> torch.Tensor:
        action = ActionSpace.world_model_action(choice.reshape(1, -1))
        self.h = self.model.transition(self.h, self.z, action)
        self._posterior(arrays)
        return self.state

    @property
    def state(self) -> torch.Tensor:
        return torch.cat([self.h, self.z], -1)


@torch.no_grad()
def posterior_states(
    model: MinecraftRSSM, batch, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Posterior (h, z) of every state of a batch of windows, flattened."""
    filtered = model.filter(batch.observations, batch.actions, generator)
    return filtered["h"].flatten(0, 1), filtered["z"].flatten(0, 1)


def reward_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    """Prediction error overall, on zero-reward and on positive-reward steps,
    with the zero and mean baselines."""
    out: dict[str, Any] = {"transitions": int(target.numel())}
    positive = target > 0
    zero = target == 0
    mean = float(target.mean())
    for name, mask in (
        ("all", torch.ones_like(target, dtype=bool)),
        ("zero", zero),
        ("positive", positive),
    ):
        if mask.sum() == 0:
            out[name] = None
            continue
        p, t = predicted[mask], target[mask]
        out[name] = {
            "count": int(mask.sum()),
            "model_mae": float((p - t).abs().mean()),
            "zero_baseline_mae": float(t.abs().mean()),
            "mean_baseline_mae": float((t - mean).abs().mean()),
            "model_mean": float(p.mean()),
            "target_mean": float(t.mean()),
        }
    if positive.any() and zero.any():
        # Does the head rank rewarded steps above unrewarded ones?
        p_pos, p_zero = predicted[positive], predicted[zero]
        sample = p_zero[
            torch.randperm(len(p_zero), generator=torch.Generator().manual_seed(0))[
                :4096
            ]
        ]
        out["positive_above_zero_rate"] = float(
            (p_pos.unsqueeze(1) > sample.unsqueeze(0)).float().mean()
        )
    return out


def isfinite(values: list[float]) -> bool:
    return all(math.isfinite(v) for v in values)
