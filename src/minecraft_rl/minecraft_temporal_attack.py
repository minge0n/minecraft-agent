"""Temporal attack commands for the block-breaking diagnostic
(docs/stage4-temporal-attack.md).

The actor of `reward-break-v1` chose attack independently every tick, and a
break needs about 15 consecutive attack ticks. Here the actor chooses an
attack command instead:

- `RELEASE_ATTACK`: one tick with attack off.
- `HOLD_ATTACK_20`: 20 consecutive ticks with attack on.

`AttackController` holds the number of forced hold ticks that remain. A new
command is accepted only when none remain (a decision tick). The pitch is
still chosen every tick, and every other factor stays at its no-op value.
Each tick is one physical action, one Minecraft tick and one RSSM transition,
in real play and in imagination alike (`temporal_step`).

`TemporalBreakAgent` is the existing actor and critic with one extra input,
the controller phase `remaining / HOLD_TICKS`, so that both know whether the
attack is forced. At a decision tick the physical attack value equals the
command (hold starts with attack on, release is attack off), so the attack
factor of a choice carries the command there. The command log probability
and entropy count only on decision ticks; the pitch counts on every tick.
"""

from typing import Any

import torch

from minecraft_rl.minecraft_actor import (
    ActionSpace,
    ActorConfig,
    Imagined,
    MinecraftRSSM,
    RewardHead,
    losses,
)
from minecraft_rl.minecraft_reward_break import ATTACK_FACTOR, BreakAgent
from minecraft_rl.minecraft_reward_pitch import PITCH_FACTOR

RELEASE_ATTACK = 0
HOLD_ATTACK_20 = 1
COMMANDS = ("RELEASE_ATTACK", "HOLD_ATTACK_20")
HOLD_TICKS = 20


class AttackController:
    """Forced hold ticks that remain, per rollout (n,). Zero is a decision
    tick."""

    def __init__(self, remaining: torch.Tensor) -> None:
        self.remaining = remaining.long()

    @classmethod
    def start(cls, n: int) -> "AttackController":
        return cls(torch.zeros(n, dtype=torch.long))

    @property
    def deciding(self) -> torch.Tensor:
        return self.remaining == 0

    def advance(self, command: torch.Tensor) -> torch.Tensor:
        """The physical attack of this tick. `command` is used only where
        the controller is deciding; elsewhere the hold continues."""
        deciding = self.deciding
        hold = deciding & (command == HOLD_ATTACK_20)
        attack = hold | ~deciding
        self.remaining = torch.where(
            hold,
            torch.full_like(self.remaining, HOLD_TICKS - 1),
            torch.where(deciding, torch.zeros_like(self.remaining), self.remaining - 1),
        )
        return attack


def _sample(
    logits: torch.Tensor, generator: torch.Generator | None, greedy: bool
) -> torch.Tensor:
    if greedy:
        return logits.argmax(-1)
    p = torch.softmax(logits, -1).reshape(-1, logits.shape[-1])
    return torch.multinomial(p, 1, generator=generator).reshape(logits.shape[:-1])


class TemporalBreakAgent(BreakAgent):
    def __init__(self, features: int, hidden: int = 256) -> None:
        super().__init__(features + 1, hidden)

    @staticmethod
    def inputs(s: torch.Tensor, remaining: torch.Tensor) -> torch.Tensor:
        phase = remaining.float().unsqueeze(-1) / HOLD_TICKS
        return torch.cat([s, phase], -1)

    @staticmethod
    def deciding(x: torch.Tensor) -> torch.Tensor:
        return x[..., -1] == 0

    def act(self, x, generator, greedy: bool = False):
        """Choices (..., 9): a pitch every tick, an attack command only on
        decision ticks. Forced hold ticks carry `HOLD_ATTACK_20` and draw no
        attack sample."""
        logits = self.logits(x)
        lead = x.shape[:-1]
        choices = torch.as_tensor(ActionSpace.NOOP).expand(*lead, len(ActionSpace.NOOP))
        choices = choices.clone()
        choices[..., PITCH_FACTOR] = _sample(logits[PITCH_FACTOR], generator, greedy)
        deciding = self.deciding(x)
        command = torch.full(lead, HOLD_ATTACK_20, dtype=torch.long)
        if deciding.any():
            command[deciding] = _sample(
                logits[ATTACK_FACTOR][deciding], generator, greedy
            )
        choices[..., ATTACK_FACTOR] = command
        return choices

    def log_prob_entropy(self, x, choices):
        logits = self.logits(x)
        deciding = self.deciding(x).float()
        per_factor = [torch.zeros(x.shape[:-1]) for _ in ActionSpace.SIZES]
        log_prob = torch.zeros(x.shape[:-1])
        for k, mask in ((PITCH_FACTOR, 1.0), (ATTACK_FACTOR, deciding)):
            log_p = torch.log_softmax(logits[k], -1)
            chosen = log_p.gather(-1, choices[..., k : k + 1]).squeeze(-1)
            log_prob = log_prob + mask * chosen
            per_factor[k] = mask * -(log_p.exp() * log_p).sum(-1)
        entropy = per_factor[PITCH_FACTOR] + per_factor[ATTACK_FACTOR]
        return log_prob, entropy, per_factor


def temporal_step(
    agent: TemporalBreakAgent,
    s: torch.Tensor,
    controller: AttackController,
    generator: torch.Generator | None,
    greedy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One tick: the agent input x, the physical choices (n, 9) and whether
    the tick was a decision tick. Advances the controller."""
    x = agent.inputs(s, controller.remaining)
    deciding = controller.deciding
    choices = agent.act(x, generator, greedy)
    choices[..., ATTACK_FACTOR] = controller.advance(choices[..., ATTACK_FACTOR]).long()
    return x, choices, deciding


@torch.no_grad()
def imagine_temporal(
    model: MinecraftRSSM,
    reward_head: RewardHead,
    agent: TemporalBreakAgent,
    h: torch.Tensor,
    z: torch.Tensor,
    horizon: int,
    generator: torch.Generator,
    remaining: torch.Tensor | None = None,
) -> Imagined:
    """`imagine` with the attack controller. Every tick is one RSSM
    transition with that tick's physical action. `features` holds the agent
    inputs [h, z, phase], so the critic's bootstrap value at the horizon
    knows a hold that is still running. Rollouts start at a decision tick
    unless `remaining` gives the controller state."""
    controller = AttackController(
        torch.zeros(h.shape[0], dtype=torch.long) if remaining is None else remaining
    )
    s = torch.cat([h, z], -1)
    inputs, choices, rewards, continues = [], [], [], []
    for _ in range(horizon):
        x, choice, _ = temporal_step(agent, s, controller, generator)
        h = model.transition(h, z, ActionSpace.world_model_action(choice))
        z = model.sample(model.prior_logits(h), generator, straight_through=False)
        s = torch.cat([h, z], -1)
        inputs.append(x)
        choices.append(choice)
        rewards.append(reward_head.reward(s))
        continues.append(torch.sigmoid(model.continue_head(s).squeeze(-1)))
    inputs.append(agent.inputs(s, controller.remaining))
    return Imagined(
        torch.stack(inputs, 1),
        torch.stack(choices, 1),
        torch.stack(rewards, 1),
        torch.stack(continues, 1),
    )


def train_temporal_actor(
    model: MinecraftRSSM,
    reward_head: RewardHead,
    agent: TemporalBreakAgent,
    optimizer: torch.optim.Optimizer,
    starts: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    config: ActorConfig,
    generator: torch.Generator,
    updates: int,
) -> list[dict[str, Any]]:
    """`train_actor` with the attack controller. `starts` is (h, z,
    remaining): the posterior states and the controller state that each
    start state is in."""
    h_all, z_all, remaining_all = starts
    history = []
    for _ in range(updates):
        index = torch.randint(
            0, h_all.shape[0], (config.start_states,), generator=generator
        )
        imagined = imagine_temporal(
            model,
            reward_head,
            agent,
            h_all[index],
            z_all[index],
            config.horizon,
            generator,
            remaining_all[index],
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
                "grad_norm": float(norm),
                "imagined_return": terms["imagined_return"],
                "imagined_reward_per_step": terms["imagined_reward_per_step"],
            }
            | command_statistics(agent, imagined)
        )
    return history


@torch.no_grad()
def command_statistics(agent: TemporalBreakAgent, imagined: Imagined) -> dict:
    """Decision-tick statistics of a batch of imagined rollouts: the share
    of decision ticks, the hold share among them, the command entropy on
    decision ticks only, the pitch entropy on all ticks and the mean critic
    value."""
    x = imagined.features[:, :-1]
    deciding = agent.deciding(x)
    _, _, per_factor = agent.log_prob_entropy(x, imagined.choices)
    held = imagined.choices[..., ATTACK_FACTOR] == HOLD_ATTACK_20
    count = int(deciding.sum())
    return {
        "decision_fraction": float(deciding.float().mean()),
        "hold_fraction_of_decisions": float(held[deciding].float().mean())
        if count
        else None,
        "command_entropy": float(per_factor[ATTACK_FACTOR][deciding].mean())
        if count
        else None,
        "pitch_entropy": float(per_factor[PITCH_FACTOR].mean()),
        "attack_fraction": float(held.float().mean()),
        "critic_value": float(agent.value(imagined.features).mean()),
    }
