"""`reward-break-v1`: a diagnostic block-breaking reward
(docs/stage4-block-breaking.md).

It asks whether the Dreamer actor can learn to break a real Minecraft block
with the existing camera and attack controls. The reward of transition t is

    reward[t] = 1.0 * (number of blocks broken by the player in step t)

from the environment's privileged block-break event (`StepEvents`), which is
never a policy input. Every break pays, the first and the hundredth alike:
no episode memory, no reward for attack presses or camera turns, nothing for
an attempt that did not finish.

For this diagnostic only, the actor chooses the pitch and attack factors of
the existing physical action space (`BreakAgent`); every other factor stays
at its no-op value. With the fixed spawn of world seed 100000 the player
stands on sand, which breaks after 15 held attack ticks with an empty hand
when the player looks straight down.
"""

from dataclasses import asdict, dataclass

import numpy

from minecraft_rl.minecraft_actor import ActionSpace
from minecraft_rl.minecraft_reward_pitch import PITCH_FACTOR, RestrictedAgent

BREAK_VERSION = "reward-break-v1"
ATTACK_FACTOR = ActionSpace.NAMES.index("attack")
# Episode array: blocks broken per step, from the privileged event.
BREAKS_KEY = "privileged_breaks"


@dataclass(frozen=True)
class BreakTask:
    version: str = BREAK_VERSION
    per_break: float = 1.0

    def to_json(self) -> dict:
        return asdict(self)


BREAK_V1 = BreakTask()


def break_rewards(breaks: numpy.ndarray, task: BreakTask = BREAK_V1) -> numpy.ndarray:
    """Rewards (T,) for the number of blocks broken in each step (T,)."""
    return (task.per_break * breaks.astype(numpy.float64)).astype(numpy.float32)


def attack_runs(attack: numpy.ndarray) -> list[int]:
    """Lengths of the runs of consecutive held-attack steps."""
    runs, current = [], 0
    for held in attack.astype(bool):
        if held:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def break_metrics(breaks: numpy.ndarray, attack: numpy.ndarray, broken_types) -> dict:
    runs = attack_runs(attack)
    at = numpy.flatnonzero(breaks > 0)
    # Attack ticks held in the run that ends with each break.
    ticks_before = []
    for t in at:
        n = 0
        while t - n >= 0 and attack[t - n]:
            n += 1
        ticks_before.append(n)
    return {
        "breaks": int(breaks.sum()),
        "unique_types": len(set(broken_types)),
        "first_break_step": int(at[0]) if len(at) else None,
        "attack_fraction": float(attack.mean()),
        "attack_runs": len(runs),
        "longest_attack_run": max(runs, default=0),
        "runs_of_15_or_more": sum(r >= 15 for r in runs),
        "attack_ticks_before_break": ticks_before,
    }


class BreakAgent(RestrictedAgent):
    """The block-breaking diagnostic: pitch and attack are free."""

    FREE = (PITCH_FACTOR, ATTACK_FACTOR)
