"""`reward-pitch-v1`: a diagnostic camera-control reward
(docs/stage4-pitch-control.md).

It asks whether the Dreamer actor can learn one action-conditioned control
task when the reward depends only on a value that the policy observes. The
reward of transition t reads only the next observation:

    reward[t] = -|pitch(o_{t+1}) - 30| / 90

It is 0 at the target pitch +30 degrees (looking down) and falls linearly to
-1.33 at -90 degrees. It needs no position, no yaw, no event and no memory.

For this diagnostic only, the actor chooses the pitch factor of the existing
physical action space; every other factor stays at its no-op value
(`PITCH_ONLY_MASK`). The camera action semantics are unchanged.
"""

from dataclasses import asdict, dataclass

import numpy
import torch

from minecraft_rl.minecraft_actor import ActionSpace, Agent

PITCH_VERSION = "reward-pitch-v1"


@dataclass(frozen=True)
class PitchTask:
    version: str = PITCH_VERSION
    target: float = 30.0
    scale: float = 90.0
    # Evaluation only: "near the target".
    near: float = 5.0

    def to_json(self) -> dict:
        return asdict(self)


PITCH_V1 = PitchTask()


def pitch_rewards(pitch: numpy.ndarray, task: PitchTask = PITCH_V1) -> numpy.ndarray:
    """Rewards (T,) for the observed pitches (T + 1,): row t uses pitch t + 1."""
    return (
        -numpy.abs(pitch[1:].astype(numpy.float64) - task.target) / task.scale
    ).astype(numpy.float32)


def pitch_metrics(pitch: numpy.ndarray, task: PitchTask = PITCH_V1) -> dict:
    error = numpy.abs(pitch.astype(numpy.float64) - task.target)
    return {
        "start_pitch": float(pitch[0]),
        "final_pitch": float(pitch[-1]),
        "final_error": float(error[-1]),
        "mean_error": float(error[1:].mean()),
        "near_fraction": float((error[1:] <= task.near).mean()),
        "return": float(pitch_rewards(pitch, task).sum()),
    }


PITCH_FACTOR = ActionSpace.NAMES.index("pitch")


class PitchOnlyAgent(Agent):
    """The existing actor and critic, with every factor except pitch fixed at
    its no-op choice. The pitch head is the only head that learns; the others
    get no gradient because their choice is fixed and their log probability
    and entropy are left out."""

    def act(self, s, generator, greedy: bool = False):
        logits = self.logits(s)[PITCH_FACTOR]
        if greedy:
            pitch = logits.argmax(-1)
        else:
            p = torch.softmax(logits, -1).reshape(-1, logits.shape[-1])
            pitch = torch.multinomial(p, 1, generator=generator).reshape(
                logits.shape[:-1]
            )
        noop = torch.as_tensor(ActionSpace.NOOP).expand(
            *pitch.shape, len(ActionSpace.NOOP)
        )
        choices = noop.clone()
        choices[..., PITCH_FACTOR] = pitch
        return choices

    def log_prob_entropy(self, s, choices):
        logits = self.logits(s)[PITCH_FACTOR]
        log_p = torch.log_softmax(logits, -1)
        log_prob = log_p.gather(
            -1, choices[..., PITCH_FACTOR : PITCH_FACTOR + 1]
        ).squeeze(-1)
        entropy = -(log_p.exp() * log_p).sum(-1)
        zero = torch.zeros_like(entropy)
        per_factor = [
            entropy if k == PITCH_FACTOR else zero
            for k in range(len(ActionSpace.SIZES))
        ]
        return log_prob, entropy, per_factor
