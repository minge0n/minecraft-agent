"""A T-maze with a random signal for Stage 2H (docs/stage2h.md).

The episode starts on a plain cell that looks like the corridor. The first
FORWARD move reaches the signal cell. Only then does the environment draw the
signal: CUE_LEFT with probability `signal_left_probability`, else CUE_RIGHT.
Plain corridor cells follow, then the junction. A turn toward the signalled
side gives +1, the other turn -1.

So the same history (start, FORWARD) can lead to two different futures, and
the later reward depends on which one happened. A world model must predict the
signal as a distribution and keep each imagined future consistent with its own
signal. The drawn signal is privileged state (`cue`) for evaluation only.
Randomness comes only from the generator given to `reset`.
"""

import torch

from minecraft_rl.tmaze import Action, Cue, Observation


class SignalTMaze:
    """Positions: 0 start, 1 signal cell, 2 .. corridor_length + 1 corridor,
    then the junction. Observations and actions are those of `TMaze`."""

    def __init__(
        self,
        corridor_length: int = 4,
        max_steps: int = 40,
        signal_left_probability: float = 0.7,
    ) -> None:
        if corridor_length < 1 or max_steps < 1:
            raise ValueError("corridor_length and max_steps must be positive")
        if not 0.0 <= signal_left_probability <= 1.0:
            raise ValueError("signal_left_probability must be in [0, 1]")
        self.corridor_length = corridor_length
        self.max_steps = max_steps
        self.signal_left_probability = signal_left_probability
        self.junction = corridor_length + 2
        self.cue = Cue.LEFT
        self.generator: torch.Generator | None = None
        self.signal_drawn = False
        self.position = 0
        self.steps = 0
        self.done = True

    def reset(self, generator: torch.Generator) -> Observation:
        """Start an episode. The signal is drawn later from `generator`."""
        self.generator = generator
        self.signal_drawn = False
        self.position = 0
        self.steps = 0
        self.done = False
        return self._observation()

    def step(self, action: Action) -> tuple[Observation, float, bool, bool]:
        if self.done:
            raise RuntimeError("Reset before stepping after the episode ends")
        self.steps += 1

        if self.position == self.junction and action != Action.FORWARD:
            self.done = True
            cued_turn = Action.LEFT if self.cue == Cue.LEFT else Action.RIGHT
            return Observation.ARM, 1.0 if action == cued_turn else -1.0, True, False

        if action == Action.FORWARD and self.position < self.junction:
            self.position += 1
        if self.position == 1 and not self.signal_drawn:
            draw = torch.rand(1, generator=self.generator).item()
            self.cue = Cue.LEFT if draw < self.signal_left_probability else Cue.RIGHT
            self.signal_drawn = True
        truncated = self.steps >= self.max_steps
        self.done = truncated
        return self._observation(), 0.0, False, truncated

    def _observation(self) -> Observation:
        if self.position == 1:
            return (
                Observation.CUE_LEFT if self.cue == Cue.LEFT else Observation.CUE_RIGHT
            )
        if self.position == self.junction:
            return Observation.JUNCTION
        return Observation.CORRIDOR
