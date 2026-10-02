"""T-maze cue-memory environment for Stages 2C-2F (docs/stage2c.md).

The agent starts on a cell showing a cue, walks a corridor whose cells all look the
same, and must turn at the junction toward the side the cue indicated.
"""

from enum import IntEnum


class Observation(IntEnum):
    CUE_LEFT = 0
    CUE_RIGHT = 1
    CORRIDOR = 2
    JUNCTION = 3
    ARM = 4


class Action(IntEnum):
    FORWARD = 0
    LEFT = 1
    RIGHT = 2


class Cue(IntEnum):
    LEFT = 0
    RIGHT = 1


class TMaze:
    """Positions: 0 is the cue cell, 1..corridor_length the corridor, then the
    junction. FORWARD moves one cell toward the junction; LEFT and RIGHT bump into
    the corridor wall and stay. At the junction LEFT or RIGHT enters an arm and ends
    the episode with reward +1 on the cued side and -1 on the other."""

    def __init__(self, corridor_length: int = 4, max_steps: int = 40) -> None:
        if corridor_length < 1 or max_steps < 1:
            raise ValueError("corridor_length and max_steps must be positive")
        self.corridor_length = corridor_length
        self.max_steps = max_steps
        self.junction = corridor_length + 1
        self.cue = Cue.LEFT
        self.position = 0
        self.steps = 0
        self.done = True

    def reset(self, cue: Cue) -> Observation:
        self.cue = cue
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
        truncated = self.steps >= self.max_steps
        self.done = truncated
        return self._observation(), 0.0, False, truncated

    def _observation(self) -> Observation:
        if self.position == 0:
            return (
                Observation.CUE_LEFT if self.cue == Cue.LEFT else Observation.CUE_RIGHT
            )
        if self.position < self.junction:
            return Observation.CORRIDOR
        return Observation.JUNCTION
