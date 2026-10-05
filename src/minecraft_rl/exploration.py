"""High-entropy exploration policy for collecting Minecraft world-model data.

This policy is a data-coverage tool, not a Minecraft strategy. It never reads
the observation, so it cannot know where resources are, what items are worth,
what to craft or how to progress. It produces physically plausible motor
behavior instead of independent random key bits:

- Movement is one state out of 9 (still, or one of 8 directions). Forward and
  back, or left and right, are never pressed together.
- Each factor keeps its value for a random number of ticks (action
  persistence), so walking, mining and holding use last long enough to have
  an effect.
- The camera follows a bounded random walk: the turn rate drifts and decays,
  so the view sweeps instead of jittering.
- Jump, sneak, sprint, attack and use are occasional and persistent.
- The hotbar selection changes rarely.

All randomness comes from one `numpy.random.Generator`, so a seed gives the
same action sequence. The parameters are recorded with every dataset.
"""

from dataclasses import asdict, dataclass

import numpy

from minecraft_rl.minecraft_interface import (
    HOTBAR_SLOTS,
    KEEP_HOTBAR_SLOT,
    MAX_CAMERA_DELTA_DEGREES,
    PlayerAction,
)

# Movement states: (forward, back, left, right). Index 0 stands still.
MOVEMENTS = (
    (False, False, False, False),
    (True, False, False, False),
    (True, False, True, False),
    (True, False, False, True),
    (False, False, True, False),
    (False, False, False, True),
    (False, True, False, False),
    (False, True, True, False),
    (False, True, False, True),
)


@dataclass(frozen=True)
class ExplorationConfig:
    """Probabilities are per tick unless named otherwise. A `*_hold` value is
    the mean number of ticks that a choice persists (geometric)."""

    movement_hold: float = 20.0
    movement_weights: tuple[float, ...] = (2.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5)
    yaw_rate_noise: float = 2.0
    pitch_rate_noise: float = 1.0
    rate_decay: float = 0.85
    max_rate_degrees: float = 12.0
    pitch_center_pull: float = 0.02
    jump_start: float = 0.04
    jump_hold: float = 6.0
    sprint_start: float = 0.01
    sprint_hold: float = 40.0
    sneak_start: float = 0.005
    sneak_hold: float = 20.0
    attack_start: float = 0.02
    attack_hold: float = 40.0
    use_start: float = 0.01
    use_hold: float = 15.0
    hotbar_change: float = 0.005

    def __post_init__(self) -> None:
        if len(self.movement_weights) != len(MOVEMENTS):
            raise ValueError("movement_weights needs one weight per movement state")
        if not 0.0 < self.max_rate_degrees <= MAX_CAMERA_DELTA_DEGREES:
            raise ValueError("max_rate_degrees must lie in (0, 45]")

    def to_json(self) -> dict:
        return asdict(self)


class _Toggle:
    """A boolean that turns on with probability `start` per tick and then
    stays on for a geometric number of ticks with mean `hold`."""

    def __init__(self, start: float, hold: float) -> None:
        self.start = start
        self.stop = 1.0 / max(hold, 1.0)
        self.on = False

    def step(self, random: numpy.random.Generator) -> bool:
        if self.on:
            self.on = random.random() >= self.stop
        else:
            self.on = random.random() < self.start
        return self.on


class ExplorationPolicy:
    """Stateful random action process. Call `reset()` at each episode start
    and `act()` once per tick. The pitch is the observed camera pitch, used
    only to pull the random walk away from looking straight up or down."""

    def __init__(self, config: ExplorationConfig, seed: int) -> None:
        self.config = config
        self.seed = seed
        self.random = numpy.random.default_rng(seed)
        weights = numpy.asarray(config.movement_weights, dtype=float)
        self._movement_p = weights / weights.sum()
        self.reset()

    def reset(self) -> None:
        c = self.config
        self.movement = 0
        self.yaw_rate = 0.0
        self.pitch_rate = 0.0
        self.jump = _Toggle(c.jump_start, c.jump_hold)
        self.sprint = _Toggle(c.sprint_start, c.sprint_hold)
        self.sneak = _Toggle(c.sneak_start, c.sneak_hold)
        self.attack = _Toggle(c.attack_start, c.attack_hold)
        self.use = _Toggle(c.use_start, c.use_hold)

    def act(self, pitch: float = 0.0) -> PlayerAction:
        c = self.config
        r = self.random
        if r.random() < 1.0 / max(c.movement_hold, 1.0):
            self.movement = int(r.choice(len(MOVEMENTS), p=self._movement_p))
        forward, back, left, right = MOVEMENTS[self.movement]

        limit = c.max_rate_degrees
        self.yaw_rate = c.rate_decay * self.yaw_rate + r.normal(0.0, c.yaw_rate_noise)
        self.pitch_rate = (
            c.rate_decay * self.pitch_rate
            + r.normal(0.0, c.pitch_rate_noise)
            - c.pitch_center_pull * pitch
        )
        self.yaw_rate = float(numpy.clip(self.yaw_rate, -limit, limit))
        self.pitch_rate = float(numpy.clip(self.pitch_rate, -limit, limit))

        sneak = self.sneak.step(r)
        sprint = self.sprint.step(r) and forward and not sneak
        hotbar = KEEP_HOTBAR_SLOT
        if r.random() < c.hotbar_change:
            hotbar = int(r.integers(HOTBAR_SLOTS))
        return PlayerAction(
            forward=forward,
            back=back,
            left=left,
            right=right,
            jump=self.jump.step(r),
            sneak=sneak,
            sprint=sprint,
            attack=self.attack.step(r),
            use=self.use.step(r),
            yaw_delta=self.yaw_rate,
            pitch_delta=self.pitch_rate,
            hotbar=hotbar,
        )
