"""`reward-control-v1`: a diagnostic movement reward (docs/stage4-control-test.md).

It asks one question: can the Dreamer actor learn a real physical control task
in Minecraft when the reward depends only on the current transition? It is not
a progression reward and does not replace `reward-v1`.

The task is fixed by the start of the episode. The target lies `distance`
blocks ahead of the start position along the start facing direction, in the
horizontal plane. The reward of transition t is

    reward[t] = scale * (d(p_t) - d(p_{t+1})),  d(p) = |(p.x, p.z) - target|

so it is positive for movement toward the target, negative for movement away,
and zero for standing still or turning in place. It telescopes: the sum over
an episode is scale * (d(p_0) - d(p_T)), so a loop that returns to the same
place earns exactly zero in total. There is no arrival bonus and no memory of
earlier positions.

The positions come from the privileged `DEBUG_PLAYER` request and are stored
as `privileged_*` arrays, never as `obs_*` arrays, so they never reach the
policy or the world model.
"""

import math
from dataclasses import asdict, dataclass

import numpy

CONTROL_VERSION = "reward-control-v1"
# Episode arrays: horizontal position x, z and the yaw, steps + 1 rows each.
POSITION_KEY = "privileged_position"
YAW_KEY = "privileged_yaw"


@dataclass(frozen=True)
class ControlTask:
    version: str = CONTROL_VERSION
    distance: float = 20.0
    scale: float = 1.0
    # For the evaluation metric "reached the target" only; never a reward.
    reach_radius: float = 1.5

    def to_json(self) -> dict:
        return asdict(self)


CONTROL_V1 = ControlTask()


def facing(yaw_degrees: float) -> tuple[float, float]:
    """The horizontal unit vector (x, z) of a Minecraft yaw: 0 is south (+z),
    90 is west (-x)."""
    yaw = math.radians(yaw_degrees)
    return -math.sin(yaw), math.cos(yaw)


def target_of(
    start: numpy.ndarray, start_yaw: float, task: ControlTask
) -> numpy.ndarray:
    dx, dz = facing(start_yaw)
    return numpy.array([start[0] + task.distance * dx, start[1] + task.distance * dz])


def distances(positions: numpy.ndarray, target: numpy.ndarray) -> numpy.ndarray:
    """Horizontal distance of each position (..., 2) to the target (2,)."""
    return numpy.linalg.norm(positions - target, axis=-1)


def control_rewards(
    positions: numpy.ndarray, target: numpy.ndarray, task: ControlTask
) -> numpy.ndarray:
    """Rewards (T,) for positions (T + 1, 2)."""
    d = distances(positions.astype(numpy.float64), target)
    return (task.scale * (d[:-1] - d[1:])).astype(numpy.float32)


def episode_task(arrays: dict, task: ControlTask = CONTROL_V1) -> dict:
    """Target, rewards and evaluation metrics of one recorded episode."""
    positions = arrays[POSITION_KEY].astype(numpy.float64)
    target = target_of(positions[0], float(arrays[YAW_KEY][0]), task)
    d = distances(positions, target)
    rewards = control_rewards(positions, target, task)
    return {
        "target": target,
        "rewards": rewards,
        "start_distance": float(d[0]),
        "final_distance": float(d[-1]),
        "min_distance": float(d.min()),
        "net_progress": float(d[0] - d[-1]),
        "reached": bool(d.min() <= task.reach_radius),
        "return": float(rewards.sum()),
    }
