import numpy
import pytest

from minecraft_rl.exploration import MOVEMENTS, ExplorationConfig, ExplorationPolicy
from minecraft_rl.minecraft_interface import KEEP_HOTBAR_SLOT

DEFAULT = ExplorationConfig()


def actions(seed: int, count: int, config=DEFAULT):
    policy = ExplorationPolicy(config, seed)
    return [policy.act() for _ in range(count)]


def test_a_seed_gives_the_same_action_sequence():
    assert actions(3, 500) == actions(3, 500)
    assert actions(3, 500) != actions(4, 500)


def test_opposite_keys_are_never_pressed_together():
    for action in actions(0, 5000):
        assert not (action.forward and action.back)
        assert not (action.left and action.right)
        assert not (action.sprint and (action.sneak or not action.forward))


def test_camera_deltas_stay_inside_the_action_bound():
    config = ExplorationConfig()
    for action in actions(1, 5000, config):
        assert abs(action.yaw_delta) <= config.max_rate_degrees
        assert abs(action.pitch_delta) <= config.max_rate_degrees


def test_choices_persist_for_several_ticks():
    sequence = actions(2, 20000)

    def mean_run(values):
        runs, length = [], 1
        for previous, current in zip(values, values[1:], strict=False):
            if current == previous:
                length += 1
            else:
                runs.append(length)
                length = 1
        return numpy.mean(runs)

    movements = [(a.forward, a.back, a.left, a.right) for a in sequence]
    assert mean_run(movements) > 8
    attacks = [a.attack for a in sequence]
    attack_runs = [length for length, on in _runs(attacks) if on]
    assert attack_runs and numpy.mean(attack_runs) > 10


def _runs(values):
    out, length = [], 1
    for previous, current in zip(values, values[1:], strict=False):
        if current == previous:
            length += 1
        else:
            out.append((length, previous))
            length = 1
    out.append((length, values[-1]))
    return out


def test_every_action_factor_occurs():
    sequence = actions(5, 20000)
    for name in (
        "forward",
        "back",
        "left",
        "right",
        "jump",
        "sneak",
        "sprint",
        "attack",
        "use",
    ):
        assert any(getattr(a, name) for a in sequence), name
    assert any(a.hotbar != KEEP_HOTBAR_SLOT for a in sequence)
    still = sum(not (a.forward or a.back or a.left or a.right) for a in sequence)
    assert 0 < still < len(sequence)


def test_pitch_pull_turns_the_camera_back_toward_the_horizon():
    policy = ExplorationPolicy(ExplorationConfig(pitch_rate_noise=0.0), 0)
    assert policy.act(pitch=80.0).pitch_delta < 0
    policy = ExplorationPolicy(ExplorationConfig(pitch_rate_noise=0.0), 0)
    assert policy.act(pitch=-80.0).pitch_delta > 0


def test_invalid_configuration_is_refused():
    with pytest.raises(ValueError):
        ExplorationConfig(movement_weights=(1.0,) * (len(MOVEMENTS) - 1))
    with pytest.raises(ValueError):
        ExplorationConfig(max_rate_degrees=50.0)
