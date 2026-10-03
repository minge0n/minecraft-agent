import random

import numpy
import pytest
import torch

from minecraft_rl import resumable
from minecraft_rl.resumable import Incomplete, Session


def test_without_a_state_file_the_session_never_stops(tmp_path):
    session = Session(None, 0.0, {"run": 1})
    assert session.load() is None
    session.boundary(lambda: {"never": "saved"})
    assert not list(tmp_path.iterdir())


def test_a_session_saves_at_the_first_boundary_after_its_limit(tmp_path):
    path = tmp_path / "state.pt"
    session = Session(path, 0.0, {"run": 1})
    assert session.load() is None
    with pytest.raises(Incomplete):
        session.boundary(lambda: {"units": 1})
    resumed = Session(path, None, {"run": 1})
    assert resumed.load() == {"units": 1}
    assert resumed.sessions == 2
    resumed.finish()
    assert not path.exists()


def test_a_state_file_of_another_run_is_refused(tmp_path):
    path = tmp_path / "state.pt"
    with pytest.raises(Incomplete):
        Session(path, 0.0, {"run": 1}).boundary(lambda: {})
    with pytest.raises(ValueError, match="another run"):
        Session(path, None, {"run": 2}).load()


def test_global_random_generators_continue_where_they_stopped(tmp_path):
    path = tmp_path / "state.pt"
    random.seed(3)
    numpy.random.seed(3)
    torch.manual_seed(3)
    with pytest.raises(Incomplete):
        Session(path, 0.0, {}).boundary(lambda: {})
    expected = (random.random(), numpy.random.rand(), torch.rand(1).item())
    random.seed(9)
    numpy.random.seed(9)
    torch.manual_seed(9)
    session = Session(path, None, {})
    session.load()
    session.restore_global_random_state()
    assert (random.random(), numpy.random.rand(), torch.rand(1).item()) == expected


def test_a_generator_state_round_trips():
    generator = torch.Generator().manual_seed(4)
    torch.rand(5, generator=generator)
    copy = resumable.restore_generator(generator.get_state())
    assert torch.equal(
        torch.rand(3, generator=copy), torch.rand(3, generator=generator)
    )
