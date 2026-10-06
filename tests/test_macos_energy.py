import platform

import pytest

from minecraft_rl import macos_energy


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS counters")
def test_energy_counters_grow_with_work():
    before = macos_energy.read()
    assert before is not None
    total = sum(i * i for i in range(2_000_000))
    after = macos_energy.read()
    used = after - before
    assert total > 0
    assert used.instructions > 0
    assert used.energy_joules >= 0
    assert 0 <= used.performance_core_energy_joules <= used.energy_joules + 1e-9


def test_a_missing_process_reports_none():
    assert macos_energy.read(pid=2**31 - 2) is None
