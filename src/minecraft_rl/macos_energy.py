"""Energy and CPU counters of a process on macOS.

macOS counts the energy that each process uses on the CPU, without root
rights, in `proc_pid_rusage` with `RUSAGE_INFO_V6`: `ri_energy_nj` is the
energy of the process on all cores in nanojoules, and `ri_penergy_nj` the
part on the performance cores. The same record counts instructions and
cycles, in total and on the performance cores.

Heat comes from energy. A pause (the duty cycle of `sweep.py`) spreads the
same energy over a longer time, but does not make it smaller. These counters
measure what does make it smaller: less work per update, or work on the
more efficient efficiency cores. This module supports macOS only.
"""

import ctypes
import ctypes.util
import os
from dataclasses import dataclass

RUSAGE_INFO_V6 = 6


class _RusageInfoV6(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        *[
            (name, ctypes.c_uint64)
            for name in (
                "ri_user_time",
                "ri_system_time",
                "ri_pkg_idle_wkups",
                "ri_interrupt_wkups",
                "ri_pageins",
                "ri_wired_size",
                "ri_resident_size",
                "ri_phys_footprint",
                "ri_proc_start_abstime",
                "ri_proc_exit_abstime",
                "ri_child_user_time",
                "ri_child_system_time",
                "ri_child_pkg_idle_wkups",
                "ri_child_interrupt_wkups",
                "ri_child_pageins",
                "ri_child_elapsed_abstime",
                "ri_diskio_bytesread",
                "ri_diskio_byteswritten",
                "ri_cpu_time_qos_default",
                "ri_cpu_time_qos_maintenance",
                "ri_cpu_time_qos_background",
                "ri_cpu_time_qos_utility",
                "ri_cpu_time_qos_legacy",
                "ri_cpu_time_qos_user_initiated",
                "ri_cpu_time_qos_user_interactive",
                "ri_billed_system_time",
                "ri_serviced_system_time",
                "ri_logical_writes",
                "ri_lifetime_max_phys_footprint",
                "ri_instructions",
                "ri_cycles",
                "ri_billed_energy",
                "ri_serviced_energy",
                "ri_interval_max_phys_footprint",
                "ri_runnable_time",
                "ri_flags",
                "ri_user_ptime",
                "ri_system_ptime",
                "ri_pinstructions",
                "ri_pcycles",
                "ri_energy_nj",
                "ri_penergy_nj",
                "ri_secure_time_in_system",
                "ri_secure_ptime_in_system",
                "ri_neural_footprint",
                "ri_lifetime_max_neural_footprint",
                "ri_interval_max_neural_footprint",
            )
        ],
        ("ri_reserved", ctypes.c_uint64 * 9),
    ]


@dataclass(frozen=True)
class Counters:
    energy_joules: float
    performance_core_energy_joules: float
    instructions: int
    performance_core_instructions: int
    cycles: int

    def __sub__(self, other: "Counters") -> "Counters":
        return Counters(
            self.energy_joules - other.energy_joules,
            self.performance_core_energy_joules - other.performance_core_energy_joules,
            self.instructions - other.instructions,
            self.performance_core_instructions - other.performance_core_instructions,
            self.cycles - other.cycles,
        )


_library = None


def _proc_pid_rusage():
    global _library
    if _library is None:
        path = ctypes.util.find_library("proc") or "/usr/lib/libSystem.B.dylib"
        _library = ctypes.CDLL(path, use_errno=True)
        _library.proc_pid_rusage.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(_RusageInfoV6),
        ]
        _library.proc_pid_rusage.restype = ctypes.c_int
    return _library.proc_pid_rusage


def read(pid: int | None = None) -> Counters | None:
    """The counters of a process (default: this one), or None where the
    platform does not provide them."""
    try:
        function = _proc_pid_rusage()
    except (OSError, AttributeError):
        return None
    info = _RusageInfoV6()
    if function(
        os.getpid() if pid is None else pid, RUSAGE_INFO_V6, ctypes.byref(info)
    ):
        return None
    return Counters(
        info.ri_energy_nj / 1e9,
        info.ri_penergy_nj / 1e9,
        info.ri_instructions,
        info.ri_pinstructions,
        info.ri_cycles,
    )
