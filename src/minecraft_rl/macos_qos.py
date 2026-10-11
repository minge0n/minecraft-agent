"""Keep a training process on the efficiency cores of a Mac.

macOS runs a process with the background policy (`PRIO_DARWIN_BG`) on the
efficiency cores at a low clock speed. The same work then needs about a
third of the energy of the performance cores, so the chip makes less heat.
The duty cycle of `sweep.py` cannot do that: a pause moves the same energy to
a later time. Measured on an M5 Pro for one Stage 3 RSSM update: 1.21 J at
default priority against 0.28 J in the background
(docs/stage3-minecraft-world-model.md).

`run_in_background` sets the policy for the whole process, the same as
`taskpolicy -b`, so a run started directly gets it too. It changes only where
and how fast the process computes, never what it computes: the canonical
runtime gives the same bits on both kinds of core.

This module supports macOS only. On other systems it does nothing.
"""

import os
import sys


def background_policy() -> bool | None:
    """True if the process runs with the background policy, None where the
    platform has no such policy."""
    if sys.platform != "darwin":
        return None
    return os.getpriority(os.PRIO_DARWIN_PROCESS, 0) == 1


def run_in_background() -> bool | None:
    """Move this process to the background policy. Returns the new state."""
    if sys.platform != "darwin":
        return None
    os.setpriority(os.PRIO_DARWIN_PROCESS, 0, os.PRIO_DARWIN_BG)
    return background_policy()
