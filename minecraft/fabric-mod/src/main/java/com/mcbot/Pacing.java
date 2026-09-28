package com.mcbot;

import java.util.Locale;

// How wall-clock time relates to granted lockstep steps. Pacing never changes what a
// step does: every STEP is exactly one client tick and one server tick in either mode.
public enum Pacing {
    // Vanilla scheduling: the server loop keeps its 50 ms tick deadline and the client
    // tick waits for the next render frame. Human-watchable reference behavior.
    PACED,
    // A pending step runs as soon as its prerequisites are ready: the server loop does
    // not sleep until its deadline, the step is granted before the tick-rate update of
    // the same server iteration, and the frame limiter is skipped while a client tick
    // permit is pending.
    UNPACED;

    static Pacing fromEnvironment() {
        String value = System.getenv().getOrDefault("MCBOT_PACING", "paced");
        return valueOf(value.toUpperCase(Locale.ROOT));
    }
}
