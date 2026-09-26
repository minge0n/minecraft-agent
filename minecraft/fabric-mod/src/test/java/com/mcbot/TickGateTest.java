package com.mcbot;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Optional;
import org.junit.jupiter.api.Test;

class TickGateTest {
    @Test
    void acceptedStepCompletesOnlyAfterExactlyOneTick() {
        TickGate gate = new TickGate();
        assertTrue(gate.requestStep(40, true, false).isEmpty());
        assertEquals(Optional.empty(), gate.observeTickEnd(40));
        assertEquals(Optional.of(new TickGate.Step(1, 40, 41)), gate.observeTickEnd(41));
        assertEquals(Optional.empty(), gate.observeTickEnd(42));
    }

    @Test
    void stepIdsIncreaseMonotonically() {
        TickGate gate = new TickGate();
        gate.requestStep(0, true, false);
        gate.observeTickEnd(1);
        gate.requestStep(1, true, false);
        assertEquals(Optional.of(new TickGate.Step(2, 1, 2)), gate.observeTickEnd(2));
    }

    @Test
    void rejectsPausedUnfrozenAndConcurrentSteps() {
        TickGate gate = new TickGate();
        assertEquals(Optional.of(new TickGate.Rejected("singleplayer_paused")), gate.requestStep(0, true, true));
        assertEquals(Optional.of(new TickGate.Rejected("not_frozen")), gate.requestStep(0, false, false));
        assertTrue(gate.requestStep(0, true, false).isEmpty());
        assertEquals(Optional.of(new TickGate.Rejected("step_pending")), gate.requestStep(0, true, false));
    }

    @Test
    void reportsAdvanceOfMoreThanOneTick() {
        TickGate gate = new TickGate();
        gate.requestStep(10, true, false);
        assertEquals(Optional.of(new TickGate.Rejected("advanced_2_ticks")), gate.observeTickEnd(12));
    }

    @Test
    void cancelClearsPendingStep() {
        TickGate gate = new TickGate();
        gate.requestStep(5, true, false);
        assertTrue(gate.cancelPendingStep());
        assertFalse(gate.cancelPendingStep());
        assertEquals(Optional.empty(), gate.observeTickEnd(6));
        assertTrue(gate.requestStep(6, true, false).isEmpty());
    }

    @Test
    void encodesProtocolLines() {
        assertEquals("v1 STATUS 7 90 true false", TickGate.encode(new TickGate.Status(7, 90, true, false)));
        assertEquals("v1 STEP 3 7 8", TickGate.encode(new TickGate.Step(3, 7, 8)));
        assertEquals("v1 ERROR not_frozen", TickGate.encode(new TickGate.Rejected("not_frozen")));
    }
}
