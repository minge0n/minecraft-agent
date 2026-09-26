package com.mcbot;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Optional;
import org.junit.jupiter.api.Test;

class TickGateTest {
    private static TickGate pendingGate(long gameTime, long clientTicks) {
        TickGate gate = new TickGate();
        assertTrue(gate.requestStep(gameTime, clientTicks, true, false, true).isEmpty());
        return gate;
    }

    @Test
    void stepWaitsForClientTickThenExactlyOneServerTick() {
        TickGate gate = pendingGate(40, 7);
        assertFalse(gate.clientTickDelivered(7, 3, 3));
        assertEquals(Optional.empty(), gate.observeTickEnd(41, 8));
        assertTrue(gate.clientTickDelivered(8, 4, 4));
        assertEquals(Optional.empty(), gate.observeTickEnd(40, 8));
        assertEquals(Optional.of(new TickGate.Step(1, 40, 41, 8)), gate.observeTickEnd(41, 8));
        assertEquals(Optional.empty(), gate.observeTickEnd(42, 8));
    }

    @Test
    void serverTickWaitsUntilClientTickEndPacketIsProcessed() {
        TickGate gate = pendingGate(40, 7);
        assertFalse(gate.clientTickDelivered(8, 4, 3));
        assertTrue(gate.clientTickDelivered(8, 4, 4));
        assertFalse(gate.clientTickDelivered(8, 4, 4));
    }

    @Test
    void stepIdsIncreaseMonotonically() {
        TickGate gate = pendingGate(0, 0);
        gate.clientTickDelivered(1, 1, 1);
        gate.observeTickEnd(1, 1);
        gate.requestStep(1, 1, true, false, true);
        gate.clientTickDelivered(2, 2, 2);
        assertEquals(Optional.of(new TickGate.Step(2, 1, 2, 2)), gate.observeTickEnd(2, 2));
    }

    @Test
    void rejectsPausedUnfrozenUngatedAndConcurrentSteps() {
        TickGate gate = new TickGate();
        assertEquals(Optional.of(new TickGate.Rejected("singleplayer_paused")), gate.requestStep(0, 0, true, true, true));
        assertEquals(Optional.of(new TickGate.Rejected("not_frozen")), gate.requestStep(0, 0, false, false, true));
        assertEquals(Optional.of(new TickGate.Rejected("client_not_gated")), gate.requestStep(0, 0, true, false, false));
        assertTrue(gate.requestStep(0, 0, true, false, true).isEmpty());
        assertEquals(Optional.of(new TickGate.Rejected("step_pending")), gate.requestStep(0, 0, true, false, true));
    }

    @Test
    void reportsServerOrClientOverAdvance() {
        TickGate server = pendingGate(10, 5);
        server.clientTickDelivered(6, 1, 1);
        assertEquals(Optional.of(new TickGate.Rejected("advanced_2_ticks")), server.observeTickEnd(12, 6));

        TickGate client = pendingGate(10, 5);
        client.clientTickDelivered(7, 1, 1);
        assertEquals(Optional.of(new TickGate.Rejected("client_advanced_2_ticks")), client.observeTickEnd(11, 7));
    }

    @Test
    void cancelClearsPendingStep() {
        TickGate gate = pendingGate(5, 0);
        assertTrue(gate.cancelPendingStep());
        assertFalse(gate.cancelPendingStep());
        assertFalse(gate.clientTickDelivered(1, 1, 1));
        assertEquals(Optional.empty(), gate.observeTickEnd(6, 1));
        assertTrue(gate.requestStep(6, 1, true, false, true).isEmpty());
    }

    @Test
    void encodesProtocolLines() {
        assertEquals("v1 STATUS 7 90 true false true 12",
                TickGate.encode(new TickGate.Status(7, 90, true, false, true, 12)));
        assertEquals("v1 STEP 3 7 8 12", TickGate.encode(new TickGate.Step(3, 7, 8, 12)));
        assertEquals("v1 ERROR not_frozen", TickGate.encode(new TickGate.Rejected("not_frozen")));
    }
}
