package com.mcbot;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class VisibleFieldSensorTest {
    @Test
    void airBubblesMatchTheHudRoundedUp() {
        assertEquals(10, VisibleFieldSensor.airBubbles(300, 300));
        assertEquals(10, VisibleFieldSensor.airBubbles(271, 300));
        assertEquals(9, VisibleFieldSensor.airBubbles(270, 300));
        assertEquals(1, VisibleFieldSensor.airBubbles(1, 300));
        assertEquals(0, VisibleFieldSensor.airBubbles(0, 300));
        assertEquals(0, VisibleFieldSensor.airBubbles(-20, 300));
    }

    @Test
    void durabilityIsFullWithoutABarAndQuantizedToThirteenSteps() {
        assertEquals(1.0F, VisibleFieldSensor.durabilityFraction(false, 0));
        assertEquals(1.0F, VisibleFieldSensor.durabilityFraction(true, 13));
        assertEquals(0.0F, VisibleFieldSensor.durabilityFraction(true, 0));
        assertEquals(6.0F / 13.0F, VisibleFieldSensor.durabilityFraction(true, 6));
        assertEquals(1.0F, VisibleFieldSensor.durabilityFraction(true, 20));
    }
}
