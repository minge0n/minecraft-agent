package com.mcbot;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import net.minecraft.world.phys.Vec3;
import org.junit.jupiter.api.Test;

class EnvironmentSettingsTest {
    @Test
    void renderDistanceDefaultsWithoutAnOverride() {
        assertEquals(EnvironmentSettings.RENDER_DISTANCE_CHUNKS, EnvironmentSettings.renderDistance(null));
        assertEquals(EnvironmentSettings.RENDER_DISTANCE_CHUNKS, EnvironmentSettings.renderDistance(" "));
    }

    @Test
    void renderDistanceAcceptsTheBenchmarkValues() {
        assertEquals(4, EnvironmentSettings.renderDistance("4"));
        assertEquals(6, EnvironmentSettings.renderDistance(" 6 "));
        assertEquals(8, EnvironmentSettings.renderDistance("8"));
    }

    @Test
    void renderDistanceRefusesInvalidValues() {
        assertThrows(IllegalArgumentException.class, () -> EnvironmentSettings.renderDistance("1"));
        assertThrows(IllegalArgumentException.class, () -> EnvironmentSettings.renderDistance("33"));
        assertThrows(IllegalArgumentException.class, () -> EnvironmentSettings.renderDistance("far"));
    }

    @Test
    void sensorChunkTestUsesTheNearestPointOfTheChunk() {
        Vec3 eye = new Vec3(8.0, 64.0, 8.0);
        assertTrue(DebugScene.withinBlocks(eye, 0, 0, 32.0));
        assertTrue(DebugScene.withinBlocks(eye, 2, 0, 32.0));
        assertFalse(DebugScene.withinBlocks(eye, 3, 0, 32.0));
        assertTrue(DebugScene.withinBlocks(eye, -2, 0, 32.0));
        assertFalse(DebugScene.withinBlocks(eye, -3, 0, 32.0));
        assertFalse(DebugScene.withinBlocks(eye, 2, 2, 32.0));
    }
}
