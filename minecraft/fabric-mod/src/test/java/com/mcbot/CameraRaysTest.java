package com.mcbot;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class CameraRaysTest {
    private static final double EPSILON = 1e-9;
    private static final int CENTER_ROW = CameraRays.ROWS / 2;
    private static final int CENTER_COLUMN = CameraRays.COLUMNS / 2;

    private static double dot(double[] a, double[] b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    @Test
    void centerRayIsTheViewDirectionInMinecraftConventions() {
        assertArrayEquals(new double[] {0, 0, 1}, CameraRays.basis(0, 0).direction(CENTER_ROW, CENTER_COLUMN), EPSILON);
        assertArrayEquals(new double[] {-1, 0, 0}, CameraRays.basis(90, 0).direction(CENTER_ROW, CENTER_COLUMN), EPSILON);
        assertArrayEquals(new double[] {0, -1, 0}, CameraRays.basis(0, 90).direction(CENTER_ROW, CENTER_COLUMN), EPSILON);
    }

    @Test
    void screenRightAndUpFollowTheCamera() {
        CameraRays.Basis south = CameraRays.basis(0, 0);
        assertArrayEquals(new double[] {-1, 0, 0}, south.right(), EPSILON);
        assertArrayEquals(new double[] {0, 1, 0}, south.up(), EPSILON);
        assertTrue(south.direction(0, CENTER_COLUMN)[1] > 0);
        assertTrue(south.direction(CENTER_ROW, CameraRays.COLUMNS - 1)[0] < 0);
    }

    @Test
    void raysAreUnitLengthAndSpanTheConfiguredFieldOfView() {
        CameraRays.Basis basis = CameraRays.basis(37, -20);
        for (int row = 0; row < CameraRays.ROWS; row++) {
            for (int column = 0; column < CameraRays.COLUMNS; column++) {
                double[] direction = basis.direction(row, column);
                assertEquals(1.0, dot(direction, direction), EPSILON);
            }
        }

        CameraRays.Basis level = CameraRays.basis(0, 0);
        double[] top = level.direction(0, CENTER_COLUMN);
        double topTangent = Math.tan(Math.acos(dot(top, level.forward())));
        double edgeTangent = Math.tan(Math.toRadians(CameraRays.VERTICAL_FOV_DEGREES / 2.0));
        assertEquals(1.0 - 1.0 / CameraRays.ROWS, topTangent / edgeTangent, EPSILON);
        assertEquals(42.75, CameraRays.horizontalHalfFovDegrees(), 0.01);
    }
}
