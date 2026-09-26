package com.mcbot;

final class CameraRays {
    static final int ROWS = 25;
    static final int COLUMNS = 33;
    static final double VERTICAL_FOV_DEGREES = 70.0;
    static final double MAX_DISTANCE = 32.0;

    private static final double HALF_HEIGHT = Math.tan(Math.toRadians(VERTICAL_FOV_DEGREES / 2.0));
    private static final double HALF_WIDTH = HALF_HEIGHT * COLUMNS / ROWS;

    record Basis(double[] forward, double[] right, double[] up) {
        double[] direction(int row, int column) {
            double x = (2.0 * (column + 0.5) / COLUMNS - 1.0) * HALF_WIDTH;
            double y = (1.0 - 2.0 * (row + 0.5) / ROWS) * HALF_HEIGHT;
            double[] direction = new double[3];
            for (int axis = 0; axis < 3; axis++) {
                direction[axis] = forward[axis] + x * right[axis] + y * up[axis];
            }
            double length = Math.sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]);
            for (int axis = 0; axis < 3; axis++) {
                direction[axis] /= length;
            }
            return direction;
        }
    }

    private CameraRays() {}

    static Basis basis(float yawDegrees, float pitchDegrees) {
        double[] forward = viewVector(pitchDegrees, yawDegrees);
        double[] up = viewVector(pitchDegrees - 90.0, yawDegrees);
        double[] right = {
            forward[1] * up[2] - forward[2] * up[1],
            forward[2] * up[0] - forward[0] * up[2],
            forward[0] * up[1] - forward[1] * up[0],
        };
        return new Basis(forward, right, up);
    }

    static double horizontalHalfFovDegrees() {
        return Math.toDegrees(Math.atan(HALF_WIDTH));
    }

    private static double[] viewVector(double pitchDegrees, double yawDegrees) {
        double pitch = Math.toRadians(pitchDegrees);
        double yaw = Math.toRadians(-yawDegrees);
        return new double[] {Math.sin(yaw) * Math.cos(pitch), -Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch)};
    }
}
