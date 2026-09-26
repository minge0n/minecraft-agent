package com.mcbot;

import java.util.Optional;

final class TickGate {
    sealed interface Response permits Status, Step, Rejected {}

    record Status(long gameTime, int serverTick, boolean frozen, boolean paused) implements Response {}

    record Step(long stepId, long gameTimeBefore, long gameTimeAfter) implements Response {}

    record Rejected(String reason) implements Response {}

    private long nextStepId = 1;
    private long pendingStepId;
    private long pendingGameTimeBefore;
    private boolean stepPending;

    Optional<Rejected> requestStep(long gameTime, boolean frozen, boolean paused) {
        if (paused) {
            return Optional.of(new Rejected("singleplayer_paused"));
        }
        if (!frozen) {
            return Optional.of(new Rejected("not_frozen"));
        }
        if (stepPending) {
            return Optional.of(new Rejected("step_pending"));
        }
        pendingStepId = nextStepId++;
        pendingGameTimeBefore = gameTime;
        stepPending = true;
        return Optional.empty();
    }

    Optional<Response> observeTickEnd(long gameTime) {
        if (!stepPending || gameTime == pendingGameTimeBefore) {
            return Optional.empty();
        }
        stepPending = false;
        long advanced = gameTime - pendingGameTimeBefore;
        if (advanced != 1) {
            return Optional.of(new Rejected("advanced_" + advanced + "_ticks"));
        }
        return Optional.of(new Step(pendingStepId, pendingGameTimeBefore, gameTime));
    }

    boolean cancelPendingStep() {
        boolean wasPending = stepPending;
        stepPending = false;
        return wasPending;
    }

    static String encode(Response response) {
        return switch (response) {
            case Status status -> "v1 STATUS " + status.gameTime() + " " + status.serverTick()
                    + " " + status.frozen() + " " + status.paused();
            case Step step -> "v1 STEP " + step.stepId() + " " + step.gameTimeBefore() + " " + step.gameTimeAfter();
            case Rejected rejected -> "v1 ERROR " + rejected.reason();
        };
    }
}
