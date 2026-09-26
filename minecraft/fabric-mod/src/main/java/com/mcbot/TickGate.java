package com.mcbot;

import java.util.Optional;

final class TickGate {
    sealed interface Response permits Status, StepOutcome {}

    sealed interface StepOutcome extends Response permits Step, Rejected {}

    record Status(long gameTime, int serverTick, boolean frozen, boolean paused, boolean clientGated, long clientTicks)
            implements Response {}

    record Step(long stepId, long gameTimeBefore, long gameTimeAfter, long clientTick) implements StepOutcome {}

    record Rejected(String reason) implements StepOutcome {}

    private enum Phase { IDLE, AWAITING_CLIENT_TICK, AWAITING_SERVER_TICK }

    private long nextStepId = 1;
    private long pendingStepId;
    private long pendingGameTimeBefore;
    private long pendingClientTicksBefore;
    private Phase phase = Phase.IDLE;

    Optional<Rejected> requestStep(long gameTime, long clientTicks, boolean frozen, boolean paused, boolean clientGated) {
        if (paused) {
            return Optional.of(new Rejected("singleplayer_paused"));
        }
        if (!frozen) {
            return Optional.of(new Rejected("not_frozen"));
        }
        if (!clientGated) {
            return Optional.of(new Rejected("client_not_gated"));
        }
        if (phase != Phase.IDLE) {
            return Optional.of(new Rejected("step_pending"));
        }
        pendingStepId = nextStepId++;
        pendingGameTimeBefore = gameTime;
        pendingClientTicksBefore = clientTicks;
        phase = Phase.AWAITING_CLIENT_TICK;
        return Optional.empty();
    }

    boolean clientTickDelivered(long clientTicks, long clientTickEndsSent, long clientTickEndsProcessed) {
        if (phase != Phase.AWAITING_CLIENT_TICK
                || clientTicks == pendingClientTicksBefore
                || clientTickEndsProcessed < clientTickEndsSent) {
            return false;
        }
        phase = Phase.AWAITING_SERVER_TICK;
        return true;
    }

    Optional<StepOutcome> observeTickEnd(long gameTime, long clientTicks) {
        if (phase != Phase.AWAITING_SERVER_TICK || gameTime == pendingGameTimeBefore) {
            return Optional.empty();
        }
        phase = Phase.IDLE;
        long advanced = gameTime - pendingGameTimeBefore;
        if (advanced != 1) {
            return Optional.of(new Rejected("advanced_" + advanced + "_ticks"));
        }
        long clientAdvanced = clientTicks - pendingClientTicksBefore;
        if (clientAdvanced != 1) {
            return Optional.of(new Rejected("client_advanced_" + clientAdvanced + "_ticks"));
        }
        return Optional.of(new Step(pendingStepId, pendingGameTimeBefore, gameTime, clientTicks));
    }

    boolean cancelPendingStep() {
        boolean wasPending = phase != Phase.IDLE;
        phase = Phase.IDLE;
        return wasPending;
    }

    static String encode(Response response) {
        return switch (response) {
            case Status status -> "v1 STATUS " + status.gameTime() + " " + status.serverTick()
                    + " " + status.frozen() + " " + status.paused()
                    + " " + status.clientGated() + " " + status.clientTicks();
            case Step step -> "v1 STEP " + step.stepId() + " " + step.gameTimeBefore()
                    + " " + step.gameTimeAfter() + " " + step.clientTick();
            case Rejected rejected -> "v1 ERROR " + rejected.reason();
        };
    }
}
