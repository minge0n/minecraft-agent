package com.mcbot;

import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

public final class Lockstep {
    public enum ClientAction { NOOP, FORWARD }

    public record ClientPlayerSnapshot(long clientTick, double x, double z, int tickCount) {}

    public static volatile boolean playersGated;
    public static volatile boolean clientGated;
    public static final AtomicLong clientTickEndsProcessed = new AtomicLong();

    private static final AtomicReference<ClientAction> clientTickPermit = new AtomicReference<>();
    private static volatile ClientAction currentClientAction = ClientAction.NOOP;
    private static volatile long clientTickEndsSent;
    private static volatile long clientTicks;
    private static volatile ClientPlayerSnapshot clientPlayer =
            new ClientPlayerSnapshot(0, Double.NaN, Double.NaN, -1);

    private Lockstep() {}

    static void arm() {
        playersGated = true;
        clientGated = true;
    }

    static void grantClientTick(ClientAction action) {
        clientTickPermit.set(action);
    }

    static void revokeClientTick() {
        clientTickPermit.set(null);
    }

    public static boolean takeClientTickPermit() {
        ClientAction action = clientTickPermit.getAndSet(null);
        if (action == null) {
            return false;
        }
        currentClientAction = action;
        return true;
    }

    static ClientAction currentClientAction() {
        return currentClientAction;
    }

    static void recordClientTick(boolean sentTickEnd, double x, double z, int playerTickCount) {
        if (sentTickEnd) {
            clientTickEndsSent++;
        }
        long tick = clientTicks + 1;
        clientPlayer = new ClientPlayerSnapshot(tick, x, z, playerTickCount);
        currentClientAction = ClientAction.NOOP;
        clientTicks = tick;
    }

    static long clientTicks() {
        return clientTicks;
    }

    static long clientTickEndsSent() {
        return clientTickEndsSent;
    }

    static ClientPlayerSnapshot clientPlayer() {
        return clientPlayer;
    }
}
