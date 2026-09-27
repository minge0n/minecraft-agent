package com.mcbot;

import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

public final class Lockstep {
    public record ClientPlayerSnapshot(
            long clientTick, double x, double y, double z, float yaw, float pitch, int tickCount,
            double vx, double vy, double vz, boolean onGround, ClientView view) {
        static final ClientPlayerSnapshot ABSENT = new ClientPlayerSnapshot(
                0, Double.NaN, Double.NaN, Double.NaN, Float.NaN, Float.NaN, -1, Double.NaN, Double.NaN, Double.NaN, false,
                ClientView.NONE);

        ClientPlayerSnapshot at(long tick) {
            return new ClientPlayerSnapshot(tick, x, y, z, yaw, pitch, tickCount, vx, vy, vz, onGround, view);
        }
    }

    // The client's own copy of the traced region at the end of a client tick; privileged diagnostics only.
    public record ClientView(long blockCrc32, int entityCount) {
        static final ClientView NONE = new ClientView(0, 0);
    }

    public static volatile boolean playersGated;
    public static volatile boolean clientGated;
    public static final AtomicLong clientTickEndsProcessed = new AtomicLong();

    private static final AtomicReference<PlayerAction> clientTickPermit = new AtomicReference<>();
    private static final AtomicLong armCount = new AtomicLong();
    private static volatile PlayerAction currentClientAction;
    private static volatile long clientTickEndsSent;
    private static volatile long clientTicks;
    private static volatile long clientTickStartedNanos;
    private static volatile long clientTickEndedNanos;
    private static volatile ClientPlayerSnapshot clientPlayer = ClientPlayerSnapshot.ABSENT;
    private static volatile RegionDigest traceRegion;

    private Lockstep() {}

    static void arm() {
        playersGated = true;
        clientGated = true;
        armCount.incrementAndGet();
    }

    static void disarm() {
        clientGated = false;
        playersGated = false;
        clientTickPermit.set(null);
        currentClientAction = null;
    }

    static long armCount() {
        return armCount.get();
    }

    static void resetClientTickEnds() {
        clientTickEndsSent = 0;
        clientTickEndsProcessed.set(0);
    }

    static void grantClientTick(PlayerAction action) {
        clientTickPermit.set(action);
    }

    static void revokeClientTick() {
        clientTickPermit.set(null);
    }

    public static boolean takeClientTickPermit() {
        PlayerAction action = clientTickPermit.getAndSet(null);
        if (action == null) {
            return false;
        }
        clientTickStartedNanos = System.nanoTime();
        currentClientAction = action;
        return true;
    }

    static PlayerAction currentClientAction() {
        return currentClientAction;
    }

    static void recordClientTick(boolean sentTickEnd, ClientPlayerSnapshot player) {
        if (sentTickEnd) {
            clientTickEndsSent++;
        }
        if (currentClientAction != null) {
            clientTickEndedNanos = System.nanoTime();
        }
        long tick = clientTicks + 1;
        clientPlayer = player.at(tick);
        currentClientAction = null;
        clientTicks = tick;
    }

    static long clientTicks() {
        return clientTicks;
    }

    static long clientTickEndsSent() {
        return clientTickEndsSent;
    }

    static long clientTickStartedNanos() {
        return clientTickStartedNanos;
    }

    static long clientTickEndedNanos() {
        return clientTickEndedNanos;
    }

    static ClientPlayerSnapshot clientPlayer() {
        return clientPlayer;
    }

    static RegionDigest traceRegion() {
        return traceRegion;
    }

    static void setTraceRegion(RegionDigest region) {
        traceRegion = region;
    }
}
