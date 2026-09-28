package com.mcbot;

import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.LockSupport;
import java.util.function.Consumer;
import net.minecraft.server.MinecraftServer;

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
    public static volatile Pacing pacing = Pacing.PACED;
    // Unpaced mode only: whether the render thread draws frames between steps. Turning
    // it off measures simulation-only throughput; it never changes step contents.
    public static volatile boolean renderFrames = true;

    private static final AtomicReference<PlayerAction> clientTickPermit = new AtomicReference<>();
    private static final AtomicLong armCount = new AtomicLong();
    private static final AtomicLong clientTickEndsQueued = new AtomicLong();
    private static final AtomicLong clientTickEndsProcessed = new AtomicLong();
    private static final AtomicLong barrierBlockedFrames = new AtomicLong();
    private static volatile PlayerAction currentClientAction;
    private static volatile long clientTickEndsSent;
    private static volatile long clientTicks;
    private static volatile long clientTickStartedNanos;
    private static volatile long clientTickEndedNanos;
    private static volatile long clientTickEndProcessedNanos;
    private static volatile boolean serverStepGranted;
    private static volatile boolean clientTickAwaitingGrant;
    // Ordering barrier: the ID of the marker the server sent after its last completed
    // step, and the newest marker the client has queued. A client tick only starts once
    // every packet the server produced during the previous step is queued ahead of it.
    private static volatile long requiredServerMarker;
    private static volatile long queuedServerMarker;
    private static volatile boolean serverMarkerPending;
    private static volatile Thread serverThread;
    private static volatile Consumer<MinecraftServer> beforeTickRateUpdate = server -> {};
    private static volatile Thread renderThread;
    private static volatile long lastFrameNanos = System.nanoTime();
    private static volatile ClientPlayerSnapshot clientPlayer = ClientPlayerSnapshot.ABSENT;
    private static volatile RegionDigest traceRegion;

    private Lockstep() {}

    static void arm() {
        renderThread = Thread.currentThread();
        playersGated = true;
        clientGated = true;
        armCount.incrementAndGet();
    }

    static void disarm() {
        clientGated = false;
        playersGated = false;
        serverStepGranted = false;
        clientTickAwaitingGrant = false;
        clientTickPermit.set(null);
        currentClientAction = null;
    }

    static long armCount() {
        return armCount.get();
    }

    static void resetConnectionCounters() {
        clientTickEndsSent = 0;
        clientTickEndsQueued.set(0);
        clientTickEndsProcessed.set(0);
        requiredServerMarker = 0;
        serverMarkerPending = false;
    }

    static void setServerThread(Thread thread) {
        serverThread = thread;
    }

    static void onBeforeTickRateUpdate(Consumer<MinecraftServer> hook) {
        beforeTickRateUpdate = hook;
    }

    // Called from the server loop just before the tick-rate manager decides whether the
    // current iteration runs game elements.
    public static void beforeTickRateUpdate(MinecraftServer server) {
        beforeTickRateUpdate.accept(server);
    }

    static void grantClientTick(PlayerAction action) {
        clientTickPermit.set(action);
        wake(renderThread);
    }

    static void revokeClientTick() {
        clientTickPermit.set(null);
    }

    public static boolean takeClientTickPermit() {
        if (!serverOutputDelivered()) {
            if (clientTickPermit.get() != null) {
                barrierBlockedFrames.incrementAndGet();
            }
            return false;
        }
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
        boolean granted = currentClientAction != null;
        if (granted) {
            clientTickEndedNanos = System.nanoTime();
        }
        long tick = clientTicks + 1;
        clientPlayer = player.at(tick);
        currentClientAction = null;
        clientTicks = tick;
        if (granted) {
            // The tick-end packet can reach the server before this point; the server
            // re-checks the grant once the client tick is recorded.
            clientTickAwaitingGrant = true;
            if (pacing == Pacing.UNPACED) {
                wake(serverThread);
            }
        }
    }

    // Called on the network thread once a client tick-end packet is queued for the server.
    public static void onClientTickEndQueued() {
        clientTickEndsQueued.incrementAndGet();
        if (pacing == Pacing.UNPACED) {
            wake(serverThread);
        }
    }

    // Called on the server thread when a client tick-end packet is handled.
    public static void onClientTickEndProcessed() {
        clientTickEndProcessedNanos = System.nanoTime();
        clientTickEndsProcessed.incrementAndGet();
    }

    // Called on the network thread once a server marker packet is queued for the client.
    public static void onServerMarkerQueued(long marker) {
        if (marker > queuedServerMarker) {
            queuedServerMarker = marker;
        }
        wake(renderThread);
    }

    static void requireServerMarker(long marker) {
        requiredServerMarker = marker;
        serverMarkerPending = true;
        if (pacing == Pacing.UNPACED) {
            wake(serverThread);
        }
    }

    static boolean serverMarkerPending() {
        return serverMarkerPending;
    }

    static void serverMarkerSent() {
        serverMarkerPending = false;
    }

    static void setServerStepGranted(boolean granted) {
        serverStepGranted = granted;
        if (granted) {
            clientTickAwaitingGrant = false;
        }
    }

    static boolean serverStepGranted() {
        return serverStepGranted;
    }

    // Unpaced mode only: the server loop has lockstep work to do right now and must not
    // wait for its wall-clock tick deadline.
    public static boolean serverWorkPending() {
        return pacing == Pacing.UNPACED
                && (serverStepGranted
                        || clientTickAwaitingGrant
                        || serverMarkerPending
                        || clientTickEndsQueued.get() > clientTickEndsProcessed.get());
    }

    // Unpaced replacement for the vanilla frame limiter: wait for the frame deadline, but
    // return as soon as a granted client tick is ready to run.
    public static void waitForFrameOrClientTick(int framerateLimit) {
        long deadline = lastFrameNanos + 1_000_000_000L / framerateLimit;
        while (!clientTickReady()) {
            long remaining = deadline - System.nanoTime();
            if (remaining <= 0) {
                break;
            }
            LockSupport.parkNanos(remaining);
        }
        lastFrameNanos = System.nanoTime();
    }

    public static boolean unpacedAndGated() {
        return pacing == Pacing.UNPACED && clientGated;
    }

    public static boolean skipFrame() {
        return unpacedAndGated() && !renderFrames;
    }

    private static boolean clientTickReady() {
        return clientTickPermit.get() != null && serverOutputDelivered();
    }

    // True once every packet the server produced before the latest required marker is
    // queued on the client. A marker that is required but not yet sent counts as missing.
    private static boolean serverOutputDelivered() {
        return !serverMarkerPending && queuedServerMarker >= requiredServerMarker;
    }

    private static void wake(Thread thread) {
        if (thread != null) {
            LockSupport.unpark(thread);
        }
    }

    static long clientTicks() {
        return clientTicks;
    }

    static long clientTickEndsSent() {
        return clientTickEndsSent;
    }

    static long clientTickEndsProcessed() {
        return clientTickEndsProcessed.get();
    }

    static long clientTickStartedNanos() {
        return clientTickStartedNanos;
    }

    static long clientTickEndedNanos() {
        return clientTickEndedNanos;
    }

    static long clientTickEndProcessedNanos() {
        return clientTickEndProcessedNanos;
    }

    static long barrierBlockedFrames() {
        return barrierBlockedFrames.get();
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
