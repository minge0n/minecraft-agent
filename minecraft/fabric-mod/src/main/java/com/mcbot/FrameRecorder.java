package com.mcbot;

import com.google.gson.JsonObject;
import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.renderpearl.api.GpuFormat;
import com.mojang.renderpearl.api.buffers.GpuBuffer;
import com.mojang.renderpearl.api.buffers.GpuBufferSlice;
import com.mojang.renderpearl.api.commands.CommandEncoder;
import com.mojang.renderpearl.api.commands.RenderPass;
import com.mojang.renderpearl.api.device.GpuDevice;
import com.mojang.renderpearl.api.textures.FilterMode;
import com.mojang.renderpearl.api.textures.GpuTexture;
import com.mojang.renderpearl.api.textures.GpuTextureView;
import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.util.Optional;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BooleanSupplier;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.RenderPipelines;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

// Session recording of the rendered first-person view (docs/decisions/recording.md).
// Frames are for humans only: they never reach the policy, and capture never blocks or
// delays a step. On the render thread every client tick is considered once, in the first
// frame drawn after it; every Nth tick is downscaled on the GPU, read back asynchronously
// and offered to a bounded queue. A sender thread streams queued frames to the recorder
// process over loopback. A frame that cannot be captured, queued or sent is counted and
// dropped, never waited for, and no recording error reaches the game loop.
public final class FrameRecorder {
    private static final Logger LOGGER = LoggerFactory.getLogger(McBotMod.MOD_ID);
    private static final int READBACK_BUFFERS = 4;
    private static final int CONNECT_TIMEOUT_MILLIS = 2_000;
    private static final long START_TIMEOUT_MILLIS = 5_000;
    private static final long STOP_TIMEOUT_MILLIS = 15_000;
    private static final int TEXTURE_USAGE = GpuTexture.USAGE_COPY_SRC | GpuTexture.USAGE_RENDER_ATTACHMENT;
    private static final int READBACK_USAGE = GpuBuffer.USAGE_MAP_READ | GpuBuffer.USAGE_COPY_DST;
    private static final FrameStream.Frame END_OF_STREAM = new FrameStream.Frame(-1, -1, 0, 0, new byte[0]);

    private static volatile Session session;

    private FrameRecorder() {}

    static boolean active() {
        return session != null;
    }

    // Render thread: called after a frame is drawn and before it is submitted.
    public static void onFrameRendered(Minecraft minecraft) {
        Session current = session;
        if (current != null) {
            current.onFrame(minecraft);
        }
    }

    // Returns once the frame of the current client tick, the episode's first recorded
    // tick, has been considered, so the first STEP cannot overtake it.
    static synchronized JsonObject start(RecordingSettings settings) throws IOException, InterruptedException {
        if (session != null) {
            throw new IllegalStateException("recording_active");
        }
        if (!Lockstep.renderFrames) {
            throw new IllegalStateException("recording_requires_rendering");
        }
        if (!Lockstep.clientGated) {
            throw new IllegalStateException("client_not_gated");
        }
        Socket socket = new Socket();
        socket.connect(new InetSocketAddress(InetAddress.getLoopbackAddress(), settings.port()), CONNECT_TIMEOUT_MILLIS);
        socket.setTcpNoDelay(true);
        Session created = new Session(settings, socket, Lockstep.clientTicks());
        created.sender.start();
        session = created;
        if (!await(() -> created.considered >= created.startTick, START_TIMEOUT_MILLIS)) {
            session = null;
            created.sender.interrupt();
            socket.close();
            throw new IllegalStateException("recording_start_timeout");
        }
        LOGGER.info("recording started at client tick {} every {} ticks, {}x{}",
                created.startTick, settings.everyTicks(), settings.width(), settings.height());
        return created.stats();
    }

    // Waits until the frame of the last stepped client tick has been considered, every
    // readback has finished and the sender has written END, then reports the counters.
    static synchronized JsonObject stop() throws InterruptedException {
        Session current = session;
        if (current == null) {
            throw new IllegalStateException("not_recording");
        }
        current.stopTick = Lockstep.clientTicks();
        current.stopRequested = true;
        boolean drained = await(() -> current.finished, STOP_TIMEOUT_MILLIS);
        if (!drained) {
            current.sender.interrupt();
            // GPU objects belong to the render thread; a late readback then fails harmlessly.
            Minecraft.getInstance().execute(current::releaseGpu);
        }
        current.sender.join(STOP_TIMEOUT_MILLIS);
        session = null;
        JsonObject stats = current.stats();
        String failure = current.failure;
        if (!drained && failure == null) {
            failure = "stop_timeout";
        }
        stats.addProperty("status", failure == null ? "complete" : "failed");
        if (failure != null) {
            stats.addProperty("error", failure);
            LOGGER.warn("recording failed: {}", failure);
        }
        return stats;
    }

    static JsonObject status() {
        Session current = session;
        JsonObject json = current == null ? new JsonObject() : current.stats();
        json.addProperty("active", current != null);
        return json;
    }

    private static boolean await(BooleanSupplier condition, long timeoutMillis) throws InterruptedException {
        long deadline = System.currentTimeMillis() + timeoutMillis;
        while (!condition.getAsBoolean()) {
            if (System.currentTimeMillis() >= deadline) {
                return false;
            }
            Thread.sleep(1);
        }
        return true;
    }

    private static final class Session {
        final RecordingSettings settings;
        final long startTick;
        final int frameBytes;
        final Socket socket;
        final BlockingQueue<FrameStream.Frame> queue;
        final Thread sender;

        // Render thread only.
        private long lastConsidered;
        private GpuTexture texture;
        private GpuTextureView textureView;
        private final GpuBuffer[] buffers = new GpuBuffer[READBACK_BUFFERS];
        private final boolean[] busy = new boolean[READBACK_BUFFERS];
        private int inFlight;

        // Shared with the tick-control and sender threads.
        volatile long considered;
        volatile boolean stopRequested;
        volatile long stopTick = -1;
        volatile boolean finished;
        volatile String failure;
        final AtomicLong due = new AtomicLong();
        final AtomicLong captured = new AtomicLong();
        final AtomicLong sent = new AtomicLong();
        final AtomicLong bytesSent = new AtomicLong();
        final AtomicLong droppedGpuBusy = new AtomicLong();
        final AtomicLong droppedQueueFull = new AtomicLong();
        final AtomicLong droppedAfterFailure = new AtomicLong();
        final AtomicLong missedUnrendered = new AtomicLong();
        final AtomicLong readbacks = new AtomicLong();
        final AtomicLong readbackNanos = new AtomicLong();
        final AtomicLong maxReadbackNanos = new AtomicLong();
        final AtomicLong queueHighWater = new AtomicLong();

        Session(RecordingSettings settings, Socket socket, long startTick) {
            this.settings = settings;
            this.socket = socket;
            this.startTick = startTick;
            this.frameBytes = settings.width() * settings.height() * 4;
            this.queue = new ArrayBlockingQueue<>(settings.queueFrames() + 1);
            this.lastConsidered = startTick - 1;
            this.considered = startTick - 1;
            this.sender = new Thread(this::send, "mcbot-recording-sender");
            this.sender.setDaemon(true);
        }

        void onFrame(Minecraft minecraft) {
            try {
                long tick = Lockstep.clientTicks();
                if (tick > lastConsidered) {
                    // A due tick without a frame drawn after it can no longer be captured.
                    for (long skipped = lastConsidered + 1; skipped < tick; skipped++) {
                        if (isDue(skipped)) {
                            due.incrementAndGet();
                            missedUnrendered.incrementAndGet();
                        }
                    }
                    lastConsidered = tick;
                    if (isDue(tick)) {
                        due.incrementAndGet();
                        capture(minecraft, tick);
                    }
                    considered = tick;
                }
                if (stopRequested && !finished && lastConsidered >= stopTick && inFlight == 0
                        && (failure != null || queue.offer(END_OF_STREAM))) {
                    releaseGpu();
                    finished = true;
                }
            } catch (RuntimeException e) {
                fail("capture: " + e);
            }
        }

        private boolean isDue(long tick) {
            return tick >= startTick && (tick - startTick) % settings.everyTicks() == 0;
        }

        private void capture(Minecraft minecraft, long tick) {
            if (failure != null) {
                droppedAfterFailure.incrementAndGet();
                return;
            }
            GpuTextureView source = minecraft.gameRenderer.mainRenderTarget().getColorTextureView();
            if (source == null) {
                missedUnrendered.incrementAndGet();
                return;
            }
            int slot = freeSlot();
            if (slot < 0) {
                droppedGpuBusy.incrementAndGet();
                return;
            }
            ensureGpu();
            CommandEncoder encoder = RenderSystem.getDevice().createCommandEncoder();
            try (RenderPass pass = encoder.createRenderPass(() -> "mcbot recording downscale", textureView, Optional.empty())) {
                RenderSystem.bindDefaultUniforms(pass);
                pass.setPipeline(RenderSystem.getCompiledPipeline(RenderPipelines.TRACY_BLIT));
                pass.setUniform("InSampler", source, RenderSystem.getSamplerCache().getClampToEdge(FilterMode.LINEAR));
                pass.draw(3, 1, 0, 0);
            }
            long gameTime = minecraft.level == null ? -1 : minecraft.level.getGameTime();
            float partialTick = minecraft.getDeltaTracker().getGameTimeDeltaPartialTick(false);
            long requested = System.nanoTime();
            busy[slot] = true;
            inFlight++;
            encoder.copyTextureToBuffer(
                    texture, buffers[slot], 0L, () -> onReadback(slot, tick, gameTime, partialTick, requested), 0);
        }

        // Render thread, a later frame: the GPU copy of one capture has completed.
        private void onReadback(int slot, long tick, long gameTime, float partialTick, long requested) {
            try {
                byte[] rgba = new byte[frameBytes];
                try (GpuBufferSlice.MappedView view = buffers[slot].map(true, false)) {
                    view.data().get(0, rgba);
                }
                long latency = System.nanoTime() - requested;
                readbacks.incrementAndGet();
                readbackNanos.addAndGet(latency);
                maxReadbackNanos.accumulateAndGet(latency, Math::max);
                if (failure != null) {
                    droppedAfterFailure.incrementAndGet();
                } else if (queue.offer(new FrameStream.Frame(tick, gameTime, partialTick, requested, rgba))) {
                    captured.incrementAndGet();
                    queueHighWater.accumulateAndGet(queue.size(), Math::max);
                } else {
                    droppedQueueFull.incrementAndGet();
                }
            } catch (RuntimeException e) {
                fail("readback: " + e);
            } finally {
                busy[slot] = false;
                inFlight--;
            }
        }

        private int freeSlot() {
            for (int slot = 0; slot < READBACK_BUFFERS; slot++) {
                if (!busy[slot]) {
                    return slot;
                }
            }
            return -1;
        }

        private void ensureGpu() {
            if (texture != null) {
                return;
            }
            GpuDevice device = RenderSystem.getDevice();
            texture = device.createTexture(
                    "mcbot recording", TEXTURE_USAGE, GpuFormat.RGBA8_UNORM, settings.width(), settings.height(), 1, 1);
            textureView = device.createTextureView(texture);
            for (int slot = 0; slot < READBACK_BUFFERS; slot++) {
                buffers[slot] = device.createBuffer(() -> "mcbot recording readback", READBACK_USAGE, frameBytes);
            }
        }

        private void releaseGpu() {
            if (texture == null) {
                return;
            }
            for (GpuBuffer buffer : buffers) {
                buffer.close();
            }
            textureView.close();
            texture.close();
            texture = null;
        }

        private void fail(String reason) {
            if (failure == null) {
                failure = reason;
                LOGGER.warn("recording error: {}", reason);
            }
        }

        private void send() {
            try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(socket.getOutputStream(), 1 << 20))) {
                FrameStream.writeStart(
                        out, settings.episode(), startTick, settings.everyTicks(), settings.width(), settings.height());
                out.flush();
                while (true) {
                    FrameStream.Frame frame = queue.take();
                    if (frame == END_OF_STREAM) {
                        FrameStream.writeEnd(out);
                        out.flush();
                        return;
                    }
                    FrameStream.writeFrame(out, frame);
                    if (queue.isEmpty()) {
                        out.flush();
                    }
                    sent.incrementAndGet();
                    bytesSent.addAndGet(frame.rgba().length);
                }
            } catch (IOException e) {
                fail("recorder_connection: " + e);
            } catch (InterruptedException e) {
                fail("sender_interrupted");
                Thread.currentThread().interrupt();
            } finally {
                try {
                    socket.close();
                } catch (IOException ignored) {
                    // Already failed or finished; nothing else to release.
                }
            }
        }

        JsonObject stats() {
            JsonObject json = new JsonObject();
            json.addProperty("episode", settings.episode());
            json.addProperty("start_client_tick", startTick);
            json.addProperty("stop_client_tick", stopTick);
            json.addProperty("every_ticks", settings.everyTicks());
            json.addProperty("width", settings.width());
            json.addProperty("height", settings.height());
            json.addProperty("queue_frames", settings.queueFrames());
            json.addProperty("due", due.get());
            json.addProperty("captured", captured.get());
            json.addProperty("sent", sent.get());
            json.addProperty("bytes_sent", bytesSent.get());
            json.addProperty("dropped_gpu_busy", droppedGpuBusy.get());
            json.addProperty("dropped_queue_full", droppedQueueFull.get());
            json.addProperty("dropped_after_failure", droppedAfterFailure.get());
            json.addProperty("missed_unrendered", missedUnrendered.get());
            json.addProperty("lost_after_capture", captured.get() - sent.get());
            json.addProperty("queue_high_water", queueHighWater.get());
            long count = readbacks.get();
            json.addProperty("readback_ms_mean", count == 0 ? 0.0 : readbackNanos.get() / 1.0e6 / count);
            json.addProperty("readback_ms_max", maxReadbackNanos.get() / 1.0e6);
            return json;
        }
    }
}
