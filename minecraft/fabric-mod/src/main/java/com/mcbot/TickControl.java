package com.mcbot;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.function.Consumer;
import java.util.function.Function;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import net.minecraft.client.Minecraft;
import net.minecraft.client.server.IntegratedServer;
import net.minecraft.core.BlockPos;
import net.minecraft.network.protocol.common.ClientboundPingPacket;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.stats.Stats;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.EntitySpawnReason;
import net.minecraft.world.entity.EntityTypes;
import net.minecraft.world.entity.Mob;
import net.minecraft.world.level.levelgen.Heightmap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

final class TickControl {
    private static final Logger LOGGER = LoggerFactory.getLogger(McBotMod.MOD_ID);
    private static final long REQUEST_TIMEOUT_SECONDS = 5;
    private static final long RESET_TIMEOUT_MILLIS = 180_000;

    private record PendingStep(CompletableFuture<String> reply, Function<TickGate.StepOutcome, String> finish) {}

    private final TickGate gate = new TickGate();
    private volatile IntegratedServer server;
    private volatile long requestReceivedNanos;
    private PendingStep pendingStep;
    private long serverStepGrantedNanos;
    private int serverMarker;
    private long markerNotBeforeServerTick;
    private int armorStandId = -1;
    private int huskId = -1;

    void start(int port) {
        Lockstep.onBeforeTickRateUpdate(this::grantIfReady);
        Lockstep.pacing = Pacing.fromEnvironment();
        Lockstep.renderFrames = !"off".equals(System.getenv("MCBOT_RENDER"));
        LOGGER.info("lockstep pacing {}, render frames {}", Lockstep.pacing, Lockstep.renderFrames);
        ServerLifecycleEvents.SERVER_STARTED.register(started -> {
            if (started instanceof IntegratedServer integrated) {
                integrated.tickRateManager().setFrozen(true);
                Lockstep.setServerThread(integrated.getRunningThread());
                // Each world has a new server whose tick count restarts at zero.
                markerNotBeforeServerTick = 0;
                server = integrated;
                LOGGER.info("experimental integrated-server world tick freeze enabled");
            }
        });
        ServerLifecycleEvents.SERVER_STOPPING.register(stopping -> {
            server = null;
            Lockstep.setServerStepGranted(false);
            if (pendingStep != null) {
                gate.cancelPendingStep();
                pendingStep.reply().complete(pendingStep.finish().apply(new TickGate.Rejected("server_stopping")));
                pendingStep = null;
            }
        });
        ServerTickEvents.END_SERVER_TICK.register(ticked -> {
            if (ticked != server) {
                return;
            }
            if (pendingStep != null && Lockstep.serverStepGranted()) {
                gate.observeTickEnd(ticked.overworld().getGameTime(), Lockstep.clientTicks()).ifPresent(outcome -> {
                    Lockstep.setServerStepGranted(false);
                    requestServerMarker(ticked.getTickCount());
                    PendingStep finished = pendingStep;
                    pendingStep = null;
                    finished.reply().complete(finished.finish().apply(outcome));
                });
            }
            sendRequestedServerMarker(ticked);
        });

        Thread listener = new Thread(() -> serve(port), "mcbot-tick-control");
        listener.setDaemon(true);
        listener.start();
    }

    // Called from the server loop just before the tick-rate manager decides whether this
    // iteration runs game elements. A step whose client tick and tick-end packet have
    // both arrived is granted here, so the stepped tick is this very iteration.
    private void grantIfReady(MinecraftServer ticked) {
        IntegratedServer current = server;
        if (ticked != current || pendingStep == null) {
            return;
        }
        if (gate.clientTickDelivered(
                Lockstep.clientTicks(), Lockstep.clientTickEndsSent(), Lockstep.clientTickEndsProcessed())) {
            serverStepGrantedNanos = System.nanoTime();
            Lockstep.setServerStepGranted(true);
            current.tickRateManager().stepGameIfPaused(1);
        }
    }

    // Ordering barrier. The client may start its next granted tick only after this
    // marker is queued on its side. A marker is sent at the end of a server iteration,
    // after that iteration broadcast its block, entity and chunk changes. A step's own
    // changes are broadcast in the stepped iteration, so its marker goes out at once. A
    // privileged edit runs as a server task, possibly inside an iteration after that
    // iteration's broadcast, so its marker waits for the end of the next iteration.
    private void requestServerMarker(long notBeforeServerTick) {
        serverMarker++;
        markerNotBeforeServerTick = Math.max(markerNotBeforeServerTick, notBeforeServerTick);
        Lockstep.requireServerMarker(serverMarker);
    }

    private void sendRequestedServerMarker(MinecraftServer ticked) {
        if (!Lockstep.serverMarkerPending() || ticked.getTickCount() < markerNotBeforeServerTick) {
            return;
        }
        for (ServerPlayer player : ticked.getPlayerList().getPlayers()) {
            player.connection.send(new ClientboundPingPacket(serverMarker));
        }
        Lockstep.serverMarkerSent();
    }

    private void serve(int port) {
        try (ServerSocket listener = new ServerSocket(port, 1, InetAddress.getLoopbackAddress())) {
            LOGGER.info("experimental tick control listening on loopback port {}", listener.getLocalPort());
            while (true) {
                try (Socket socket = listener.accept();
                     BufferedReader input = new BufferedReader(new InputStreamReader(socket.getInputStream(), StandardCharsets.UTF_8));
                     BufferedWriter output = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream(), StandardCharsets.UTF_8))) {
                    socket.setTcpNoDelay(true);
                    for (String request = input.readLine(); request != null; request = input.readLine()) {
                        requestReceivedNanos = System.nanoTime();
                        output.write(request.startsWith("v2 ") ? respondV2(request.substring(3)) : respond(request));
                        output.write('\n');
                        output.flush();
                    }
                } catch (IOException e) {
                    LOGGER.warn("tick control connection closed: {}", e.toString());
                }
            }
        } catch (IOException e) {
            LOGGER.error("tick control cannot listen on loopback port {}", port, e);
        }
    }

    private String respond(String request) {
        if (request.equals("v1 QUIT")) {
            quit();
            return "v1 QUIT";
        }
        return onServerThread("v1", reply -> handle(request, reply));
    }

    private String respondV2(String request) {
        int split = request.indexOf(' ');
        String command = split < 0 ? request : request.substring(0, split);
        JsonObject payload;
        try {
            payload = split < 0 ? new JsonObject() : JsonParser.parseString(request.substring(split + 1)).getAsJsonObject();
        } catch (RuntimeException e) {
            return "v2 ERROR malformed_json";
        }
        return switch (command) {
            case "QUIT" -> {
                quit();
                yield "v2 QUIT {}";
            }
            case "RESET" -> reset(payload);
            default -> onServerThread("v2", reply -> handleV2(command, payload, reply));
        };
    }

    private String onServerThread(String version, Consumer<CompletableFuture<String>> handler) {
        IntegratedServer current = server;
        if (current == null) {
            return version + " ERROR no_integrated_server";
        }
        CompletableFuture<String> reply = new CompletableFuture<>();
        current.execute(() -> {
            if (reply.isDone()) {
                return;
            }
            try {
                handler.accept(reply);
            } catch (RuntimeException e) {
                LOGGER.warn("tick control request failed", e);
                reply.complete(version + " ERROR " + errorReason(e));
            }
        });
        try {
            return reply.get(REQUEST_TIMEOUT_SECONDS, TimeUnit.SECONDS);
        } catch (TimeoutException e) {
            current.execute(() -> cancelStep(current, reply));
            return version + " ERROR timeout";
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return version + " ERROR interrupted";
        } catch (Exception e) {
            return version + " ERROR " + e.getClass().getSimpleName();
        }
    }

    private static String errorReason(RuntimeException e) {
        if (e instanceof IllegalArgumentException || e instanceof IllegalStateException) {
            return e.getMessage();
        }
        return e.getClass().getSimpleName();
    }

    private void quit() {
        Minecraft minecraft = Minecraft.getInstance();
        minecraft.execute(minecraft::stop);
    }

    private void handle(String request, CompletableFuture<String> reply) {
        IntegratedServer current = server;
        long gameTime = current.overworld().getGameTime();
        switch (request) {
            case "v1 STATUS" -> reply.complete(TickGate.encode(status(current)));
            case "v1 STEP NOOP" -> requestStep(current, PlayerAction.NOOP, reply, TickGate::encode);
            case "v1 STEP FORWARD" -> requestStep(current, PlayerAction.FORWARD, reply, TickGate::encode);
            case "v1 DEBUG_SPAWN" -> reply.complete(spawnProbeEntities(current));
            case "v1 DEBUG_PROBE" -> reply.complete(probeEntities(current));
            case "v1 DEBUG_PLAYER" -> reply.complete(probePlayer(current, gameTime));
            default -> reply.complete("v1 ERROR unknown_request");
        }
    }

    private void handleV2(String command, JsonObject payload, CompletableFuture<String> reply) {
        IntegratedServer current = server;
        switch (command) {
            case "SCHEMA" -> reply.complete(v2("SCHEMA", VisibleFieldSensor.schema()));
            case "STATUS" -> reply.complete(v2("STATUS", statusJson(status(current))));
            case "PACING" -> {
                Lockstep.pacing = Pacing.valueOf(payload.get("mode").getAsString().toUpperCase(Locale.ROOT));
                if (payload.has("render_frames")) {
                    Lockstep.renderFrames = payload.get("render_frames").getAsBoolean();
                }
                reply.complete(v2("PACING", pacingJson()));
            }
            case "OBSERVE" -> reply.complete(v2("OBSERVE", observationReply(firstPlayer(current))));
            case "STEP" -> {
                long receivedNanos = requestReceivedNanos;
                long acceptedNanos = System.nanoTime();
                PlayerAction action = PlayerAction.fromJson(payload);
                requestStep(current, action, reply, outcome -> finishV2Step(current, outcome, receivedNanos, acceptedNanos));
            }
            default -> {
                if (!command.startsWith("DEBUG_")) {
                    throw new IllegalArgumentException("unknown_request");
                }
                JsonObject result = DebugScene.handle(
                        command, current.overworld().getGameTime(), firstPlayer(current), payload);
                if (DebugScene.mutatesWorld(command)) {
                    // The edit reaches the client before its next granted tick.
                    requestServerMarker(current.getTickCount() + 1L);
                }
                reply.complete(v2(command, result));
            }
        }
    }

    // Timing phases in milliseconds, in order; each ends where the next begins.
    // dispatch: request read on the listener thread -> handled on the server thread.
    // client_wait: -> the client tick starts (next frame, ordering barrier).
    // client_tick: the granted client tick itself.
    // tick_end_sync: client tick end -> its tick-end packet is handled on the server.
    // server_wait: -> the step is granted in the server loop (wall-clock pacing).
    // server_tick: the granted logical server tick.
    // observation / encode: sensor construction and reply serialization.
    private String finishV2Step(
            IntegratedServer current, TickGate.StepOutcome outcome, long receivedNanos, long acceptedNanos) {
        if (outcome instanceof TickGate.Rejected rejected) {
            return "v2 ERROR " + rejected.reason();
        }
        TickGate.Step step = (TickGate.Step) outcome;
        long serverTickEndedNanos = System.nanoTime();
        ServerPlayer player = firstPlayer(current);
        JsonObject observation = VisibleFieldSensor.observe(player);
        long observedNanos = System.nanoTime();
        String encodedObservation = observation.toString();
        long encodedNanos = System.nanoTime();

        long clientTickStarted = Lockstep.clientTickStartedNanos();
        long clientTickEnded = Lockstep.clientTickEndedNanos();
        // The tick-end packet may be handled before the render thread records the end of
        // the client tick; sync then counts as zero and waiting starts at the tick end.
        long tickEndProcessed = Math.max(Lockstep.clientTickEndProcessedNanos(), clientTickEnded);
        JsonObject timing = new JsonObject();
        timing.addProperty("dispatch_ms", millis(receivedNanos, acceptedNanos));
        timing.addProperty("client_wait_ms", millis(acceptedNanos, clientTickStarted));
        timing.addProperty("client_tick_ms", millis(clientTickStarted, clientTickEnded));
        timing.addProperty("tick_end_sync_ms", millis(clientTickEnded, tickEndProcessed));
        timing.addProperty("server_wait_ms", millis(tickEndProcessed, serverStepGrantedNanos));
        timing.addProperty("server_tick_ms", millis(serverStepGrantedNanos, serverTickEndedNanos));
        timing.addProperty("observation_ms", millis(serverTickEndedNanos, observedNanos));
        timing.addProperty("encode_ms", millis(observedNanos, encodedNanos));
        timing.addProperty("server_total_ms", millis(receivedNanos, encodedNanos));

        JsonObject info = new JsonObject();
        info.addProperty("game_time", player.level().getGameTime());
        info.addProperty("step_id", step.stepId());
        info.addProperty("tick_before", step.gameTimeBefore());
        info.addProperty("tick_after", step.gameTimeAfter());
        info.addProperty("client_tick", step.clientTick());
        info.addProperty("pacing", Lockstep.pacing.name().toLowerCase(Locale.ROOT));
        info.add("timing", timing);
        return "v2 STEP {\"observation\":" + encodedObservation
                + ",\"terminated\":" + player.isDeadOrDying()
                + ",\"info\":" + info + "}";
    }

    private static JsonObject observationReply(ServerPlayer player) {
        JsonObject reply = new JsonObject();
        reply.add("observation", VisibleFieldSensor.observe(player));
        reply.addProperty("terminated", player.isDeadOrDying());
        JsonObject info = new JsonObject();
        info.addProperty("game_time", player.level().getGameTime());
        reply.add("info", info);
        return reply;
    }

    private String reset(JsonObject payload) {
        long seed;
        WorldRequest.Preset preset;
        try {
            seed = payload.get("seed").getAsLong();
            preset = WorldRequest.Preset.valueOf(payload.get("preset").getAsString().toUpperCase(Locale.ROOT));
        } catch (RuntimeException e) {
            return "v2 ERROR invalid_reset";
        }

        long started = System.nanoTime();
        IntegratedServer previous = server;
        long armedBefore = Lockstep.armCount();
        McBotClient.requestFreshWorld(new WorldRequest(seed, preset));
        long deadline = System.currentTimeMillis() + RESET_TIMEOUT_MILLIS;
        while (System.currentTimeMillis() < deadline) {
            IntegratedServer current = server;
            if (current != null && current != previous && Lockstep.armCount() > armedBefore) {
                return onServerThread("v2", reply -> {
                    JsonObject result = observationReply(firstPlayer(current));
                    JsonObject info = result.getAsJsonObject("info");
                    info.addProperty("seed", seed);
                    info.addProperty("preset", preset.name().toLowerCase(Locale.ROOT));
                    info.addProperty("level_id", McBotClient.currentLevelId());
                    info.addProperty("reset_ms", millis(started, System.nanoTime()));
                    reply.complete(v2("RESET", result));
                });
            }
            try {
                Thread.sleep(50);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return "v2 ERROR interrupted";
            }
        }
        return "v2 ERROR reset_timeout";
    }

    private static String v2(String command, JsonObject payload) {
        return "v2 " + command + " " + payload;
    }

    private static double millis(long fromNanos, long toNanos) {
        return (toNanos - fromNanos) / 1.0e6;
    }

    private static TickGate.Status status(IntegratedServer current) {
        return new TickGate.Status(
                current.overworld().getGameTime(), current.getTickCount(), current.tickRateManager().isFrozen(),
                current.isPaused(), Lockstep.clientGated, Lockstep.clientTicks());
    }

    private static JsonObject statusJson(TickGate.Status status) {
        JsonObject json = new JsonObject();
        json.addProperty("game_time", status.gameTime());
        json.addProperty("server_tick", status.serverTick());
        json.addProperty("frozen", status.frozen());
        json.addProperty("paused", status.paused());
        json.addProperty("client_gated", status.clientGated());
        json.addProperty("client_ticks", status.clientTicks());
        json.add("pacing", pacingJson());
        json.add("client_settings", EnvironmentSettings.describe(Minecraft.getInstance()));
        return json;
    }

    private static JsonObject pacingJson() {
        JsonObject pacing = new JsonObject();
        pacing.addProperty("mode", Lockstep.pacing.name().toLowerCase(Locale.ROOT));
        pacing.addProperty("render_frames", Lockstep.renderFrames);
        pacing.addProperty("barrier_blocked_frames", Lockstep.barrierBlockedFrames());
        return pacing;
    }

    private void requestStep(
            IntegratedServer current,
            PlayerAction action,
            CompletableFuture<String> reply,
            Function<TickGate.StepOutcome, String> finish) {
        var rejected = gate.requestStep(
                current.overworld().getGameTime(), Lockstep.clientTicks(), current.tickRateManager().isFrozen(),
                current.isPaused(), Lockstep.clientGated);
        if (rejected.isPresent()) {
            reply.complete(finish.apply(rejected.get()));
            return;
        }
        pendingStep = new PendingStep(reply, finish);
        Lockstep.grantClientTick(action);
    }

    private void cancelStep(IntegratedServer current, CompletableFuture<String> reply) {
        if (pendingStep == null || pendingStep.reply() != reply) {
            return;
        }
        pendingStep = null;
        gate.cancelPendingStep();
        Lockstep.revokeClientTick();
        Lockstep.setServerStepGranted(false);
        current.tickRateManager().stopStepping();
    }

    private static ServerPlayer firstPlayer(IntegratedServer current) {
        List<ServerPlayer> players = current.getPlayerList().getPlayers();
        if (players.isEmpty()) {
            throw new IllegalStateException("no_player");
        }
        return players.getFirst();
    }

    private String spawnProbeEntities(IntegratedServer current) {
        ServerPlayer player = firstPlayer(current);
        ServerLevel level = player.level();
        BlockPos ground = level.getHeightmapPos(Heightmap.Types.MOTION_BLOCKING, player.blockPosition());
        Entity armorStand = EntityTypes.ARMOR_STAND.create(level, EntitySpawnReason.COMMAND);
        Mob husk = EntityTypes.HUSK.create(level, EntitySpawnReason.COMMAND);
        if (armorStand == null || husk == null) {
            return "v1 ERROR spawn_failed";
        }

        armorStand.setPos(ground.getX() - 2.5, ground.getY() + 10.0, ground.getZ() + 0.5);
        husk.setPos(ground.getX() + 6.5, ground.getY(), ground.getZ() + 0.5);
        husk.setPersistenceRequired();
        husk.setTarget(player);
        level.addFreshEntity(armorStand);
        level.addFreshEntity(husk);
        armorStandId = armorStand.getId();
        huskId = husk.getId();
        return "v1 DEBUG_SPAWN " + armorStandId + " " + huskId;
    }

    private String probeEntities(IntegratedServer current) {
        ServerPlayer player = firstPlayer(current);
        Entity armorStand = player.level().getEntity(armorStandId);
        Entity husk = player.level().getEntity(huskId);
        if (armorStand == null || husk == null) {
            return "v1 ERROR probe_entities_missing";
        }
        return "v1 DEBUG_PROBE " + current.overworld().getGameTime() + " " + armorStand.getY()
                + " " + husk.getX() + " " + husk.getZ() + " " + husk.distanceTo(player);
    }

    private String probePlayer(IntegratedServer current, long gameTime) {
        ServerPlayer player = firstPlayer(current);
        int playTime = player.getStats().getValue(Stats.CUSTOM.get(Stats.PLAY_TIME));
        Lockstep.ClientPlayerSnapshot client = Lockstep.clientPlayer();
        return "v1 DEBUG_PLAYER " + gameTime
                + " " + player.getX() + " " + player.getZ() + " " + player.tickCount + " " + playTime
                + " " + client.clientTick() + " " + client.x() + " " + client.z() + " " + client.tickCount();
    }
}
