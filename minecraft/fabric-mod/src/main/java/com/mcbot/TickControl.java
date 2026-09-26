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
    private PendingStep pendingStep;
    private long serverStepGrantedNanos;
    private int armorStandId = -1;
    private int huskId = -1;

    void start(int port) {
        ServerLifecycleEvents.SERVER_STARTED.register(started -> {
            if (started instanceof IntegratedServer integrated) {
                integrated.tickRateManager().setFrozen(true);
                server = integrated;
                LOGGER.info("experimental integrated-server world tick freeze enabled");
            }
        });
        ServerLifecycleEvents.SERVER_STOPPING.register(stopping -> {
            server = null;
            if (pendingStep != null) {
                gate.cancelPendingStep();
                pendingStep.reply().complete(pendingStep.finish().apply(new TickGate.Rejected("server_stopping")));
                pendingStep = null;
            }
        });
        ServerTickEvents.START_SERVER_TICK.register(ticked -> {
            if (ticked != server || pendingStep == null) {
                return;
            }
            long clientTicks = Lockstep.clientTicks();
            long clientTickEndsSent = Lockstep.clientTickEndsSent();
            if (gate.clientTickDelivered(clientTicks, clientTickEndsSent, Lockstep.clientTickEndsProcessed.get())) {
                serverStepGrantedNanos = System.nanoTime();
                ticked.tickRateManager().stepGameIfPaused(1);
            }
        });
        ServerTickEvents.END_SERVER_TICK.register(ticked -> {
            if (ticked != server || pendingStep == null) {
                return;
            }
            gate.observeTickEnd(ticked.overworld().getGameTime(), Lockstep.clientTicks()).ifPresent(outcome -> {
                PendingStep finished = pendingStep;
                pendingStep = null;
                finished.reply().complete(finished.finish().apply(outcome));
            });
        });

        Thread listener = new Thread(() -> serve(port), "mcbot-tick-control");
        listener.setDaemon(true);
        listener.start();
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
            case "OBSERVE" -> reply.complete(v2("OBSERVE", observationReply(firstPlayer(current))));
            case "STEP" -> {
                long acceptedNanos = System.nanoTime();
                PlayerAction action = PlayerAction.fromJson(payload);
                requestStep(current, action, reply, outcome -> finishV2Step(current, outcome, acceptedNanos));
            }
            default -> {
                if (!command.startsWith("DEBUG_")) {
                    throw new IllegalArgumentException("unknown_request");
                }
                reply.complete(v2(command, DebugScene.handle(
                        command, current.overworld().getGameTime(), firstPlayer(current), payload)));
            }
        }
    }

    private String finishV2Step(IntegratedServer current, TickGate.StepOutcome outcome, long acceptedNanos) {
        if (outcome instanceof TickGate.Rejected rejected) {
            return "v2 ERROR " + rejected.reason();
        }
        TickGate.Step step = (TickGate.Step) outcome;
        long serverTickEndedNanos = System.nanoTime();
        ServerPlayer player = firstPlayer(current);
        JsonObject reply = observationReply(player);
        long observedNanos = System.nanoTime();

        JsonObject timing = new JsonObject();
        timing.addProperty("client_wait_ms", millis(acceptedNanos, Lockstep.clientTickStartedNanos()));
        timing.addProperty("client_tick_ms", millis(Lockstep.clientTickStartedNanos(), Lockstep.clientTickEndedNanos()));
        timing.addProperty("tick_end_wait_ms", millis(Lockstep.clientTickEndedNanos(), serverStepGrantedNanos));
        timing.addProperty("server_step_ms", millis(serverStepGrantedNanos, serverTickEndedNanos));
        timing.addProperty("observation_ms", millis(serverTickEndedNanos, observedNanos));
        timing.addProperty("total_ms", millis(acceptedNanos, observedNanos));

        JsonObject info = reply.getAsJsonObject("info");
        info.addProperty("step_id", step.stepId());
        info.addProperty("tick_before", step.gameTimeBefore());
        info.addProperty("tick_after", step.gameTimeAfter());
        info.addProperty("client_tick", step.clientTick());
        info.add("timing", timing);
        return v2("STEP", reply);
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
        return json;
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
