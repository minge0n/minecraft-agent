package com.mcbot;

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
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
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

    private final TickGate gate = new TickGate();
    private volatile IntegratedServer server;
    private CompletableFuture<String> pendingStepReply;
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
            if (pendingStepReply != null) {
                gate.cancelPendingStep();
                pendingStepReply.complete("v1 ERROR server_stopping");
                pendingStepReply = null;
            }
        });
        ServerTickEvents.START_SERVER_TICK.register(ticked -> {
            if (ticked != server || pendingStepReply == null) {
                return;
            }
            long clientTicks = Lockstep.clientTicks();
            long clientTickEndsSent = Lockstep.clientTickEndsSent();
            if (gate.clientTickDelivered(clientTicks, clientTickEndsSent, Lockstep.clientTickEndsProcessed.get())) {
                ticked.tickRateManager().stepGameIfPaused(1);
            }
        });
        ServerTickEvents.END_SERVER_TICK.register(ticked -> {
            if (ticked != server || pendingStepReply == null) {
                return;
            }
            gate.observeTickEnd(ticked.overworld().getGameTime(), Lockstep.clientTicks()).ifPresent(result -> {
                pendingStepReply.complete(TickGate.encode(result));
                pendingStepReply = null;
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
                    for (String request = input.readLine(); request != null; request = input.readLine()) {
                        output.write(respond(request));
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
            Minecraft minecraft = Minecraft.getInstance();
            minecraft.execute(minecraft::stop);
            return "v1 QUIT";
        }

        IntegratedServer current = server;
        if (current == null) {
            return "v1 ERROR no_integrated_server";
        }
        CompletableFuture<String> reply = new CompletableFuture<>();
        current.execute(() -> handle(current, request, reply));
        try {
            return reply.get(5, TimeUnit.SECONDS);
        } catch (TimeoutException e) {
            current.execute(() -> cancelStep(current, reply));
            return "v1 ERROR timeout";
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return "v1 ERROR interrupted";
        } catch (Exception e) {
            return "v1 ERROR " + e.getClass().getSimpleName();
        }
    }

    private void handle(IntegratedServer current, String request, CompletableFuture<String> reply) {
        if (reply.isDone()) {
            return;
        }
        long gameTime = current.overworld().getGameTime();
        boolean frozen = current.tickRateManager().isFrozen();
        switch (request) {
            case "v1 STATUS" -> reply.complete(TickGate.encode(new TickGate.Status(
                    gameTime, current.getTickCount(), frozen, current.isPaused(),
                    Lockstep.clientGated, Lockstep.clientTicks())));
            case "v1 STEP NOOP" -> requestStep(current, gameTime, frozen, Lockstep.ClientAction.NOOP, reply);
            case "v1 STEP FORWARD" -> requestStep(current, gameTime, frozen, Lockstep.ClientAction.FORWARD, reply);
            case "v1 DEBUG_SPAWN" -> reply.complete(spawnProbeEntities(current));
            case "v1 DEBUG_PROBE" -> reply.complete(probeEntities(current));
            case "v1 DEBUG_PLAYER" -> reply.complete(probePlayer(current));
            default -> reply.complete("v1 ERROR unknown_request");
        }
    }

    private void requestStep(
            IntegratedServer current,
            long gameTime,
            boolean frozen,
            Lockstep.ClientAction action,
            CompletableFuture<String> reply) {
        var rejected = gate.requestStep(gameTime, Lockstep.clientTicks(), frozen, current.isPaused(), Lockstep.clientGated);
        if (rejected.isPresent()) {
            reply.complete(TickGate.encode(rejected.get()));
            return;
        }
        pendingStepReply = reply;
        Lockstep.grantClientTick(action);
    }

    private void cancelStep(IntegratedServer current, CompletableFuture<String> reply) {
        if (pendingStepReply != reply) {
            return;
        }
        pendingStepReply = null;
        gate.cancelPendingStep();
        Lockstep.revokeClientTick();
        current.tickRateManager().stopStepping();
    }

    private String spawnProbeEntities(IntegratedServer current) {
        List<ServerPlayer> players = current.getPlayerList().getPlayers();
        if (players.isEmpty()) {
            return "v1 ERROR no_player";
        }
        ServerPlayer player = players.getFirst();
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
        List<ServerPlayer> players = current.getPlayerList().getPlayers();
        if (players.isEmpty()) {
            return "v1 ERROR no_player";
        }
        ServerPlayer player = players.getFirst();
        Entity armorStand = player.level().getEntity(armorStandId);
        Entity husk = player.level().getEntity(huskId);
        if (armorStand == null || husk == null) {
            return "v1 ERROR probe_entities_missing";
        }
        return "v1 DEBUG_PROBE " + current.overworld().getGameTime() + " " + armorStand.getY()
                + " " + husk.getX() + " " + husk.getZ() + " " + husk.distanceTo(player);
    }

    private String probePlayer(IntegratedServer current) {
        List<ServerPlayer> players = current.getPlayerList().getPlayers();
        if (players.isEmpty()) {
            return "v1 ERROR no_player";
        }
        ServerPlayer player = players.getFirst();
        int playTime = player.getStats().getValue(Stats.CUSTOM.get(Stats.PLAY_TIME));
        Lockstep.ClientPlayerSnapshot client = Lockstep.clientPlayer();
        return "v1 DEBUG_PLAYER " + current.overworld().getGameTime()
                + " " + player.getX() + " " + player.getZ() + " " + player.tickCount + " " + playTime
                + " " + client.clientTick() + " " + client.x() + " " + client.z() + " " + client.tickCount();
    }
}
