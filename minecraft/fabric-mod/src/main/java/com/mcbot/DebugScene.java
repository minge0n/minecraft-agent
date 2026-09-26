package com.mcbot;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.util.Map;
import java.util.TreeMap;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Registry;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.resources.Identifier;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.EntitySpawnReason;
import net.minecraft.world.entity.EntityTypes;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.entity.Mob;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.levelgen.Heightmap;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.Vec3;

final class DebugScene {
    private static final int ARENA_RADIUS = 12;
    private static final int ARENA_HEIGHT = 7;
    private static final double ENTITY_CLEAR_RADIUS = 48.0;
    private static final int MAX_FILL_BLOCKS = 4096;
    private static final int MAX_NEARBY_RADIUS = 32;
    private static final int COMBAT_WEAPON_SLOT = 1;

    private DebugScene() {}

    static JsonObject handle(String command, long gameTime, ServerPlayer player, JsonObject payload) {
        return switch (command) {
            case "DEBUG_SCENE" -> scene(
                    player,
                    payload.get("name").getAsString(),
                    payload.has("ai") && payload.get("ai").getAsBoolean());
            case "DEBUG_FILL" -> fill(player.level(), payload);
            case "DEBUG_BLOCK" -> block(player.level(), blockPos(payload.getAsJsonArray("pos")));
            case "DEBUG_ENTITY" -> entity(player.level(), payload.get("id").getAsInt());
            case "DEBUG_PLAYER" -> player(gameTime, player);
            case "DEBUG_KILL" -> kill(player);
            case "DEBUG_NEARBY" -> nearby(player, payload.get("radius").getAsInt());
            case "DEBUG_REGISTRY" -> registry();
            default -> throw new IllegalArgumentException("unknown_request");
        };
    }

    private static JsonObject scene(ServerPlayer player, String name, boolean ai) {
        ServerLevel level = player.level();
        BlockPos origin = prepareArena(player);
        JsonObject scene = new JsonObject();
        scene.addProperty("name", name);
        scene.add("origin", position(origin));
        switch (name) {
            case "visibility" -> buildVisibility(level, origin, scene);
            case "combat" -> {
                Mob husk = spawnHusk(level, origin, 0.5, 2.5, ai);
                if (ai) {
                    husk.setTarget(player);
                }
                player.getInventory().setItem(COMBAT_WEAPON_SLOT, new ItemStack(Items.STONE_SWORD));
                scene.addProperty("entity", husk.getId());
                scene.addProperty("weapon_slot", COMBAT_WEAPON_SLOT);
            }
            case "mining" -> {
                BlockPos target = origin.offset(0, 1, 2);
                level.setBlock(target, Blocks.DIRT.defaultBlockState(), Block.UPDATE_ALL);
                scene.add("target_block", position(target));
            }
            default -> throw new IllegalArgumentException("unknown_scene");
        }
        return scene;
    }

    private static BlockPos prepareArena(ServerPlayer player) {
        ServerLevel level = player.level();
        BlockPos origin = level.getHeightmapPos(Heightmap.Types.MOTION_BLOCKING, player.blockPosition());
        BlockPos low = origin.offset(-ARENA_RADIUS, 0, -ARENA_RADIUS);
        BlockPos high = origin.offset(ARENA_RADIUS, ARENA_HEIGHT, ARENA_RADIUS);
        for (BlockPos pos : BlockPos.betweenClosed(low, high)) {
            level.setBlock(pos, Blocks.AIR.defaultBlockState(), Block.UPDATE_ALL);
        }
        AABB entityArea = new AABB(origin).inflate(ENTITY_CLEAR_RADIUS);
        for (Entity entity : level.getEntities(player, entityArea, entity -> !(entity instanceof Player))) {
            entity.discard();
        }

        player.connection.teleport(origin.getX() + 0.5, origin.getY(), origin.getZ() + 0.5, 0.0F, 0.0F);
        player.setDeltaMovement(Vec3.ZERO);
        player.resetFallDistance();
        player.removeAllEffects();
        player.clearFire();
        player.setHealth(player.getMaxHealth());
        player.getFoodData().setFoodLevel(20);
        player.getInventory().clearContent();
        return origin;
    }

    private static void buildVisibility(ServerLevel level, BlockPos origin, JsonObject scene) {
        BlockPos visible = origin.offset(0, 1, 5);
        BlockPos hidden = origin.offset(3, 1, 7);
        BlockPos behind = origin.offset(0, 1, -5);
        BlockPos wallFrom = origin.offset(1, 0, 4);
        BlockPos wallTo = origin.offset(7, 3, 4);
        for (BlockPos pos : BlockPos.betweenClosed(wallFrom, wallTo)) {
            level.setBlock(pos, Blocks.STONE.defaultBlockState(), Block.UPDATE_ALL);
        }
        level.setBlock(visible, Blocks.DIAMOND_BLOCK.defaultBlockState(), Block.UPDATE_ALL);
        level.setBlock(hidden, Blocks.EMERALD_BLOCK.defaultBlockState(), Block.UPDATE_ALL);
        level.setBlock(behind, Blocks.GOLD_BLOCK.defaultBlockState(), Block.UPDATE_ALL);
        Mob husk = spawnHusk(level, origin, 5.5, 7.5, false);

        scene.add("visible_block", position(visible));
        scene.add("hidden_block", position(hidden));
        scene.add("behind_block", position(behind));
        scene.add("wall_from", position(wallFrom));
        scene.add("wall_to", position(wallTo));
        scene.addProperty("hidden_entity", husk.getId());
    }

    private static Mob spawnHusk(ServerLevel level, BlockPos origin, double dx, double dz, boolean ai) {
        Mob husk = EntityTypes.HUSK.create(level, EntitySpawnReason.COMMAND);
        if (husk == null) {
            throw new IllegalStateException("spawn_failed");
        }
        husk.snapTo(origin.getX() + dx, origin.getY(), origin.getZ() + dz, 180.0F, 0.0F);
        husk.setNoAi(!ai);
        husk.setPersistenceRequired();
        husk.setSilent(true);
        level.addFreshEntity(husk);
        return husk;
    }

    private static JsonObject fill(ServerLevel level, JsonObject payload) {
        BlockPos from = blockPos(payload.getAsJsonArray("from"));
        BlockPos to = blockPos(payload.getAsJsonArray("to"));
        long volume = (long) (Math.abs(to.getX() - from.getX()) + 1)
                * (Math.abs(to.getY() - from.getY()) + 1)
                * (Math.abs(to.getZ() - from.getZ()) + 1);
        if (volume > MAX_FILL_BLOCKS) {
            throw new IllegalArgumentException("fill_too_large");
        }
        Identifier name = Identifier.parse(payload.get("block").getAsString());
        if (!BuiltInRegistries.BLOCK.containsKey(name)) {
            throw new IllegalArgumentException("unknown_block");
        }
        BlockState state = BuiltInRegistries.BLOCK.getValue(name).defaultBlockState();
        for (BlockPos pos : BlockPos.betweenClosed(from, to)) {
            level.setBlock(pos, state, Block.UPDATE_ALL);
        }
        JsonObject reply = new JsonObject();
        reply.addProperty("filled", volume);
        return reply;
    }

    private static JsonObject block(ServerLevel level, BlockPos pos) {
        JsonObject reply = new JsonObject();
        reply.add("pos", position(pos));
        reply.addProperty("block", BuiltInRegistries.BLOCK.getKey(level.getBlockState(pos).getBlock()).toString());
        return reply;
    }

    private static JsonObject entity(ServerLevel level, int id) {
        Entity entity = level.getEntity(id);
        if (entity == null) {
            JsonObject missing = new JsonObject();
            missing.addProperty("id", id);
            missing.addProperty("exists", false);
            return missing;
        }
        JsonObject reply = describeEntity(entity);
        reply.addProperty("exists", true);
        return reply;
    }

    private static JsonObject describeEntity(Entity entity) {
        JsonObject reply = new JsonObject();
        reply.addProperty("id", entity.getId());
        reply.addProperty("type", BuiltInRegistries.ENTITY_TYPE.getKey(entity.getType()).toString());
        reply.addProperty("x", entity.getX());
        reply.addProperty("y", entity.getY());
        reply.addProperty("z", entity.getZ());
        if (entity instanceof LivingEntity living) {
            reply.addProperty("health", living.getHealth());
            reply.addProperty("hurt_time", living.hurtTime);
            reply.addProperty("alive", living.isAlive());
        }
        return reply;
    }

    private static JsonObject player(long gameTime, ServerPlayer player) {
        JsonObject server = new JsonObject();
        server.addProperty("x", player.getX());
        server.addProperty("y", player.getY());
        server.addProperty("z", player.getZ());
        server.addProperty("yaw", player.getYRot());
        server.addProperty("pitch", player.getXRot());
        server.addProperty("eye_y", player.getEyeY());
        server.addProperty("health", player.getHealth());
        server.addProperty("food", player.getFoodData().getFoodLevel());
        server.addProperty("tick_count", player.tickCount);
        server.addProperty("selected_slot", player.getInventory().getSelectedSlot());

        Lockstep.ClientPlayerSnapshot snapshot = Lockstep.clientPlayer();
        JsonObject client = new JsonObject();
        client.addProperty("tick", snapshot.clientTick());
        client.addProperty("x", snapshot.x());
        client.addProperty("y", snapshot.y());
        client.addProperty("z", snapshot.z());
        client.addProperty("yaw", snapshot.yaw());
        client.addProperty("pitch", snapshot.pitch());
        client.addProperty("tick_count", snapshot.tickCount());

        JsonObject reply = new JsonObject();
        reply.addProperty("game_time", gameTime);
        reply.add("server", server);
        reply.add("client", client);
        return reply;
    }

    private static JsonObject kill(ServerPlayer player) {
        player.kill(player.level());
        JsonObject reply = new JsonObject();
        reply.addProperty("dead", player.isDeadOrDying());
        return reply;
    }

    private static JsonObject nearby(ServerPlayer player, int radius) {
        if (radius < 1 || radius > MAX_NEARBY_RADIUS) {
            throw new IllegalArgumentException("radius_out_of_range");
        }
        ServerLevel level = player.level();
        BlockPos center = BlockPos.containing(player.getEyePosition());
        Map<String, Integer> blocks = new TreeMap<>();
        Map<String, Integer> fluids = new TreeMap<>();
        for (BlockPos pos : BlockPos.betweenClosed(
                center.offset(-radius, -radius, -radius), center.offset(radius, radius, radius))) {
            BlockState state = level.getBlockState(pos);
            if (!state.isAir()) {
                blocks.merge(BuiltInRegistries.BLOCK.getKey(state.getBlock()).toString(), 1, Integer::sum);
            }
            if (!state.getFluidState().isEmpty()) {
                fluids.merge(BuiltInRegistries.FLUID.getKey(state.getFluidState().getType()).toString(), 1, Integer::sum);
            }
        }
        JsonArray entities = new JsonArray();
        for (Entity entity : level.getEntities(player, new AABB(center).inflate(radius), entity -> true)) {
            entities.add(describeEntity(entity));
        }

        JsonObject reply = new JsonObject();
        reply.addProperty("radius", radius);
        reply.add("blocks", counts(blocks));
        reply.add("fluids", counts(fluids));
        reply.add("entities", entities);
        return reply;
    }

    private static JsonObject registry() {
        JsonObject reply = new JsonObject();
        reply.add("block", names(BuiltInRegistries.BLOCK));
        reply.add("fluid", names(BuiltInRegistries.FLUID));
        reply.add("entity", names(BuiltInRegistries.ENTITY_TYPE));
        reply.add("item", names(BuiltInRegistries.ITEM));
        return reply;
    }

    private static <T> JsonArray names(Registry<T> registry) {
        JsonArray names = new JsonArray();
        for (int id = 0; id < registry.size(); id++) {
            names.add(String.valueOf(registry.getKey(registry.byId(id))));
        }
        return names;
    }

    private static JsonObject counts(Map<String, Integer> counts) {
        JsonObject json = new JsonObject();
        counts.forEach(json::addProperty);
        return json;
    }

    private static BlockPos blockPos(JsonArray coordinates) {
        return new BlockPos(coordinates.get(0).getAsInt(), coordinates.get(1).getAsInt(), coordinates.get(2).getAsInt());
    }

    private static JsonArray position(BlockPos pos) {
        JsonArray coordinates = new JsonArray();
        coordinates.add(pos.getX());
        coordinates.add(pos.getY());
        coordinates.add(pos.getZ());
        return coordinates;
    }
}
