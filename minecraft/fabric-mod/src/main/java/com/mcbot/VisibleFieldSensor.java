package com.mcbot;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.util.List;
import java.util.Optional;
import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.EntityTypes;
import net.minecraft.world.entity.player.Inventory;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.BlockGetter;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.InfestedBlock;
import net.minecraft.world.level.block.RenderShape;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.material.FluidState;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.BlockHitResult;
import net.minecraft.world.phys.Vec3;

final class VisibleFieldSensor {
    static final String SCHEMA = "visible-field-v1";
    static final int KIND_NONE = 0;
    static final int KIND_BLOCK = 1;
    static final int KIND_FLUID = 2;
    static final int KIND_ENTITY = 3;
    static final int KINDS = 4;

    private record Hit(int kind, int type, double distance) {}

    private static final Hit MISS = new Hit(KIND_NONE, 0, CameraRays.MAX_DISTANCE);

    private VisibleFieldSensor() {}

    static JsonObject schema() {
        JsonObject schema = new JsonObject();
        schema.addProperty("observation_schema", SCHEMA);
        schema.addProperty("rows", CameraRays.ROWS);
        schema.addProperty("columns", CameraRays.COLUMNS);
        schema.addProperty("vertical_fov_degrees", CameraRays.VERTICAL_FOV_DEGREES);
        schema.addProperty("horizontal_fov_degrees", 2.0 * CameraRays.horizontalHalfFovDegrees());
        schema.addProperty("max_distance", CameraRays.MAX_DISTANCE);
        schema.addProperty("ray_kinds", KINDS);
        schema.addProperty("block_types", BuiltInRegistries.BLOCK.size());
        schema.addProperty("fluid_types", BuiltInRegistries.FLUID.size());
        schema.addProperty("entity_types", BuiltInRegistries.ENTITY_TYPE.size());
        schema.addProperty("item_types", BuiltInRegistries.ITEM.size());
        schema.addProperty("hotbar_slots", PlayerAction.HOTBAR_SLOTS);
        schema.addProperty("max_camera_delta_degrees", PlayerAction.MAX_CAMERA_DELTA_DEGREES);
        return schema;
    }

    static JsonObject observe(ServerPlayer player) {
        ServerLevel level = player.level();
        Vec3 eye = player.getEyePosition();
        CameraRays.Basis basis = CameraRays.basis(player.getYRot(), player.getXRot());
        List<Entity> entities = level.getEntities(
                player, new AABB(eye, eye).inflate(CameraRays.MAX_DISTANCE), entity -> isVisibleEntity(entity, player));

        JsonArray kinds = new JsonArray();
        JsonArray types = new JsonArray();
        JsonArray distances = new JsonArray();
        for (int row = 0; row < CameraRays.ROWS; row++) {
            for (int column = 0; column < CameraRays.COLUMNS; column++) {
                double[] direction = basis.direction(row, column);
                Vec3 end = eye.add(
                        direction[0] * CameraRays.MAX_DISTANCE,
                        direction[1] * CameraRays.MAX_DISTANCE,
                        direction[2] * CameraRays.MAX_DISTANCE);
                Hit blockHit = BlockGetter.traverseBlocks(
                        eye, end, level, (context, pos) -> blockOrFluidHit(context, eye, end, pos), context -> MISS);
                Hit hit = nearestEntityHit(entities, eye, end, blockHit);
                kinds.add(hit.kind());
                types.add(hit.type());
                distances.add((float) hit.distance());
            }
        }

        Inventory inventory = player.getInventory();
        JsonArray hotbarItems = new JsonArray();
        JsonArray hotbarCounts = new JsonArray();
        for (int slot = 0; slot < PlayerAction.HOTBAR_SLOTS; slot++) {
            ItemStack stack = inventory.getItem(slot);
            hotbarItems.add(BuiltInRegistries.ITEM.getId(stack.getItem()));
            hotbarCounts.add(stack.getCount());
        }

        JsonObject observation = new JsonObject();
        observation.addProperty("schema", SCHEMA);
        observation.add("ray_kind", kinds);
        observation.add("ray_type", types);
        observation.add("ray_distance", distances);
        observation.addProperty("health", player.getHealth());
        observation.addProperty("food", player.getFoodData().getFoodLevel());
        observation.addProperty("selected_slot", inventory.getSelectedSlot());
        observation.add("hotbar_item", hotbarItems);
        observation.add("hotbar_count", hotbarCounts);
        observation.addProperty("pitch", player.getXRot());
        return observation;
    }

    private static boolean isVisibleEntity(Entity entity, ServerPlayer viewer) {
        return !entity.isSpectator()
                && !entity.isInvisibleTo(viewer)
                && entity.getType() != EntityTypes.MARKER
                && entity.getType() != EntityTypes.INTERACTION;
    }

    private static Hit nearestEntityHit(List<Entity> entities, Vec3 eye, Vec3 end, Hit blockHit) {
        Hit nearest = blockHit;
        for (Entity entity : entities) {
            Optional<Vec3> hit = entity.getBoundingBox().clip(eye, end);
            if (hit.isEmpty()) {
                continue;
            }
            double distance = eye.distanceTo(hit.get());
            if (distance < nearest.distance()) {
                nearest = new Hit(KIND_ENTITY, BuiltInRegistries.ENTITY_TYPE.getId(entity.getType()), distance);
            }
        }
        return nearest;
    }

    private static Hit blockOrFluidHit(ServerLevel level, Vec3 from, Vec3 to, BlockPos pos) {
        if (!level.hasChunkAt(pos)) {
            return MISS;
        }
        BlockState state = level.getBlockState(pos);
        Hit nearest = null;
        if (isRendered(state)) {
            BlockHitResult blockHit = state.getShape(level, pos).clip(from, to, pos);
            if (blockHit != null) {
                nearest = new Hit(KIND_BLOCK, visibleBlockType(state), from.distanceTo(blockHit.getLocation()));
            }
        }
        FluidState fluid = state.getFluidState();
        if (!fluid.isEmpty()) {
            BlockHitResult fluidHit = fluid.getShape(level, pos).clip(from, to, pos);
            if (fluidHit != null) {
                double distance = from.distanceTo(fluidHit.getLocation());
                if (nearest == null || distance < nearest.distance()) {
                    nearest = new Hit(KIND_FLUID, BuiltInRegistries.FLUID.getId(fluid.getType()), distance);
                }
            }
        }
        return nearest;
    }

    private static boolean isRendered(BlockState state) {
        return !state.isAir() && (state.getRenderShape() != RenderShape.INVISIBLE || state.hasBlockEntity());
    }

    private static int visibleBlockType(BlockState state) {
        Block block = state.getBlock();
        if (block instanceof InfestedBlock infested) {
            block = infested.getHostBlock();
        }
        return BuiltInRegistries.BLOCK.getId(block);
    }
}
