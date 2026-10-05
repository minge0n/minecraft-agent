package com.mcbot;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.effect.MobEffectInstance;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.EntityTypes;
import net.minecraft.world.entity.EquipmentSlot;
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
    static final String SCHEMA = "visible-field-v2";
    static final int KIND_NONE = 0;
    static final int KIND_BLOCK = 1;
    static final int KIND_FLUID = 2;
    static final int KIND_ENTITY = 3;
    static final int KINDS = 4;
    // Self state, at the resolution the HUD and the inventory screen show it.
    static final int INVENTORY_SLOTS = 36;
    static final int ARMOR_SLOTS = 4;
    static final int EFFECT_SLOTS = 8;
    static final int AIR_BUBBLES = 10;
    static final int DURABILITY_BAR_WIDTH = 13;
    static final int INFINITE_EFFECT_SECONDS = -1;
    private static final EquipmentSlot[] ARMOR_ORDER = {
        EquipmentSlot.FEET, EquipmentSlot.LEGS, EquipmentSlot.CHEST, EquipmentSlot.HEAD
    };

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
        schema.addProperty("effect_types", BuiltInRegistries.MOB_EFFECT.size());
        schema.addProperty("hotbar_slots", PlayerAction.HOTBAR_SLOTS);
        schema.addProperty("inventory_slots", INVENTORY_SLOTS);
        schema.addProperty("armor_slots", ARMOR_SLOTS);
        schema.addProperty("effect_slots", EFFECT_SLOTS);
        schema.addProperty("air_bubbles", AIR_BUBBLES);
        schema.addProperty("durability_steps", DURABILITY_BAR_WIDTH);
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
        JsonArray inventoryItems = new JsonArray();
        JsonArray inventoryCounts = new JsonArray();
        JsonArray inventoryDurability = new JsonArray();
        for (int slot = 0; slot < INVENTORY_SLOTS; slot++) {
            addStack(inventory.getItem(slot), inventoryItems, inventoryCounts, inventoryDurability);
        }
        JsonArray armorItems = new JsonArray();
        JsonArray armorDurability = new JsonArray();
        for (EquipmentSlot slot : ARMOR_ORDER) {
            addStack(player.getItemBySlot(slot), armorItems, new JsonArray(), armorDurability);
        }
        JsonArray offhandItem = new JsonArray();
        JsonArray offhandCount = new JsonArray();
        JsonArray offhandDurability = new JsonArray();
        addStack(player.getOffhandItem(), offhandItem, offhandCount, offhandDurability);

        JsonObject observation = new JsonObject();
        observation.addProperty("schema", SCHEMA);
        observation.add("ray_kind", kinds);
        observation.add("ray_type", types);
        observation.add("ray_distance", distances);
        observation.addProperty("health", player.getHealth());
        observation.addProperty("max_health", player.getMaxHealth());
        observation.addProperty("absorption", player.getAbsorptionAmount());
        observation.addProperty("food", player.getFoodData().getFoodLevel());
        observation.addProperty("air_bubbles", airBubbles(player.getAirSupply(), player.getMaxAirSupply()));
        observation.addProperty("armor", player.getArmorValue());
        observation.addProperty("xp_level", player.experienceLevel);
        observation.addProperty("xp_progress", player.experienceProgress);
        observation.addProperty("selected_slot", inventory.getSelectedSlot());
        observation.add("inventory_item", inventoryItems);
        observation.add("inventory_count", inventoryCounts);
        observation.add("inventory_durability", inventoryDurability);
        observation.add("armor_item", armorItems);
        observation.add("armor_durability", armorDurability);
        observation.addProperty("offhand_item", offhandItem.get(0).getAsInt());
        observation.addProperty("offhand_count", offhandCount.get(0).getAsInt());
        observation.addProperty("offhand_durability", offhandDurability.get(0).getAsFloat());
        addEffects(player, observation);
        observation.addProperty("pitch", player.getXRot());
        return observation;
    }

    // Item id, count and the durability bar of one stack. The bar is what the HUD
    // shows: 13 steps, full when the item takes no damage or is undamaged.
    private static void addStack(ItemStack stack, JsonArray items, JsonArray counts, JsonArray durability) {
        items.add(BuiltInRegistries.ITEM.getId(stack.getItem()));
        counts.add(stack.getCount());
        durability.add(durabilityFraction(stack.isBarVisible(), stack.getBarWidth()));
    }

    static float durabilityFraction(boolean barVisible, int barWidth) {
        if (!barVisible) {
            return 1.0F;
        }
        return Math.max(0, Math.min(DURABILITY_BAR_WIDTH, barWidth)) / (float) DURABILITY_BAR_WIDTH;
    }

    // The bubbles the HUD draws above the food bar: 10 at full air, rounded up.
    static int airBubbles(int air, int maxAir) {
        if (maxAir <= 0 || air <= 0) {
            return 0;
        }
        return Math.min(AIR_BUBBLES, (int) Math.ceil(air * (double) AIR_BUBBLES / maxAir));
    }

    // Effects with a HUD icon, sorted by registry id, in fixed slots. The type is the
    // registry id plus one, so 0 marks an empty slot. Seconds are rounded up, as the
    // inventory screen shows them; -1 is an infinite effect.
    private static void addEffects(ServerPlayer player, JsonObject observation) {
        List<MobEffectInstance> effects = new ArrayList<>();
        for (MobEffectInstance effect : player.getActiveEffects()) {
            if (effect.showIcon()) {
                effects.add(effect);
            }
        }
        effects.sort(Comparator.comparingInt(effect -> BuiltInRegistries.MOB_EFFECT.getId(effect.getEffect().value())));
        JsonArray types = new JsonArray();
        JsonArray amplifiers = new JsonArray();
        JsonArray seconds = new JsonArray();
        for (int slot = 0; slot < EFFECT_SLOTS; slot++) {
            if (slot < effects.size()) {
                MobEffectInstance effect = effects.get(slot);
                types.add(BuiltInRegistries.MOB_EFFECT.getId(effect.getEffect().value()) + 1);
                amplifiers.add(effect.getAmplifier());
                seconds.add(effect.isInfiniteDuration()
                        ? INFINITE_EFFECT_SECONDS
                        : (effect.getDuration() + 19) / 20);
            } else {
                types.add(0);
                amplifiers.add(0);
                seconds.add(0);
            }
        }
        observation.add("effect_type", types);
        observation.add("effect_amplifier", amplifiers);
        observation.add("effect_seconds", seconds);
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
