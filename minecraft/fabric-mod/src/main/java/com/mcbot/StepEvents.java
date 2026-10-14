package com.mcbot;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.util.ArrayList;
import java.util.List;
import net.fabricmc.fabric.api.event.player.PlayerBlockBreakEvents;
import net.minecraft.advancements.AdvancementHolder;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.server.level.ServerPlayer;

// Privileged per-step events for the reward (docs/decisions/reward.md): the block types
// that the player broke and the advancements it completed during the step. The step
// reply carries them in an `events` field next to the observation, never inside it.
public final class StepEvents {
    private static final List<Integer> brokenBlocks = new ArrayList<>();
    private static final List<String> advancements = new ArrayList<>();
    private static JsonObject abstractResult;

    private StepEvents() {}

    static void register() {
        PlayerBlockBreakEvents.AFTER.register((level, player, pos, state, blockEntity) -> {
            if (player instanceof ServerPlayer) {
                synchronized (StepEvents.class) {
                    brokenBlocks.add(BuiltInRegistries.BLOCK.getId(state.getBlock()));
                }
            }
        });
    }

    // Called from the PlayerAdvancements mixin when an advancement becomes complete.
    // Advancements without a display are recipe unlocks; they are reported as such.
    public static synchronized void advancementCompleted(AdvancementHolder holder) {
        if (holder.value().display().isPresent()) {
            advancements.add(holder.id().toString());
        }
    }

    static synchronized void abstractAction(JsonObject result) {
        abstractResult = result;
    }

    // Events since the last call; called once per completed step.
    static synchronized JsonObject drain() {
        JsonObject events = new JsonObject();
        JsonArray broken = new JsonArray();
        brokenBlocks.forEach(broken::add);
        JsonArray completed = new JsonArray();
        advancements.forEach(completed::add);
        events.add("blocks_broken", broken);
        events.add("advancements", completed);
        if (abstractResult != null) {
            events.add("abstract", abstractResult);
        }
        brokenBlocks.clear();
        advancements.clear();
        abstractResult = null;
        return events;
    }

    static synchronized void clear() {
        drain();
    }
}
