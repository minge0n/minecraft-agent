package com.mcbot.mixin;

import com.mcbot.StepEvents;
import net.minecraft.advancements.AdvancementHolder;
import net.minecraft.advancements.AdvancementProgress;
import net.minecraft.server.PlayerAdvancements;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

// Reports an advancement once it becomes complete, for the privileged step events.
@Mixin(PlayerAdvancements.class)
public abstract class PlayerAdvancementsMixin {
    @Shadow
    public abstract AdvancementProgress getOrStartProgress(AdvancementHolder holder);

    @Inject(method = "award", at = @At("RETURN"))
    private void mcbot$reportCompletion(
            AdvancementHolder holder, String criterion, CallbackInfoReturnable<Boolean> callback) {
        if (callback.getReturnValueZ() && getOrStartProgress(holder).isDone()) {
            StepEvents.advancementCompleted(holder);
        }
    }
}
