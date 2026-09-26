package com.mcbot.mixin;

import com.mcbot.Lockstep;
import net.minecraft.server.ServerTickRateManager;
import net.minecraft.world.TickRateManager;
import net.minecraft.world.entity.Entity;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

@Mixin(TickRateManager.class)
public abstract class TickRateManagerMixin {
    @Shadow
    public abstract boolean runsNormally();

    @Inject(method = "isEntityFrozen", at = @At("HEAD"), cancellable = true)
    private void mcbot$freezePlayersInLockstep(Entity entity, CallbackInfoReturnable<Boolean> callback) {
        if (Lockstep.playersGated && (Object) this instanceof ServerTickRateManager) {
            callback.setReturnValue(!runsNormally());
        }
    }
}
