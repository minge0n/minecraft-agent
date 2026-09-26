package com.mcbot.mixin;

import com.mcbot.Lockstep;
import net.minecraft.client.DeltaTracker;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

@Mixin(DeltaTracker.Timer.class)
public abstract class DeltaTrackerTimerMixin {
    @Inject(method = "advanceGameTime", at = @At("RETURN"), cancellable = true)
    private void mcbot$runOnlyGrantedClientTicks(long currentMs, CallbackInfoReturnable<Integer> callback) {
        if (Lockstep.clientGated) {
            callback.setReturnValue(Lockstep.takeClientTickPermit() ? 1 : 0);
        }
    }
}
