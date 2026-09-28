package com.mcbot.mixin;

import com.mcbot.Lockstep;
import java.util.function.BooleanSupplier;
import net.minecraft.server.MinecraftServer;
import net.minecraft.util.Util;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

// Unpaced lockstep: while a step's packets or its granted tick are pending, the server
// loop reports no remaining time, so it stops waiting for the wall-clock deadline and
// runs the next iteration at once. The deadline restarts from now, so idle frozen
// iterations keep vanilla pacing. Tick contents are unchanged.
@Mixin(MinecraftServer.class)
public abstract class MinecraftServerMixin {
    @Shadow
    private long nextTickTimeNanos;

    @Inject(method = "haveTime", at = @At("HEAD"), cancellable = true)
    private void mcbot$runPendingLockstepWork(CallbackInfoReturnable<Boolean> callback) {
        if (Lockstep.serverWorkPending()) {
            nextTickTimeNanos = Util.getNanos();
            callback.setReturnValue(false);
        }
    }

    // Grant a ready step before the tick-rate manager decides whether this iteration
    // runs game elements, so the granted tick is this iteration rather than the next.
    @Inject(
            method = "tickServer",
            at = @At(value = "INVOKE", target = "Lnet/minecraft/server/ServerTickRateManager;tick()V"))
    private void mcbot$grantReadyStep(BooleanSupplier haveTime, CallbackInfo callback) {
        Lockstep.beforeTickRateUpdate((MinecraftServer) (Object) this);
    }
}
