package com.mcbot.mixin;

import com.mcbot.Lockstep;
import net.minecraft.network.protocol.game.ServerboundClientTickEndPacket;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.server.network.ServerGamePacketListenerImpl;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

@Mixin(ServerGamePacketListenerImpl.class)
public abstract class ServerGamePacketListenerImplMixin {
    @Shadow
    public ServerPlayer player;

    @Inject(method = "tickPlayer", at = @At("HEAD"), cancellable = true)
    private void mcbot$gatePlayerTick(CallbackInfoReturnable<Boolean> callback) {
        if (Lockstep.playersGated && !player.level().tickRateManager().runsNormally()) {
            callback.setReturnValue(false);
        }
    }

    @Inject(method = "handleClientTickEnd", at = @At("TAIL"))
    private void mcbot$countClientTickEnd(ServerboundClientTickEndPacket packet, CallbackInfo callback) {
        Lockstep.clientTickEndsProcessed.incrementAndGet();
    }
}
