package com.mcbot.mixin;

import com.mcbot.Lockstep;
import net.minecraft.network.PacketListener;
import net.minecraft.network.PacketProcessor;
import net.minecraft.network.protocol.Packet;
import net.minecraft.network.protocol.common.ClientboundPingPacket;
import net.minecraft.network.protocol.game.ServerboundClientTickEndPacket;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

// Observes when lockstep-relevant packets become queued for their main thread, after the
// enqueue, so a woken thread is guaranteed to find them.
@Mixin(PacketProcessor.class)
public abstract class PacketProcessorMixin {
    @Inject(method = "scheduleIfPossible", at = @At("TAIL"))
    private <T extends PacketListener> void mcbot$observeQueuedPacket(T listener, Packet<T> packet, CallbackInfo callback) {
        if (packet instanceof ServerboundClientTickEndPacket) {
            Lockstep.onClientTickEndQueued();
        } else if (packet instanceof ClientboundPingPacket ping) {
            Lockstep.onServerMarkerQueued(ping.getId());
        }
    }
}
