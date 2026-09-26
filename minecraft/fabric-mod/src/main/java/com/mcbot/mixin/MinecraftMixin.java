package com.mcbot.mixin;

import com.mcbot.McBotClient;
import net.minecraft.client.Minecraft;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(Minecraft.class)
public abstract class MinecraftMixin {
    @Inject(method = "pauseGame", at = @At("HEAD"), cancellable = true)
    private void mcbot$suppressPauseMenu(CallbackInfo callback) {
        if (McBotClient.observerMode) {
            callback.cancel();
        }
    }
}
