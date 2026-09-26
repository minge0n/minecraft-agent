package com.mcbot.mixin;

import com.mcbot.McBotClient;
import net.minecraft.client.MouseHandler;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(MouseHandler.class)
public abstract class MouseHandlerMixin {
    @Inject(method = {"grabMouse", "onButton", "onScroll"}, at = @At("HEAD"), cancellable = true)
    private void mcbot$ignoreMouseInObserverMode(CallbackInfo callback) {
        if (McBotClient.observerMode) {
            callback.cancel();
        }
    }
}
