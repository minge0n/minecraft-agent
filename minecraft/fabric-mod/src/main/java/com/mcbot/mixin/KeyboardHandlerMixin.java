package com.mcbot.mixin;

import com.mcbot.McBotClient;
import net.minecraft.client.KeyboardHandler;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(KeyboardHandler.class)
public abstract class KeyboardHandlerMixin {
    @Inject(method = {"keyPress", "charTyped"}, at = @At("HEAD"), cancellable = true)
    private void mcbot$ignoreKeysInObserverMode(CallbackInfo callback) {
        if (McBotClient.observerMode) {
            callback.cancel();
        }
    }
}
