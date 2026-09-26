package com.mcbot.mixin;

import com.mcbot.McBotClient;
import com.mojang.blaze3d.platform.InputConstants;
import net.minecraft.client.KeyboardHandler;
import net.minecraft.client.Minecraft;
import net.minecraft.client.input.KeyEvent;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(KeyboardHandler.class)
public abstract class KeyboardHandlerMixin {
    @Inject(method = "keyPress", at = @At("HEAD"), cancellable = true)
    private void mcbot$filterKeysInObserverMode(long handle, int action, KeyEvent event, CallbackInfo callback) {
        if (!McBotClient.observerMode) {
            return;
        }
        if (event.key() == InputConstants.KEY_GRAVE) {
            if (action == InputConstants.PRESS) {
                McBotClient.toggleHumanControl(Minecraft.getInstance());
            }
            callback.cancel();
            return;
        }
        if (!McBotClient.humanControl) {
            callback.cancel();
        }
    }

    @Inject(method = "charTyped", at = @At("HEAD"), cancellable = true)
    private void mcbot$filterCharsInObserverMode(CallbackInfo callback) {
        if (McBotClient.observerMode && !McBotClient.humanControl) {
            callback.cancel();
        }
    }
}
