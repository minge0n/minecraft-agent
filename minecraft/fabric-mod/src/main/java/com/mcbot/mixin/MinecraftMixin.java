package com.mcbot.mixin;

import com.llamalad7.mixinextras.injector.ModifyExpressionValue;
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

    // Vanilla only continues a held attack (block breaking) while the mouse is grabbed,
    // which observer mode never does.
    @ModifyExpressionValue(
            method = "handleKeybinds",
            at = @At(value = "INVOKE", target = "Lnet/minecraft/client/MouseHandler;isMouseGrabbed()Z"))
    private boolean mcbot$continueScriptedAttack(boolean mouseGrabbed) {
        return mouseGrabbed || McBotClient.scriptedAttackHeld;
    }
}
