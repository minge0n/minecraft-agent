package com.mcbot.mixin;

import com.llamalad7.mixinextras.injector.ModifyExpressionValue;
import com.mcbot.FrameRecorder;
import com.mcbot.Lockstep;
import com.mcbot.McBotClient;
import net.minecraft.client.FramerateLimiter;
import net.minecraft.client.Minecraft;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.Redirect;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(Minecraft.class)
public abstract class MinecraftMixin {
    // Idle wait between skipped frames, so the render loop neither spins nor stalls
    // window event handling while no step is pending.
    private static final int SKIPPED_FRAME_WAIT_FPS = 20;

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

    // Unpaced lockstep: the frame limiter returns as soon as a granted client tick is
    // ready instead of sleeping until the frame deadline.
    @Redirect(
            method = "renderFrame",
            at = @At(value = "INVOKE", target = "Lnet/minecraft/client/FramerateLimiter;limitDisplayFPS(I)V"))
    private void mcbot$limitFramesUnlessStepPending(int framerateLimit) {
        if (Lockstep.unpacedAndGated()) {
            Lockstep.waitForFrameOrClientTick(framerateLimit);
        } else {
            FramerateLimiter.limitDisplayFPS(framerateLimit);
        }
    }

    // Simulation-only benchmarking: skip drawing frames entirely between steps. Client
    // ticks still run from runTick; only the render pass and present are skipped.
    @Inject(method = "renderFrame", at = @At("HEAD"), cancellable = true)
    private void mcbot$skipFrameWhenRenderingOff(boolean advanceGameTime, CallbackInfo callback) {
        if (Lockstep.skipFrame()) {
            Lockstep.waitForFrameOrClientTick(SKIPPED_FRAME_WAIT_FPS);
            callback.cancel();
        }
    }

    // Session recording: the frame is fully drawn and its commands are not yet
    // submitted, so the capture copy joins this frame's command stream.
    @Inject(
            method = "renderFrame",
            at = @At(value = "INVOKE", target = "Lcom/mojang/renderpearl/api/commands/CommandEncoder;submit()V"))
    private void mcbot$captureRecordingFrame(boolean advanceGameTime, CallbackInfo callback) {
        FrameRecorder.onFrameRendered((Minecraft) (Object) this);
    }
}
