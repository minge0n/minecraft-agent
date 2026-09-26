package com.mcbot;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.minecraft.client.KeyMapping;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.screens.AccessibilityOnboardingScreen;
import net.minecraft.client.gui.screens.Screen;
import net.minecraft.client.gui.screens.TitleScreen;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.client.tutorial.TutorialSteps;
import net.minecraft.core.HolderLookup;
import net.minecraft.core.registries.Registries;
import net.minecraft.network.chat.Component;
import net.minecraft.world.Difficulty;
import net.minecraft.world.level.GameType;
import net.minecraft.world.level.LevelSettings;
import net.minecraft.world.level.WorldDataConfiguration;
import net.minecraft.world.level.levelgen.WorldDimensions;
import net.minecraft.world.level.levelgen.WorldOptions;
import net.minecraft.world.level.levelgen.presets.WorldPresets;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class McBotClient implements ClientModInitializer {
    private static final Logger LOGGER = LoggerFactory.getLogger(McBotMod.MOD_ID);

    public static volatile boolean observerMode;
    public static volatile boolean humanControl;

    private boolean probeWorldRequested;
    private boolean scriptedForwardDown;

    @Override
    public void onInitializeClient() {
        String port = System.getenv("MCBOT_TICK_PORT");
        if (port == null) {
            return;
        }

        long worldSeed = Long.parseLong(System.getenv().getOrDefault("MCBOT_WORLD_SEED", "12345"));
        observerMode = true;
        new TickControl().start(Integer.parseInt(port));
        ClientTickEvents.START_CLIENT_TICK.register(this::applyScriptedAction);
        ClientTickEvents.END_CLIENT_TICK.register(minecraft -> {
            finishClientTick(minecraft);
            openProbeWorld(minecraft, worldSeed);
        });
    }

    public static void toggleHumanControl(Minecraft minecraft) {
        humanControl = !humanControl;
        if (humanControl) {
            if (minecraft.gui.screen() == null) {
                minecraft.mouseHandler.grabMouse();
            }
        } else {
            minecraft.mouseHandler.releaseMouse();
            KeyMapping.releaseAll();
        }
        String message = humanControl
                ? "mcbot: human control ON - press ` to release"
                : "mcbot: observer mode - press ` to take control";
        minecraft.gui.hud.setOverlayMessage(Component.literal(message), false);
        LOGGER.info("human control {}", humanControl ? "enabled" : "disabled");
    }

    private void applyScriptedAction(Minecraft minecraft) {
        if (Lockstep.currentClientAction() == Lockstep.ClientAction.FORWARD) {
            minecraft.options.keyUp.setDown(true);
            scriptedForwardDown = true;
        }
    }

    private void finishClientTick(Minecraft minecraft) {
        if (scriptedForwardDown) {
            minecraft.options.keyUp.setDown(false);
            scriptedForwardDown = false;
        }

        boolean sentTickEnd = minecraft.level != null && minecraft.getConnection() != null && !minecraft.isPaused();
        LocalPlayer player = minecraft.player;
        if (player == null) {
            Lockstep.recordClientTick(sentTickEnd, Double.NaN, Double.NaN, -1);
        } else {
            Lockstep.recordClientTick(sentTickEnd, player.getX(), player.getZ(), player.tickCount);
        }

        if (!Lockstep.clientGated && player != null && minecraft.getConnection() != null
                && minecraft.getConnection().hasClientLoaded() && minecraft.gui.screen() == null) {
            Lockstep.arm();
            LOGGER.info("lockstep armed: client and player ticks now run only on STEP");
        }
    }

    private void openProbeWorld(Minecraft minecraft, long worldSeed) {
        if (probeWorldRequested || minecraft.gui.overlay() != null) {
            return;
        }
        Screen screen = minecraft.gui.screen();
        if (!(screen instanceof TitleScreen || screen instanceof AccessibilityOnboardingScreen)) {
            return;
        }

        probeWorldRequested = true;
        minecraft.options.tutorialStep = TutorialSteps.NONE;
        String levelId = "mcbot-tick-probe-" + System.currentTimeMillis();
        LevelSettings settings = new LevelSettings(
                levelId,
                GameType.SURVIVAL,
                new LevelSettings.DifficultySettings(Difficulty.NORMAL, false, false),
                true,
                WorldDataConfiguration.DEFAULT);
        WorldOptions options = new WorldOptions(worldSeed, false, false);
        LOGGER.info("creating disposable flat probe world {} with seed {}", levelId, worldSeed);
        minecraft.schedule(() -> minecraft.createWorldOpenFlows()
                .createFreshLevel(levelId, settings, options, McBotClient::flatWorld, new TitleScreen()));
    }

    private static WorldDimensions flatWorld(HolderLookup.Provider registries) {
        return registries.lookupOrThrow(Registries.WORLD_PRESET)
                .getOrThrow(WorldPresets.FLAT)
                .value()
                .createWorldDimensions();
    }
}
