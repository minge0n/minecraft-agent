package com.mcbot;

import com.mcbot.mixin.KeyMappingAccessor;
import java.io.IOException;
import java.util.function.Function;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.minecraft.client.KeyMapping;
import net.minecraft.client.Minecraft;
import net.minecraft.client.Options;
import net.minecraft.client.gui.screens.AccessibilityOnboardingScreen;
import net.minecraft.client.gui.screens.Screen;
import net.minecraft.client.gui.screens.TitleScreen;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.client.tutorial.TutorialSteps;
import net.minecraft.core.HolderLookup;
import net.minecraft.core.registries.Registries;
import net.minecraft.network.chat.Component;
import net.minecraft.resources.ResourceKey;
import net.minecraft.util.Mth;
import net.minecraft.world.Difficulty;
import net.minecraft.world.level.GameType;
import net.minecraft.world.level.LevelSettings;
import net.minecraft.world.level.WorldDataConfiguration;
import net.minecraft.world.level.levelgen.WorldDimensions;
import net.minecraft.world.level.levelgen.WorldOptions;
import net.minecraft.world.level.levelgen.presets.WorldPreset;
import net.minecraft.world.level.levelgen.presets.WorldPresets;
import net.minecraft.world.level.storage.LevelStorageSource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class McBotClient implements ClientModInitializer {
    private static final Logger LOGGER = LoggerFactory.getLogger(McBotMod.MOD_ID);
    private static final String LEVEL_ID_PREFIX = "mcbot-episode-";

    public static volatile boolean observerMode;
    public static volatile boolean humanControl;
    public static volatile boolean scriptedAttackHeld;

    private static volatile WorldRequest pendingWorld;
    private static volatile String currentLevelId = "";

    private PlayerAction heldAction = PlayerAction.NOOP;

    @Override
    public void onInitializeClient() {
        String port = System.getenv("MCBOT_TICK_PORT");
        if (port == null) {
            return;
        }

        long worldSeed = Long.parseLong(System.getenv().getOrDefault("MCBOT_WORLD_SEED", "12345"));
        observerMode = true;
        pendingWorld = new WorldRequest(worldSeed, WorldRequest.Preset.FLAT);
        new TickControl().start(Integer.parseInt(port));
        ClientTickEvents.START_CLIENT_TICK.register(this::applyScriptedAction);
        ClientTickEvents.END_CLIENT_TICK.register(minecraft -> {
            finishClientTick(minecraft);
            openPendingWorld(minecraft);
        });
    }

    static String currentLevelId() {
        return currentLevelId;
    }

    static void requestFreshWorld(WorldRequest request) {
        Minecraft minecraft = Minecraft.getInstance();
        minecraft.execute(() -> {
            Lockstep.disarm();
            KeyMapping.releaseAll();
            scriptedAttackHeld = false;
            String previousLevelId = currentLevelId;
            if (minecraft.level != null) {
                minecraft.disconnectFromWorld(Component.literal("mcbot episode reset"));
            }
            deleteEpisodeWorld(minecraft, previousLevelId);
            Lockstep.resetClientTickEnds();
            pendingWorld = request;
        });
    }

    private static void deleteEpisodeWorld(Minecraft minecraft, String levelId) {
        if (!levelId.startsWith(LEVEL_ID_PREFIX)) {
            return;
        }
        try (LevelStorageSource.LevelStorageAccess access = minecraft.getLevelSource().createAccess(levelId)) {
            access.deleteLevel();
        } catch (IOException e) {
            LOGGER.warn("could not delete disposable world {}: {}", levelId, e.toString());
        }
    }

    public static void toggleHumanControl(Minecraft minecraft) {
        humanControl = !humanControl;
        KeyMapping.releaseAll();
        scriptedAttackHeld = false;
        if (humanControl) {
            if (minecraft.gui.screen() == null) {
                minecraft.mouseHandler.grabMouse();
            }
        } else {
            minecraft.mouseHandler.releaseMouse();
        }
        String message = humanControl
                ? "mcbot: human control ON - press ` to release"
                : "mcbot: observer mode - press ` to take control";
        minecraft.gui.hud.setOverlayMessage(Component.literal(message), false);
        LOGGER.info("human control {}", humanControl ? "enabled" : "disabled");
    }

    private void applyScriptedAction(Minecraft minecraft) {
        PlayerAction action = Lockstep.currentClientAction();
        LocalPlayer player = minecraft.player;
        if (action == null || player == null) {
            return;
        }

        player.setYRot(player.getYRot() + action.yawDelta());
        player.setXRot(Mth.clamp(player.getXRot() + action.pitchDelta(), -90.0F, 90.0F));
        if (action.hotbar() != PlayerAction.KEEP_HOTBAR_SLOT) {
            player.getInventory().setSelectedSlot(action.hotbar());
        }
        if (humanControl) {
            heldAction = PlayerAction.NOOP;
            return;
        }

        Options options = minecraft.options;
        options.keyUp.setDown(action.forward());
        options.keyDown.setDown(action.back());
        options.keyLeft.setDown(action.left());
        options.keyRight.setDown(action.right());
        options.keyJump.setDown(action.jump());
        options.keyShift.setDown(action.sneak());
        options.keySprint.setDown(action.sprint());
        holdButton(options.keyAttack, action.attack(), heldAction.attack());
        holdButton(options.keyUse, action.use(), heldAction.use());
        scriptedAttackHeld = action.attack();
        heldAction = action;
    }

    private static void holdButton(KeyMapping key, boolean down, boolean wasDown) {
        key.setDown(down);
        if (down && !wasDown) {
            KeyMappingAccessor clicks = (KeyMappingAccessor) key;
            clicks.mcbot$setClickCount(clicks.mcbot$getClickCount() + 1);
        }
    }

    private void finishClientTick(Minecraft minecraft) {
        boolean sentTickEnd = minecraft.level != null && minecraft.getConnection() != null && !minecraft.isPaused();
        LocalPlayer player = minecraft.player;
        if (player == null) {
            Lockstep.recordClientTick(sentTickEnd, Double.NaN, Double.NaN, Double.NaN, Float.NaN, Float.NaN, -1);
        } else {
            Lockstep.recordClientTick(sentTickEnd, player.getX(), player.getY(), player.getZ(),
                    player.getYRot(), player.getXRot(), player.tickCount);
        }

        if (!Lockstep.clientGated && player != null && minecraft.getConnection() != null
                && minecraft.getConnection().hasClientLoaded() && minecraft.gui.screen() == null) {
            heldAction = PlayerAction.NOOP;
            Lockstep.arm();
            LOGGER.info("lockstep armed: client and player ticks now run only on STEP");
        }
    }

    private void openPendingWorld(Minecraft minecraft) {
        WorldRequest request = pendingWorld;
        if (request == null || minecraft.gui.overlay() != null) {
            return;
        }
        Screen screen = minecraft.gui.screen();
        if (!(screen instanceof TitleScreen || screen instanceof AccessibilityOnboardingScreen)) {
            return;
        }

        pendingWorld = null;
        Options options = minecraft.options;
        options.tutorialStep = TutorialSteps.NONE;
        options.toggleCrouch().set(false);
        options.toggleSprint().set(false);
        options.toggleAttack().set(false);
        options.toggleUse().set(false);
        String levelId = LEVEL_ID_PREFIX + System.currentTimeMillis();
        currentLevelId = levelId;
        LevelSettings settings = new LevelSettings(
                levelId,
                GameType.SURVIVAL,
                new LevelSettings.DifficultySettings(Difficulty.NORMAL, false, false),
                true,
                WorldDataConfiguration.DEFAULT);
        WorldOptions worldOptions = new WorldOptions(request.seed(), false, false);
        ResourceKey<WorldPreset> preset = switch (request.preset()) {
            case FLAT -> WorldPresets.FLAT;
            case NORMAL -> WorldPresets.NORMAL;
        };
        LOGGER.info("creating disposable {} world {} with seed {}", request.preset(), levelId, request.seed());
        minecraft.schedule(() -> minecraft.createWorldOpenFlows()
                .createFreshLevel(levelId, settings, worldOptions, dimensions(preset), new TitleScreen()));
    }

    private static Function<HolderLookup.Provider, WorldDimensions> dimensions(ResourceKey<WorldPreset> preset) {
        return registries -> registries.lookupOrThrow(Registries.WORLD_PRESET)
                .getOrThrow(preset)
                .value()
                .createWorldDimensions();
    }
}
