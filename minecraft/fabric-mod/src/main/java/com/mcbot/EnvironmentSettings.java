package com.mcbot;

import com.google.gson.JsonObject;
import net.minecraft.client.CloudStatus;
import net.minecraft.client.GraphicsPreset;
import net.minecraft.client.InactivityFpsLimit;
import net.minecraft.client.Minecraft;
import net.minecraft.client.Options;
import net.minecraft.server.level.ParticleStatus;
import net.minecraft.sounds.SoundSource;

// Environment client settings applied in observer mode, independent of any personal
// options file. Rendering is for human observation and recording only; the policy
// observation never reads the framebuffer. Simulation distance is a simulation
// parameter and is fixed and reported so runs stay comparable.
final class EnvironmentSettings {
    // The lowest render distance that keeps the sensor range loaded and every chunk
    // inside the simulation distance entity-ticking (docs/decisions/simulation-throughput.md).
    static final int RENDER_DISTANCE_CHUNKS = 6;
    // The integrated server uses the render distance as its view distance, so it
    // also sets which chunks stay loaded. Values outside this range are refused.
    static final int MIN_RENDER_DISTANCE_CHUNKS = 2;
    static final int MAX_RENDER_DISTANCE_CHUNKS = 32;
    static final int SIMULATION_DISTANCE_CHUNKS = 6;
    static final int FRAME_LIMIT = 260;

    private EnvironmentSettings() {}

    // MCBOT_RENDER_DISTANCE selects the render distance in chunks for a benchmark;
    // without it the client uses RENDER_DISTANCE_CHUNKS.
    static int renderDistance(String value) {
        if (value == null || value.isBlank()) {
            return RENDER_DISTANCE_CHUNKS;
        }
        int chunks;
        try {
            chunks = Integer.parseInt(value.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("MCBOT_RENDER_DISTANCE must be an integer", e);
        }
        if (chunks < MIN_RENDER_DISTANCE_CHUNKS || chunks > MAX_RENDER_DISTANCE_CHUNKS) {
            throw new IllegalArgumentException("MCBOT_RENDER_DISTANCE out of range: " + chunks);
        }
        return chunks;
    }

    static void apply(Minecraft minecraft) {
        Options options = minecraft.options;
        options.graphicsPreset().set(GraphicsPreset.FAST);
        options.renderDistance().set(renderDistance(System.getenv("MCBOT_RENDER_DISTANCE")));
        options.simulationDistance().set(SIMULATION_DISTANCE_CHUNKS);
        options.enableVsync().set(false);
        options.framerateLimit().set(FRAME_LIMIT);
        options.inactivityFpsLimit().set(InactivityFpsLimit.MINIMIZED);
        options.particles().set(ParticleStatus.MINIMAL);
        options.cloudStatus().set(CloudStatus.OFF);
        options.entityShadows().set(false);
        options.pauseOnLostFocus = false;
        mute(options);
    }

    // The environment plays no sound: the policy never hears it, and a muted
    // client is quieter for the person next to the machine. Set MCBOT_SOUND=on
    // to keep the volumes of the options file.
    static boolean soundEnabled() {
        return "on".equalsIgnoreCase(System.getenv("MCBOT_SOUND"));
    }

    static void mute(Options options) {
        if (!soundEnabled()) {
            options.getSoundSourceOptionInstance(SoundSource.MASTER).set(0.0);
        }
    }

    static JsonObject describe(Minecraft minecraft) {
        Options options = minecraft.options;
        JsonObject json = new JsonObject();
        json.addProperty("graphics_preset", options.graphicsPreset().get().getSerializedName());
        json.addProperty("render_distance_chunks", options.renderDistance().get());
        json.addProperty("simulation_distance_chunks", options.simulationDistance().get());
        json.addProperty("vsync", options.enableVsync().get());
        json.addProperty("frame_limit", options.framerateLimit().get());
        json.addProperty("particles", options.particles().get().name().toLowerCase(java.util.Locale.ROOT));
        json.addProperty("master_volume", options.getSoundSourceVolume(SoundSource.MASTER));
        json.addProperty("framebuffer_width", minecraft.getWindow().getWidth());
        json.addProperty("framebuffer_height", minecraft.getWindow().getHeight());
        return json;
    }
}
