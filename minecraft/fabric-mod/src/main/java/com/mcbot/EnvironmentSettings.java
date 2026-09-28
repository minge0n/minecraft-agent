package com.mcbot;

import com.google.gson.JsonObject;
import net.minecraft.client.CloudStatus;
import net.minecraft.client.GraphicsPreset;
import net.minecraft.client.InactivityFpsLimit;
import net.minecraft.client.Minecraft;
import net.minecraft.client.Options;
import net.minecraft.server.level.ParticleStatus;

// Environment client settings applied in observer mode, independent of any personal
// options file. Rendering is for human observation and recording only; the policy
// observation never reads the framebuffer. Simulation distance is a simulation
// parameter and is fixed and reported so runs stay comparable.
final class EnvironmentSettings {
    static final int RENDER_DISTANCE_CHUNKS = 8;
    static final int SIMULATION_DISTANCE_CHUNKS = 6;
    static final int FRAME_LIMIT = 260;

    private EnvironmentSettings() {}

    static void apply(Minecraft minecraft) {
        Options options = minecraft.options;
        options.graphicsPreset().set(GraphicsPreset.FAST);
        options.renderDistance().set(RENDER_DISTANCE_CHUNKS);
        options.simulationDistance().set(SIMULATION_DISTANCE_CHUNKS);
        options.enableVsync().set(false);
        options.framerateLimit().set(FRAME_LIMIT);
        options.inactivityFpsLimit().set(InactivityFpsLimit.MINIMIZED);
        options.particles().set(ParticleStatus.MINIMAL);
        options.cloudStatus().set(CloudStatus.OFF);
        options.entityShadows().set(false);
        options.pauseOnLostFocus = false;
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
        json.addProperty("framebuffer_width", minecraft.getWindow().getWidth());
        json.addProperty("framebuffer_height", minecraft.getWindow().getHeight());
        return json;
    }
}
