package com.mcbot;

import com.google.gson.JsonObject;

// Parameters of one recording session, sent with the v2 RECORD_START command.
// episode: episode index echoed to the recorder; port: loopback port of the recorder
// process; everyTicks: capture every Nth client tick; width/height: downscaled frame
// size (even, for 4:2:0 video); queueFrames: capacity of the bounded frame queue
// between the render thread and the sender.
record RecordingSettings(long episode, int port, int everyTicks, int width, int height, int queueFrames) {
    RecordingSettings {
        require(episode >= 0, "invalid_episode");
        require(port > 0 && port <= 65535, "invalid_recording_port");
        require(everyTicks >= 1, "invalid_every_ticks");
        require(width >= 16 && width <= 1920 && width % 2 == 0, "invalid_recording_width");
        require(height >= 16 && height <= 1080 && height % 2 == 0, "invalid_recording_height");
        require(queueFrames >= 1 && queueFrames <= 1024, "invalid_queue_frames");
    }

    static RecordingSettings fromJson(JsonObject payload) {
        try {
            return new RecordingSettings(
                    payload.get("episode").getAsLong(),
                    payload.get("port").getAsInt(),
                    payload.get("every_ticks").getAsInt(),
                    payload.get("width").getAsInt(),
                    payload.get("height").getAsInt(),
                    payload.get("queue_frames").getAsInt());
        } catch (NullPointerException | ClassCastException | UnsupportedOperationException | NumberFormatException e) {
            throw new IllegalArgumentException("invalid_recording_settings");
        }
    }

    private static void require(boolean condition, String reason) {
        if (!condition) {
            throw new IllegalArgumentException(reason);
        }
    }
}
