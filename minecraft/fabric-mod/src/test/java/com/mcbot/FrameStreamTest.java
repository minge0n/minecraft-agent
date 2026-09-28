package com.mcbot;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import org.junit.jupiter.api.Test;

class FrameStreamTest {
    private static void assertHeader(DataInputStream in, byte type) throws IOException {
        assertEquals(FrameStream.MAGIC, in.readInt());
        assertEquals(FrameStream.VERSION, in.readShort());
        assertEquals(type, in.readByte());
    }

    @Test
    void streamIsStartFramesEnd() throws IOException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        DataOutputStream out = new DataOutputStream(bytes);
        byte[] rgba = {1, 2, 3, 4, 5, 6, 7, 8};
        FrameStream.writeStart(out, 7, 100, 2, 2, 1);
        FrameStream.writeFrame(out, new FrameStream.Frame(102, 40, 1.0F, 55, rgba));
        FrameStream.writeEnd(out);

        DataInputStream in = new DataInputStream(new ByteArrayInputStream(bytes.toByteArray()));
        assertHeader(in, FrameStream.START);
        assertEquals(7, in.readLong());
        assertEquals(100, in.readLong());
        assertEquals(2, in.readInt());
        assertEquals(2, in.readInt());
        assertEquals(1, in.readInt());
        assertHeader(in, FrameStream.FRAME);
        assertEquals(102, in.readLong());
        assertEquals(40, in.readLong());
        assertEquals(1.0F, in.readFloat());
        assertEquals(55, in.readLong());
        assertEquals(rgba.length, in.readInt());
        byte[] pixels = new byte[rgba.length];
        in.readFully(pixels);
        assertArrayEquals(rgba, pixels);
        assertHeader(in, FrameStream.END);
        assertEquals(-1, in.read());
    }

    @Test
    void settingsParseAndRejectInvalidValues() {
        JsonObject payload = JsonParser.parseString(
                "{\"episode\":3,\"port\":40001,\"every_ticks\":2,\"width\":426,\"height\":240,\"queue_frames\":64}")
                .getAsJsonObject();
        assertEquals(new RecordingSettings(3, 40001, 2, 426, 240, 64), RecordingSettings.fromJson(payload));

        payload.addProperty("width", 425);
        assertEquals("invalid_recording_width",
                assertThrows(IllegalArgumentException.class, () -> RecordingSettings.fromJson(payload)).getMessage());
        payload.addProperty("width", 426);
        payload.addProperty("every_ticks", 0);
        assertEquals("invalid_every_ticks",
                assertThrows(IllegalArgumentException.class, () -> RecordingSettings.fromJson(payload)).getMessage());
        payload.remove("every_ticks");
        assertEquals("invalid_recording_settings",
                assertThrows(IllegalArgumentException.class, () -> RecordingSettings.fromJson(payload)).getMessage());
    }
}
