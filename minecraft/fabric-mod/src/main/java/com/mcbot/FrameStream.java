package com.mcbot;

import java.io.DataOutputStream;
import java.io.IOException;

// Wire format of the session recording stream, mirrored by src/minecraft_rl/recording.py.
// Big-endian. Every message starts with MAGIC, VERSION and a type byte. A stream is one
// START, any number of FRAMEs and one END; a stream without END was cut short.
final class FrameStream {
    static final int MAGIC = 0x4D434652; // "MCFR"
    static final short VERSION = 1;
    static final byte START = 0;
    static final byte FRAME = 1;
    static final byte END = 2;

    // One rendered frame, captured in the first frame drawn after client tick clientTick.
    // Pixels are RGBA rows in GPU readback order (bottom row first).
    record Frame(long clientTick, long gameTime, float partialTick, long captureNanos, byte[] rgba) {}

    private FrameStream() {}

    static void writeStart(DataOutputStream out, long episode, long startClientTick, int everyTicks, int width, int height)
            throws IOException {
        header(out, START);
        out.writeLong(episode);
        out.writeLong(startClientTick);
        out.writeInt(everyTicks);
        out.writeInt(width);
        out.writeInt(height);
    }

    static void writeFrame(DataOutputStream out, Frame frame) throws IOException {
        header(out, FRAME);
        out.writeLong(frame.clientTick());
        out.writeLong(frame.gameTime());
        out.writeFloat(frame.partialTick());
        out.writeLong(frame.captureNanos());
        out.writeInt(frame.rgba().length);
        out.write(frame.rgba());
    }

    static void writeEnd(DataOutputStream out) throws IOException {
        header(out, END);
    }

    private static void header(DataOutputStream out, byte type) throws IOException {
        out.writeInt(MAGIC);
        out.writeShort(VERSION);
        out.writeByte(type);
    }
}
