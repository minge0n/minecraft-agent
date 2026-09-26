package com.mcbot;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import java.util.Set;

public record PlayerAction(
        boolean forward,
        boolean back,
        boolean left,
        boolean right,
        boolean jump,
        boolean sneak,
        boolean sprint,
        boolean attack,
        boolean use,
        float yawDelta,
        float pitchDelta,
        int hotbar) {
    public static final float MAX_CAMERA_DELTA_DEGREES = 45.0F;
    public static final int KEEP_HOTBAR_SLOT = -1;
    public static final int HOTBAR_SLOTS = 9;

    public static final PlayerAction NOOP =
            new PlayerAction(false, false, false, false, false, false, false, false, false, 0, 0, KEEP_HOTBAR_SLOT);
    public static final PlayerAction FORWARD =
            new PlayerAction(true, false, false, false, false, false, false, false, false, 0, 0, KEEP_HOTBAR_SLOT);

    static final Set<String> FIELDS = Set.of(
            "forward", "back", "left", "right", "jump", "sneak", "sprint", "attack", "use",
            "yaw_delta", "pitch_delta", "hotbar");

    static PlayerAction fromJson(JsonObject json) {
        for (String key : json.keySet()) {
            if (!FIELDS.contains(key)) {
                throw new IllegalArgumentException("unknown_action_field_" + key);
            }
        }
        for (String key : FIELDS) {
            if (!json.has(key)) {
                throw new IllegalArgumentException("missing_action_field_" + key);
            }
        }
        return new PlayerAction(
                button(json, "forward"),
                button(json, "back"),
                button(json, "left"),
                button(json, "right"),
                button(json, "jump"),
                button(json, "sneak"),
                button(json, "sprint"),
                button(json, "attack"),
                button(json, "use"),
                cameraDelta(json, "yaw_delta"),
                cameraDelta(json, "pitch_delta"),
                hotbarSlot(json));
    }

    private static boolean button(JsonObject json, String key) {
        JsonElement value = json.get(key);
        if (!(value instanceof JsonPrimitive primitive) || !primitive.isBoolean()) {
            throw new IllegalArgumentException("action_field_not_boolean_" + key);
        }
        return primitive.getAsBoolean();
    }

    private static float cameraDelta(JsonObject json, String key) {
        JsonElement value = json.get(key);
        if (!(value instanceof JsonPrimitive primitive) || !primitive.isNumber()) {
            throw new IllegalArgumentException("action_field_not_number_" + key);
        }
        double degrees = primitive.getAsDouble();
        if (!Double.isFinite(degrees) || Math.abs(degrees) > MAX_CAMERA_DELTA_DEGREES) {
            throw new IllegalArgumentException("action_field_out_of_range_" + key);
        }
        return (float) degrees;
    }

    private static int hotbarSlot(JsonObject json) {
        JsonElement value = json.get("hotbar");
        if (!(value instanceof JsonPrimitive primitive) || !primitive.isNumber()) {
            throw new IllegalArgumentException("action_field_not_number_hotbar");
        }
        double slot = primitive.getAsDouble();
        if (slot != Math.rint(slot) || slot < KEEP_HOTBAR_SLOT || slot >= HOTBAR_SLOTS) {
            throw new IllegalArgumentException("action_field_out_of_range_hotbar");
        }
        return (int) slot;
    }
}
