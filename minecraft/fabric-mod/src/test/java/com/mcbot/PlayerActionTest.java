package com.mcbot;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.junit.jupiter.api.Test;

class PlayerActionTest {
    private static JsonObject noop() {
        return JsonParser.parseString("""
                {"forward": false, "back": false, "left": false, "right": false, "jump": false,
                 "sneak": false, "sprint": false, "attack": false, "use": false,
                 "yaw_delta": 0, "pitch_delta": 0, "hotbar": -1}
                """).getAsJsonObject();
    }

    private static String rejection(JsonObject json) {
        return assertThrows(IllegalArgumentException.class, () -> PlayerAction.fromJson(json)).getMessage();
    }

    @Test
    void parsesNoopAndSimultaneousControls() {
        assertEquals(PlayerAction.NOOP, PlayerAction.fromJson(noop()));

        JsonObject combined = noop();
        combined.addProperty("forward", true);
        combined.addProperty("jump", true);
        combined.addProperty("attack", true);
        combined.addProperty("yaw_delta", -12.5);
        combined.addProperty("pitch_delta", 3);
        combined.addProperty("hotbar", 8);
        assertEquals(
                new PlayerAction(true, false, false, false, true, false, false, true, false, -12.5F, 3.0F, 8),
                PlayerAction.fromJson(combined));
    }

    @Test
    void rejectsUnknownAndMissingFields() {
        JsonObject unknown = noop();
        unknown.addProperty("craft", "wooden_pickaxe");
        assertEquals("unknown_action_field_craft", rejection(unknown));

        JsonObject missing = noop();
        missing.remove("sprint");
        assertEquals("missing_action_field_sprint", rejection(missing));
    }

    @Test
    void rejectsWrongTypesAndOutOfRangeValues() {
        JsonObject numericButton = noop();
        numericButton.addProperty("jump", 1);
        assertEquals("action_field_not_boolean_jump", rejection(numericButton));

        JsonObject largeTurn = noop();
        largeTurn.addProperty("yaw_delta", 45.5);
        assertEquals("action_field_out_of_range_yaw_delta", rejection(largeTurn));

        JsonObject slot = noop();
        slot.addProperty("hotbar", 9);
        assertEquals("action_field_out_of_range_hotbar", rejection(slot));

        JsonObject fractionalSlot = noop();
        fractionalSlot.addProperty("hotbar", 1.5);
        assertEquals("action_field_out_of_range_hotbar", rejection(fractionalSlot));
    }
}
