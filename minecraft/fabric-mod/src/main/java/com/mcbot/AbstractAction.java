package com.mcbot;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import java.util.Locale;

// The abstract factor of player-action-v3 (docs/decisions/actions.md). It abstracts
// GUI mouse work only: the agent still chooses the item and the slot. Logical slots:
// 0-8 hotbar, 9-35 main inventory, 36-39 armor (feet, legs, chest, head), 40 offhand.
public record AbstractAction(Kind kind, int source, int destination, int item) {
    public enum Kind { NONE, MOVE_ITEM, CRAFT }

    public static final int LOGICAL_SLOTS = 41;
    public static final int FIRST_ARMOR_SLOT = 36;
    public static final int OFFHAND_SLOT = 40;
    public static final AbstractAction NONE = new AbstractAction(Kind.NONE, -1, -1, -1);

    static AbstractAction moveItem(int source, int destination) {
        return new AbstractAction(Kind.MOVE_ITEM, source, destination, -1);
    }

    static AbstractAction craft(int item) {
        return new AbstractAction(Kind.CRAFT, -1, -1, item);
    }

    boolean isNone() {
        return kind == Kind.NONE;
    }

    // Reads the optional `abstract` object of a STEP payload. `itemTypes` is the size of
    // the item registry; a craft target outside it is a malformed request.
    static AbstractAction fromJson(JsonElement element, int itemTypes) {
        if (element == null || element.isJsonNull()) {
            return NONE;
        }
        if (!element.isJsonObject()) {
            throw new IllegalArgumentException("abstract_not_object");
        }
        JsonObject json = element.getAsJsonObject();
        JsonElement kindValue = json.get("kind");
        if (!(kindValue instanceof JsonPrimitive primitive) || !primitive.isString()) {
            throw new IllegalArgumentException("abstract_kind_missing");
        }
        Kind kind;
        try {
            kind = Kind.valueOf(primitive.getAsString().toUpperCase(Locale.ROOT));
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("abstract_kind_unknown");
        }
        return switch (kind) {
            case NONE -> {
                expectKeys(json, 1);
                yield NONE;
            }
            case MOVE_ITEM -> {
                expectKeys(json, 3);
                yield moveItem(integer(json, "source", LOGICAL_SLOTS), integer(json, "destination", LOGICAL_SLOTS));
            }
            case CRAFT -> {
                expectKeys(json, 2);
                yield craft(integer(json, "item", itemTypes));
            }
        };
    }

    private static void expectKeys(JsonObject json, int count) {
        if (json.size() != count) {
            throw new IllegalArgumentException("abstract_fields");
        }
    }

    private static int integer(JsonObject json, String key, int bound) {
        JsonElement value = json.get(key);
        if (!(value instanceof JsonPrimitive primitive) || !primitive.isNumber()) {
            throw new IllegalArgumentException("abstract_field_not_number_" + key);
        }
        double number = primitive.getAsDouble();
        if (number != Math.rint(number) || number < 0 || number >= bound) {
            throw new IllegalArgumentException("abstract_field_out_of_range_" + key);
        }
        return (int) number;
    }

    JsonObject toJson() {
        JsonObject json = new JsonObject();
        json.addProperty("kind", kind.name().toLowerCase(Locale.ROOT));
        switch (kind) {
            case MOVE_ITEM -> {
                json.addProperty("source", source);
                json.addProperty("destination", destination);
            }
            case CRAFT -> json.addProperty("item", item);
            case NONE -> {}
        }
        return json;
    }
}
