package com.mcbot;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.gson.JsonParser;
import org.junit.jupiter.api.Test;

class AbstractActionTest {
    private static final int ITEMS = 1658;

    private static AbstractAction parse(String json) {
        return AbstractAction.fromJson(JsonParser.parseString(json), ITEMS);
    }

    @Test
    void aMissingAbstractFieldIsNone() {
        assertTrue(AbstractAction.fromJson(null, ITEMS).isNone());
        assertTrue(parse("{\"kind\": \"none\"}").isNone());
    }

    @Test
    void moveAndCraftParseTheirArguments() {
        assertEquals(AbstractAction.moveItem(3, 40), parse("{\"kind\": \"move_item\", \"source\": 3, \"destination\": 40}"));
        assertEquals(AbstractAction.craft(17), parse("{\"kind\": \"craft\", \"item\": 17}"));
    }

    @Test
    void malformedAbstractActionsAreRejected() {
        String[] bad = {
            "{\"kind\": \"smelt\", \"item\": 1}",
            "{\"kind\": \"move_item\", \"source\": 41, \"destination\": 0}",
            "{\"kind\": \"move_item\", \"source\": 0}",
            "{\"kind\": \"craft\", \"item\": 1658}",
            "{\"kind\": \"craft\", \"item\": 1.5}",
            "{\"kind\": \"craft\", \"item\": 3, \"extra\": 1}",
            "[]",
        };
        for (String json : bad) {
            assertThrows(IllegalArgumentException.class, () -> parse(json), json);
        }
    }

    @Test
    void toJsonRoundTrips() {
        AbstractAction move = AbstractAction.moveItem(9, 36);
        assertEquals(move, AbstractAction.fromJson(move.toJson(), ITEMS));
        AbstractAction craft = AbstractAction.craft(5);
        assertEquals(craft, AbstractAction.fromJson(craft.toJson(), ITEMS));
        assertFalse(craft.isNone());
    }
}
