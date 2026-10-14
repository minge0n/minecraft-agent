package com.mcbot;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import com.mcbot.InventoryMoves.Outcome;
import com.mcbot.InventoryMoves.Stack;
import java.util.Set;
import org.junit.jupiter.api.Test;

class InventoryMovesTest {
    private static final int DIRT = 10;
    private static final int STONE = 11;
    private static final int HELMET = 20;
    private static final int CURSED_HELMET = 21;
    private static final int HEAD = 39;
    private static final int FEET = 36;
    private static final int OFFHAND = 40;

    // Helmets fit the head slot only; one cursed helmet cannot leave armor; stacks of 64.
    private static final InventoryMoves.Rules RULES = new InventoryMoves.Rules() {
        @Override
        public boolean mayPlace(int slot, Stack stack) {
            boolean armorSlot = slot >= FEET && slot < OFFHAND;
            boolean helmet = stack.item() == HELMET || stack.item() == CURSED_HELMET;
            return !armorSlot || (helmet && slot == HEAD);
        }

        @Override
        public boolean mayLeave(int slot, Stack stack) {
            return !(slot >= FEET && slot < OFFHAND && stack.item() == CURSED_HELMET);
        }

        @Override
        public int maxStackSize(Stack stack) {
            return Set.of(HELMET, CURSED_HELMET).contains(stack.item()) ? 1 : 64;
        }
    };

    private static Stack stack(int item, int count) {
        return new Stack(item, null, count);
    }

    @Test
    void aStackMovesIntoAnEmptySlot() {
        var result = InventoryMoves.move(9, 0, stack(DIRT, 12), Stack.EMPTY, RULES);
        assertEquals(Outcome.MOVED, result.outcome());
        assertEquals(Stack.EMPTY, result.source());
        assertEquals(stack(DIRT, 12), result.destination());
    }

    @Test
    void equalStacksMergeUpToTheMaximum() {
        var result = InventoryMoves.move(9, 0, stack(DIRT, 40), stack(DIRT, 50), RULES);
        assertEquals(Outcome.MERGED, result.outcome());
        assertEquals(stack(DIRT, 26), result.source());
        assertEquals(stack(DIRT, 64), result.destination());
        assertEquals(Outcome.INVALID_FULL,
                InventoryMoves.move(9, 0, stack(DIRT, 1), stack(DIRT, 64), RULES).outcome());
    }

    @Test
    void differentStacksSwap() {
        var result = InventoryMoves.move(2, 5, stack(DIRT, 3), stack(STONE, 7), RULES);
        assertEquals(Outcome.SWAPPED, result.outcome());
        assertEquals(stack(STONE, 7), result.source());
        assertEquals(stack(DIRT, 3), result.destination());
    }

    @Test
    void anyItemFitsTheOffhand() {
        assertEquals(Outcome.MOVED, InventoryMoves.move(0, OFFHAND, stack(DIRT, 64), Stack.EMPTY, RULES).outcome());
    }

    @Test
    void armorSlotsTakeOnlyMatchingArmorAndOneItem() {
        assertEquals(Outcome.MOVED, InventoryMoves.move(4, HEAD, stack(HELMET, 1), Stack.EMPTY, RULES).outcome());
        assertEquals(Outcome.INVALID_DESTINATION,
                InventoryMoves.move(4, FEET, stack(HELMET, 1), Stack.EMPTY, RULES).outcome());
        assertEquals(Outcome.INVALID_DESTINATION,
                InventoryMoves.move(4, HEAD, stack(DIRT, 1), Stack.EMPTY, RULES).outcome());
    }

    @Test
    void aSwapOutOfArmorMustFitBothWays() {
        // The helmet would move into slot 4 and the dirt into the head slot: invalid.
        var result = InventoryMoves.move(4, HEAD, stack(DIRT, 1), stack(HELMET, 1), RULES);
        assertEquals(Outcome.INVALID_DESTINATION, result.outcome());
    }

    @Test
    void cursedArmorCannotLeaveItsSlot() {
        assertEquals(Outcome.INVALID_SOURCE_LOCKED,
                InventoryMoves.move(HEAD, 3, stack(CURSED_HELMET, 1), Stack.EMPTY, RULES).outcome());
    }

    @Test
    void invalidMovesChangeNothing() {
        var empty = InventoryMoves.move(3, 4, Stack.EMPTY, stack(DIRT, 2), RULES);
        assertEquals(Outcome.INVALID_EMPTY_SOURCE, empty.outcome());
        assertFalse(empty.changed());
        assertEquals(stack(DIRT, 2), empty.destination());
        var same = InventoryMoves.move(3, 3, stack(DIRT, 2), stack(DIRT, 2), RULES);
        assertEquals(Outcome.INVALID_SAME_SLOT, same.outcome());
        assertFalse(same.changed());
    }

    @Test
    void stacksWithOtherComponentsDoNotMerge() {
        var named = new Stack(DIRT, "named", 5);
        assertEquals(Outcome.SWAPPED, InventoryMoves.move(1, 2, named, stack(DIRT, 5), RULES).outcome());
    }
}
