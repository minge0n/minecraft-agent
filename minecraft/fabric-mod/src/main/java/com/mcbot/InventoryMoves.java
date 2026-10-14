package com.mcbot;

// The slot rules of move_item (docs/decisions/actions.md) on plain values, so they can
// be tested without a game. A stack is an item id, a component key and a count; id 0
// is empty. The caller supplies the rules that need game data: the maximum stack size
// and whether an item may enter a slot.
final class InventoryMoves {
    enum Outcome { MOVED, MERGED, SWAPPED, INVALID_EMPTY_SOURCE, INVALID_SAME_SLOT,
        INVALID_DESTINATION, INVALID_SOURCE_LOCKED, INVALID_FULL }

    record Stack(int item, Object components, int count) {
        static final Stack EMPTY = new Stack(0, null, 0);

        boolean isEmpty() {
            return item == 0 || count <= 0;
        }

        boolean sameKind(Stack other) {
            return item == other.item && java.util.Objects.equals(components, other.components);
        }

        Stack withCount(int newCount) {
            return newCount <= 0 ? EMPTY : new Stack(item, components, newCount);
        }
    }

    record Result(Outcome outcome, Stack source, Stack destination) {
        boolean changed() {
            return outcome == Outcome.MOVED || outcome == Outcome.MERGED || outcome == Outcome.SWAPPED;
        }
    }

    interface Rules {
        // Whether `stack` may be placed into logical slot `slot`.
        boolean mayPlace(int slot, Stack stack);

        // Whether `stack` may leave logical slot `slot` (Curse of Binding on armor).
        boolean mayLeave(int slot, Stack stack);

        int maxStackSize(Stack stack);
    }

    private InventoryMoves() {}

    static Result move(int source, int destination, Stack from, Stack to, Rules rules) {
        if (from.isEmpty()) {
            return new Result(Outcome.INVALID_EMPTY_SOURCE, from, to);
        }
        if (source == destination) {
            return new Result(Outcome.INVALID_SAME_SLOT, from, to);
        }
        if (!rules.mayLeave(source, from)) {
            return new Result(Outcome.INVALID_SOURCE_LOCKED, from, to);
        }
        if (!rules.mayPlace(destination, from)) {
            return new Result(Outcome.INVALID_DESTINATION, from, to);
        }
        if (to.isEmpty()) {
            int limit = Math.min(rules.maxStackSize(from), slotLimit(destination));
            if (from.count() <= limit) {
                return new Result(Outcome.MOVED, Stack.EMPTY, from);
            }
            return new Result(Outcome.MOVED, from.withCount(from.count() - limit), from.withCount(limit));
        }
        if (from.sameKind(to)) {
            int limit = Math.min(rules.maxStackSize(to), slotLimit(destination));
            int room = limit - to.count();
            if (room <= 0) {
                return new Result(Outcome.INVALID_FULL, from, to);
            }
            int moved = Math.min(room, from.count());
            return new Result(Outcome.MERGED, from.withCount(from.count() - moved), to.withCount(to.count() + moved));
        }
        // A swap moves the destination stack into the source slot, so it must fit there.
        if (!rules.mayLeave(destination, to) || !rules.mayPlace(source, to)) {
            return new Result(Outcome.INVALID_DESTINATION, from, to);
        }
        if (from.count() > slotLimit(destination) || to.count() > slotLimit(source)) {
            return new Result(Outcome.INVALID_DESTINATION, from, to);
        }
        return new Result(Outcome.SWAPPED, to, from);
    }

    // Armor slots hold one item.
    static int slotLimit(int slot) {
        return slot >= AbstractAction.FIRST_ARMOR_SLOT && slot < AbstractAction.OFFHAND_SLOT ? 1 : Integer.MAX_VALUE;
    }
}
