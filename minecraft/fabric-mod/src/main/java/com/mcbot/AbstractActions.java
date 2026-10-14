package com.mcbot;

import com.google.gson.JsonObject;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Holder;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.util.Prediction;
import net.minecraft.util.context.ContextMap;
import net.minecraft.world.entity.EquipmentSlot;
import net.minecraft.world.entity.player.Inventory;
import net.minecraft.world.entity.player.StackedItemContents;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.crafting.CraftingInput;
import net.minecraft.world.item.crafting.CraftingRecipe;
import net.minecraft.world.item.crafting.RecipeHolder;
import net.minecraft.world.item.crafting.RecipeType;
import net.minecraft.world.item.crafting.ShapedRecipe;
import net.minecraft.world.item.crafting.ShapelessRecipe;
import net.minecraft.world.item.crafting.display.SlotDisplayContext;
import net.minecraft.world.item.enchantment.EnchantmentEffectComponents;
import net.minecraft.world.item.enchantment.EnchantmentHelper;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.phys.BlockHitResult;
import net.minecraft.world.phys.HitResult;

// Runs the abstract factor of a step on the server, at the end of the stepped tick
// (docs/decisions/actions.md). An invalid action changes nothing. The result names
// the outcome for logs and tests; it is privileged and never part of an observation.
final class AbstractActions {
    private static final EquipmentSlot[] ARMOR_ORDER = {
        EquipmentSlot.FEET, EquipmentSlot.LEGS, EquipmentSlot.CHEST, EquipmentSlot.HEAD
    };
    private static final int MAIN_INVENTORY_SLOTS = 36;
    private static final int SMALL_GRID = 2;

    private AbstractActions() {}

    static JsonObject run(ServerPlayer player, AbstractAction action) {
        JsonObject result = action.toJson();
        String outcome = switch (action.kind()) {
            case NONE -> "none";
            case MOVE_ITEM -> moveItem(player, action.source(), action.destination());
            case CRAFT -> craft(player, action.item());
        };
        result.addProperty("outcome", outcome);
        result.addProperty("valid", !outcome.startsWith("invalid"));
        return result;
    }

    // ---- move_item ----

    static ItemStack get(ServerPlayer player, int slot) {
        if (slot < MAIN_INVENTORY_SLOTS) {
            return player.getInventory().getItem(slot);
        }
        if (slot < AbstractAction.OFFHAND_SLOT) {
            return player.getItemBySlot(ARMOR_ORDER[slot - AbstractAction.FIRST_ARMOR_SLOT]);
        }
        return player.getItemBySlot(EquipmentSlot.OFFHAND);
    }

    private static void set(ServerPlayer player, int slot, ItemStack stack) {
        if (slot < MAIN_INVENTORY_SLOTS) {
            player.getInventory().setItem(slot, stack);
        } else if (slot < AbstractAction.OFFHAND_SLOT) {
            player.setItemSlot(ARMOR_ORDER[slot - AbstractAction.FIRST_ARMOR_SLOT], stack);
        } else {
            player.setItemSlot(EquipmentSlot.OFFHAND, stack);
        }
    }

    private static String moveItem(ServerPlayer player, int source, int destination) {
        ItemStack from = get(player, source);
        ItemStack to = get(player, destination);
        // Item stacks with equal components compare equal through their component patch.
        InventoryMoves.Stack fromValue = value(from);
        InventoryMoves.Stack toValue = value(to);
        InventoryMoves.Rules rules = new InventoryMoves.Rules() {
            @Override
            public boolean mayPlace(int slot, InventoryMoves.Stack stack) {
                ItemStack real = stack == fromValue ? from : to;
                if (slot >= AbstractAction.FIRST_ARMOR_SLOT && slot < AbstractAction.OFFHAND_SLOT) {
                    return player.isEquippableInSlot(real, ARMOR_ORDER[slot - AbstractAction.FIRST_ARMOR_SLOT]);
                }
                return true;
            }

            @Override
            public boolean mayLeave(int slot, InventoryMoves.Stack stack) {
                ItemStack real = stack == fromValue ? from : to;
                boolean armor = slot >= AbstractAction.FIRST_ARMOR_SLOT && slot < AbstractAction.OFFHAND_SLOT;
                return !(armor && !player.isCreative()
                        && EnchantmentHelper.has(real, EnchantmentEffectComponents.PREVENT_ARMOR_CHANGE));
            }

            @Override
            public int maxStackSize(InventoryMoves.Stack stack) {
                return (stack == fromValue ? from : to).getMaxStackSize();
            }
        };
        InventoryMoves.Result result = InventoryMoves.move(source, destination, fromValue, toValue, rules);
        if (result.changed()) {
            set(player, source, restore(result.source(), from, to));
            set(player, destination, restore(result.destination(), from, to));
            player.inventoryMenu.broadcastChanges();
        }
        return result.outcome().name().toLowerCase(java.util.Locale.ROOT);
    }

    private static InventoryMoves.Stack value(ItemStack stack) {
        if (stack.isEmpty()) {
            return InventoryMoves.Stack.EMPTY;
        }
        return new InventoryMoves.Stack(
                BuiltInRegistries.ITEM.getId(stack.getItem()), stack.getComponentsPatch(), stack.getCount());
    }

    // Maps a result stack back to a real item stack of the same kind.
    private static ItemStack restore(InventoryMoves.Stack value, ItemStack from, ItemStack to) {
        if (value.isEmpty()) {
            return ItemStack.EMPTY;
        }
        ItemStack template = value.item() == BuiltInRegistries.ITEM.getId(from.getItem()) && !from.isEmpty() ? from : to;
        return template.copyWithCount(value.count());
    }

    // ---- craft ----

    private static String craft(ServerPlayer player, int itemId) {
        Item target = BuiltInRegistries.ITEM.byId(itemId);
        ServerLevel level = player.level();
        var context = SlotDisplayContext.fromLevel(level);
        List<RecipeHolder<CraftingRecipe>> candidates = new ArrayList<>();
        for (RecipeHolder<?> holder : level.recipeAccess().getRecipes()) {
            if (holder.value().getType() != RecipeType.CRAFTING) {
                continue;
            }
            @SuppressWarnings("unchecked")
            RecipeHolder<CraftingRecipe> crafting = (RecipeHolder<CraftingRecipe>) holder;
            Optional<ItemStack> result = previewResult(crafting.value(), context);
            if (result.isPresent() && result.get().getItem() == target) {
                candidates.add(crafting);
            }
        }
        if (candidates.isEmpty()) {
            return "invalid_no_recipe";
        }
        candidates.sort(Comparator.comparing(holder -> holder.id().identifier().toString()));
        boolean table = lookingAtCraftingTable(player);
        boolean needsTable = false;
        for (RecipeHolder<CraftingRecipe> holder : candidates) {
            CraftingRecipe recipe = holder.value();
            if (!fitsSmallGrid(recipe) && !table) {
                needsTable = true;
                continue;
            }
            List<Holder<Item>> chosen = new ArrayList<>();
            StackedItemContents contents = new StackedItemContents();
            Inventory inventory = player.getInventory();
            for (int slot = 0; slot < MAIN_INVENTORY_SLOTS; slot++) {
                ItemStack stack = inventory.getItem(slot);
                if (Inventory.isUsableForCrafting(stack)) {
                    contents.accountStack(stack);
                }
            }
            if (!contents.canCraft(recipe, chosen::add)) {
                continue;
            }
            return assemble(player, holder, chosen) ? "crafted" : "invalid_assembly";
        }
        return needsTable ? "invalid_no_crafting_table" : "invalid_missing_ingredients";
    }

    // The result of a shaped or shapeless recipe. Special recipes (dyeing, map
    // copying and so on) have no fixed result and are not craft targets.
    private static Optional<ItemStack> previewResult(CraftingRecipe recipe, ContextMap context) {
        if (recipe.isSpecial() || !(recipe instanceof ShapedRecipe || recipe instanceof ShapelessRecipe)) {
            return Optional.empty();
        }
        return recipe.display().stream()
                .map(display -> display.result().resolveForFirstStack(context))
                .filter(stack -> !stack.isEmpty())
                .findFirst();
    }

    private static boolean fitsSmallGrid(CraftingRecipe recipe) {
        if (recipe instanceof ShapedRecipe shaped) {
            return shaped.getWidth() <= SMALL_GRID && shaped.getHeight() <= SMALL_GRID;
        }
        return recipe.placementInfo().ingredients().size() <= SMALL_GRID * SMALL_GRID;
    }

    private static boolean lookingAtCraftingTable(ServerPlayer player) {
        HitResult hit = player.pick(player.blockInteractionRange(), 1.0F, false);
        if (hit.getType() != HitResult.Type.BLOCK) {
            return false;
        }
        BlockPos pos = ((BlockHitResult) hit).getBlockPos();
        return player.level().getBlockState(pos).is(Blocks.CRAFTING_TABLE);
    }

    // Removes one item per chosen ingredient, assembles the result from the removed
    // items and returns leftovers (empty buckets, bottles) to the player.
    private static boolean assemble(ServerPlayer player, RecipeHolder<CraftingRecipe> holder, List<Holder<Item>> chosen) {
        Inventory inventory = player.getInventory();
        List<ItemStack> taken = new ArrayList<>();
        for (Holder<Item> item : chosen) {
            int slot = findSlot(inventory, item);
            if (slot < 0) {
                for (ItemStack stack : taken) {
                    inventory.placeItemBackInInventory(stack, Prediction.SERVER_ONLY);
                }
                return false;
            }
            taken.add(inventory.removeItem(slot, 1));
        }
        CraftingRecipe recipe = holder.value();
        CraftingInput input = gridInput(recipe, taken);
        if (!recipe.matches(input, player.level())) {
            for (ItemStack stack : taken) {
                inventory.placeItemBackInInventory(stack, Prediction.SERVER_ONLY);
            }
            return false;
        }
        ItemStack result = recipe.assemble(input);
        result.onCraftedBy(player, result.getCount());
        player.triggerRecipeCrafted(holder, taken);
        for (ItemStack remainder : recipe.getRemainingItems(input)) {
            if (!remainder.isEmpty()) {
                give(player, remainder);
            }
        }
        give(player, result);
        player.inventoryMenu.broadcastChanges();
        return true;
    }

    private static int findSlot(Inventory inventory, Holder<Item> item) {
        for (int slot = 0; slot < MAIN_INVENTORY_SLOTS; slot++) {
            ItemStack stack = inventory.getItem(slot);
            if (!stack.isEmpty() && stack.typeHolder().equals(item) && Inventory.isUsableForCrafting(stack)) {
                return slot;
            }
        }
        return -1;
    }

    // Lays the taken items out the way the recipe expects: a shaped recipe uses its own
    // width and height and its pattern positions, a shapeless recipe a single row.
    private static CraftingInput gridInput(CraftingRecipe recipe, List<ItemStack> taken) {
        if (recipe instanceof ShapedRecipe shaped) {
            int width = shaped.getWidth();
            int height = shaped.getHeight();
            List<ItemStack> grid = new ArrayList<>();
            int next = 0;
            for (Optional<net.minecraft.world.item.crafting.Ingredient> ingredient : shaped.getIngredients()) {
                grid.add(ingredient.isPresent() ? taken.get(next++) : ItemStack.EMPTY);
            }
            return CraftingInput.of(width, height, grid);
        }
        return CraftingInput.of(taken.size(), 1, taken);
    }

    private static void give(ServerPlayer player, ItemStack stack) {
        if (!player.getInventory().add(stack) && !stack.isEmpty()) {
            player.drop(stack, false, Prediction.SERVER_ONLY);
        }
    }
}
