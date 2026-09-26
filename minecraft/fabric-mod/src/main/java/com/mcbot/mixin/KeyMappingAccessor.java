package com.mcbot.mixin;

import net.minecraft.client.KeyMapping;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

@Mixin(KeyMapping.class)
public interface KeyMappingAccessor {
    @Accessor("clickCount")
    int mcbot$getClickCount();

    @Accessor("clickCount")
    void mcbot$setClickCount(int clickCount);
}
