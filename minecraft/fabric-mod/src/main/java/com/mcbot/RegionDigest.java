package com.mcbot;

import java.nio.ByteBuffer;
import java.util.zip.CRC32;
import net.minecraft.core.BlockPos;
import net.minecraft.world.level.BlockGetter;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.phys.AABB;

// Fingerprint of a block region, used by privileged replay diagnostics on both the
// server and the client copy of the world.
record RegionDigest(BlockPos from, BlockPos to) {
    AABB box() {
        return AABB.encapsulatingFullBlocks(from, to);
    }

    long volume() {
        return (long) (Math.abs(to.getX() - from.getX()) + 1)
                * (Math.abs(to.getY() - from.getY()) + 1)
                * (Math.abs(to.getZ() - from.getZ()) + 1);
    }

    long blocks(BlockGetter level) {
        CRC32 crc = new CRC32();
        ByteBuffer buffer = ByteBuffer.allocate(Integer.BYTES);
        for (BlockPos pos : BlockPos.betweenClosed(from, to)) {
            buffer.clear();
            buffer.putInt(Block.getId(level.getBlockState(pos)));
            crc.update(buffer.array());
        }
        return crc.getValue();
    }
}
