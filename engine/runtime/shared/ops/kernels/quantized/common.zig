pub fn writePackedBits(buffer: []u8, bit_index: usize, bit_width: u8, value: u8) void {
    var remaining = bit_width;
    var source: u16 = value;
    var dst_bit_index = bit_index;
    while (remaining > 0) {
        const byte_index = dst_bit_index / 8;
        const bit_offset: u3 = @intCast(dst_bit_index % 8);
        const available: u8 = 8 - @as(u8, bit_offset);
        const chunk_bits: u8 = @min(remaining, available);
        const mask: u16 = (@as(u16, 1) << @intCast(chunk_bits)) - 1;
        const chunk: u8 = @intCast(source & mask);
        buffer[byte_index] |= chunk << bit_offset;
        source >>= @intCast(chunk_bits);
        dst_bit_index += chunk_bits;
        remaining -= chunk_bits;
    }
}

pub fn readPackedBits(buffer: []const u8, bit_index: usize, bit_width: u8) u8 {
    var remaining = bit_width;
    var src_bit_index = bit_index;
    var result: u16 = 0;
    var result_shift: u8 = 0;
    while (remaining > 0) {
        const byte_index = src_bit_index / 8;
        const bit_offset: u3 = @intCast(src_bit_index % 8);
        const available: u8 = 8 - @as(u8, bit_offset);
        const chunk_bits: u8 = @min(remaining, available);
        const mask: u8 = (@as(u8, 1) << @intCast(chunk_bits)) - 1;
        const chunk = (buffer[byte_index] >> bit_offset) & mask;
        result |= @as(u16, chunk) << @intCast(result_shift);
        src_bit_index += chunk_bits;
        result_shift += chunk_bits;
        remaining -= chunk_bits;
    }
    return @intCast(result);
}

pub fn loadQ8Vector16(bytes: []const u8, start: usize) @Vector(16, f32) {
    var values: [16]f32 = undefined;
    inline for (0..16) |lane| {
        const q: i8 = @bitCast(bytes[start + lane]);
        values[lane] = @floatFromInt(q);
    }
    return values;
}

pub fn loadQ6Vector8(bytes: []const u8, start: usize) @Vector(8, f32) {
    const packed24_a = @as(u32, bytes[start]) |
        (@as(u32, bytes[start + 1]) << 8) |
        (@as(u32, bytes[start + 2]) << 16);
    const packed24_b = @as(u32, bytes[start + 3]) |
        (@as(u32, bytes[start + 4]) << 8) |
        (@as(u32, bytes[start + 5]) << 16);

    var values: [8]f32 = undefined;
    inline for (0..4) |lane| {
        const encoded: u8 = @intCast((packed24_a >> (lane * 6)) & 0x3F);
        values[lane] = @floatFromInt(@as(i32, encoded) - 32);
    }
    inline for (0..4) |lane| {
        const encoded: u8 = @intCast((packed24_b >> (lane * 6)) & 0x3F);
        values[4 + lane] = @floatFromInt(@as(i32, encoded) - 32);
    }
    return values;
}
