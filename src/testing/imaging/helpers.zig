const std = @import("std");

pub fn decodeBase64Alloc(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    const decoded_len = try std.base64.standard.Decoder.calcSizeForSlice(text);
    const bytes = try allocator.alloc(u8, decoded_len);
    errdefer allocator.free(bytes);
    try std.base64.standard.Decoder.decode(bytes, text);
    return bytes;
}

pub fn writeU32le(dst: []u8, value: u32) void {
    dst[0] = @intCast(value & 0xff);
    dst[1] = @intCast((value >> 8) & 0xff);
    dst[2] = @intCast((value >> 16) & 0xff);
    dst[3] = @intCast((value >> 24) & 0xff);
}

pub fn writeVp8lHeader(dst: []u8, width: usize, height: usize, has_alpha: bool) void {
    const bits: u32 =
        @as(u32, @intCast(width - 1)) |
        (@as(u32, @intCast(height - 1)) << 14) |
        (@as(u32, @intFromBool(has_alpha)) << 28);
    writeU32le(dst, bits);
}

pub fn writeBit(dst: []u8, bit_pos: *usize, value: u1) void {
    const byte_index = bit_pos.* / 8;
    const bit_index: u3 = @intCast(bit_pos.* % 8);
    if (value == 1) dst[byte_index] |= @as(u8, 1) << bit_index;
    bit_pos.* += 1;
}

pub fn writeBits(dst: []u8, bit_pos: *usize, value: usize, count: usize) void {
    for (0..count) |i| {
        writeBit(dst, bit_pos, @intCast((value >> @intCast(i)) & 1));
    }
}
