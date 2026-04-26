const std = @import("std");
const common = @import("common.zig");

pub fn encodeQ8Row(output: []u8, values: []const f32) void {
    var max_abs: f32 = 0.0;
    for (values) |value| max_abs = @max(max_abs, @abs(value));
    const scale: f32 = if (max_abs == 0.0) 1.0 else max_abs / 127.0;
    std.mem.writeInt(u32, output[0..4], @bitCast(scale), .little);
    const inv = 1.0 / scale;
    for (values, 0..) |value, idx| {
        const q = std.math.clamp(@as(i32, @intFromFloat(@round(value * inv))), -127, 127);
        output[4 + idx] = @bitCast(@as(i8, @intCast(q)));
    }
}

pub fn encodeQ6Row(output: []u8, values: []const f32) void {
    @memset(output[4..], 0);
    var max_abs: f32 = 0.0;
    for (values) |value| max_abs = @max(max_abs, @abs(value));
    const scale: f32 = if (max_abs == 0.0) 1.0 else max_abs / 31.0;
    std.mem.writeInt(u32, output[0..4], @bitCast(scale), .little);
    const inv = 1.0 / scale;

    var bit_index: usize = 0;
    for (values) |value| {
        const q = std.math.clamp(@as(i32, @intFromFloat(@round(value * inv))), -32, 31);
        const encoded: u8 = @intCast(q + 32);
        common.writePackedBits(output[4..], bit_index, 6, encoded);
        bit_index += 6;
    }
}

pub fn encodeQ4Row(output: []u8, values: []const f32) void {
    var max_abs: f32 = 0.0;
    for (values) |value| max_abs = @max(max_abs, @abs(value));
    const scale: f32 = if (max_abs == 0.0) 1.0 else max_abs / 7.0;
    std.mem.writeInt(u32, output[0..4], @bitCast(scale), .little);
    const inv = 1.0 / scale;
    @memset(output[4..], 0);
    for (values, 0..) |value, idx| {
        const q = std.math.clamp(@as(i32, @intFromFloat(@round(value * inv))), -8, 7);
        const nibble: u8 = @intCast(q + 8);
        const byte_index = 4 + idx / 2;
        if (idx % 2 == 0) {
            output[byte_index] = nibble;
        } else {
            output[byte_index] |= nibble << 4;
        }
    }
}
