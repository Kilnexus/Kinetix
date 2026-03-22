const std = @import("std");
const types = @import("types.zig");

pub const ImageU8 = types.ImageU8;

pub fn packArgb(alpha: u8, red: u8, green: u8, blue: u8) u32 {
    return (@as(u32, alpha) << 24) |
        (@as(u32, red) << 16) |
        (@as(u32, green) << 8) |
        @as(u32, blue);
}

pub fn updateColorCache(color_cache: ?[]u32, color_cache_bits: usize, pixel: u32) void {
    if (color_cache == null or color_cache_bits == 0) return;
    const index = (@as(usize, 0x1e35a7bd) * @as(usize, pixel)) >> @intCast(32 - color_cache_bits);
    color_cache.?[index] = pixel;
}

pub fn argbToRgb8(allocator: std.mem.Allocator, pixels: []const u32, width: usize, height: usize) !ImageU8 {
    var image = try ImageU8.init(allocator, width, height, 3);
    errdefer image.deinit();
    for (pixels, 0..) |pixel, i| {
        image.data[i * 3] = @intCast((pixel >> 16) & 0xff);
        image.data[i * 3 + 1] = @intCast((pixel >> 8) & 0xff);
        image.data[i * 3 + 2] = @intCast(pixel & 0xff);
    }
    return image;
}

pub fn divRoundUp(num: usize, den: usize) usize {
    return (num + den - 1) / den;
}

pub fn colorIndexWidthBits(color_table_size: usize) usize {
    if (color_table_size <= 2) return 3;
    if (color_table_size <= 4) return 2;
    if (color_table_size <= 16) return 1;
    return 0;
}
